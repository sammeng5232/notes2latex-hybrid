"""Job orchestration: wire settings -> engines -> pipeline, run in background."""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional

from n2lh.compiler.latex import LatexCompiler
from n2lh.config import Settings
from n2lh.export import safe_stem, save_outputs
from n2lh.pipeline.graph import DocumentPipeline
from n2lh.pipeline.ingest import ingest_files
from n2lh.recognition.base import PageImage, Recognizer
from n2lh.recognition.heuristic import HeuristicRecognizer
from n2lh.recognition.prompts import make_preamble
from n2lh.recognition.vlm import VLMRecognizer
from n2lh.store import JobStore


def build_engines(settings: Settings) -> "tuple[Recognizer, Optional[Recognizer]]":
    """Return (primary, fixer) per the engine mode.

    - heuristic: offline skeleton engine, self-fixing (safe mode). No API.
    - vlm: cloud/local VLM does everything (transcribe + repair).
    - hybrid: offline primary (TrOCR checkpoint if configured, else heuristic)
      with a VLM reserved for repairs/escalation -- API tokens are spent only
      on pages that fail local processing.
    """
    vlm = None
    if settings.vlm_ready:
        vlm = VLMRecognizer(settings.vlm_base_url, settings.vlm_model,
                            settings.vlm_api_key, settings.vlm_timeout,
                            use_proxy=settings.vlm_use_proxy,
                            stall_timeout=settings.vlm_stall_timeout,
                            retries=settings.vlm_retries,
                            thinking=settings.vlm_thinking)

    if settings.engine == "vlm":
        if vlm is None:
            raise ValueError("engine=vlm requires vlm_base_url and vlm_model")
        return vlm, vlm

    if settings.engine == "hybrid":
        primary: Recognizer
        if settings.trocr_ready:
            from n2lh.recognition.trocr import TrOCRRecognizer
            primary = TrOCRRecognizer(settings.trocr_model_dir)
        else:
            primary = HeuristicRecognizer()
        fixer = vlm if (vlm is not None and settings.escalate_to_vlm) else None
        return primary, fixer

    # heuristic
    heur = HeuristicRecognizer()
    return heur, heur


class JobManager:
    def __init__(self, store: JobStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings
        self._threads: dict = {}
        self._cancels: dict = {}
        # This process has no job threads yet, so any job the store still calls
        # pending/running was orphaned by an app that was closed or crashed.
        self.store.mark_interrupted()

    # ---------------------------------------------------------------- public
    def start_job(self, uploaded_paths: List[Path], filenames: List[str]) -> dict:
        job = self.store.create(filenames)
        # Move uploads into the job dir.
        dest_dir = self.store.uploads_dir(job["id"])
        for i, src in enumerate(uploaded_paths):
            target = dest_dir / f"input-{i:02d}{src.suffix.lower()}"
            if src.resolve() != target.resolve():
                target.write_bytes(src.read_bytes())
        self.store.update(job["id"], status="pending")
        self._cancels[job["id"]] = threading.Event()
        thread = threading.Thread(target=self._run_job, args=(job["id"],), daemon=True)
        self._threads[job["id"]] = thread
        thread.start()
        return self.store.get(job["id"])

    def is_running(self, job_id: str) -> bool:
        t = self._threads.get(job_id)
        return bool(t and t.is_alive())

    def cancel(self, job_id: str) -> bool:
        """Ask a running job to stop. In-flight VLM requests are aborted, the
        remaining pages are marked not-attempted, and whatever was already
        transcribed is still assembled into a document."""
        event = self._cancels.get(job_id)
        if event is None or not self.is_running(job_id):
            return False
        event.set()
        self.store.append_event(job_id, {"type": "cancel_requested", "ts": time.time()})
        return True

    # --------------------------------------------------------------- internal
    def _run_job(self, job_id: str) -> None:
        try:
            self._execute(job_id)
        except Exception as exc:  # noqa: BLE001 - surface anything to the UI
            self.store.update(job_id, status="error", error=f"{exc}")
            self.store.append_event(job_id, {
                "type": "job_error", "error": str(exc),
                "trace": traceback.format_exc()[-2000:],
            })

    def _execute(self, job_id: str) -> None:
        store = self.store
        store.update(job_id, status="running")
        store.append_event(job_id, {"type": "job_start", "job": job_id})

        uploads = sorted(store.uploads_dir(job_id).glob("input-*"))
        page_paths = ingest_files(uploads, store.job_dir(job_id) / "pages",
                                  dpi=self.settings.dpi)
        store.append_event(job_id, {"type": "ingested", "pages": len(page_paths)})

        cancel = self._cancels.setdefault(job_id, threading.Event())
        primary, fixer = build_engines(self.settings)
        # Let "Cancel" abort an in-flight request instead of waiting it out.
        for engine in (primary, fixer):
            if engine is not None and hasattr(engine, "cancel_event"):
                engine.cancel_event = cancel
        compiler = LatexCompiler(self.settings.latex_engine,
                                 self.settings.compile_timeout, cleanup_aux=True)
        pipeline = DocumentPipeline(primary, compiler, fixer,
                                    self.settings.max_retries,
                                    self.settings.context_lines,
                                    self.settings.vlm_parallel_workers,
                                    cancel_event=cancel,
                                    preamble=make_preamble(self.settings.doc_font_pt))
        pages = [PageImage(index=i + 1, path=p) for i, p in enumerate(page_paths)]

        def on_event(event: dict) -> None:
            store.append_event(job_id, event)

        try:
            result = pipeline.run(pages, store.out_dir(job_id), on_event=on_event)
        finally:
            # One HTTP client per job: release its sockets when the job ends.
            for engine in {id(e): e for e in (primary, fixer) if e is not None}.values():
                close = getattr(engine, "close", None)   # tolerate duck-typed engines
                if callable(close):
                    close()

        store.set_pages(job_id, [
            {
                "index": p.index,
                "image": f"pages/{p.index:04d}.png",
                "latex": p.latex,
                "status": p.status,
                "attempts": p.attempts,
                "engine": p.engine,
                "compile_ok": p.compile_ok,
            }
            for p in result.pages
        ])
        cancelled = cancel.is_set()
        saved_to = self._auto_save(job_id)
        store.update(
            job_id, status="cancelled" if cancelled else "done",
            # A stopped-early job keeps its (partial) output; say why.
            error=result.aborted if (result.aborted and not cancelled) else None,
            result={
                "ok": result.ok,
                "pages_ok": result.n_ok,
                "pages": len(result.pages),
                "pdf": "out/document.pdf" if result.pdf_path else None,
                "tex": "out/document.tex",
                "aborted": result.aborted,
                "saved_to": saved_to,
            },
        )

    def _auto_save(self, job_id: str) -> Optional[str]:
        """Copy the result into the configured output folder, if there is one.

        Never fatal: the document is already safe in the job folder, so a full
        disk or a folder that has since been removed is reported, not raised.
        """
        folder = (self.settings.output_dir or "").strip()
        if not folder:
            return None
        job = self.store.get(job_id) or {}
        names = job.get("filenames") or []
        stem = safe_stem(names[0] if names else "", f"notes-{job_id}")
        try:
            written = save_outputs(self.store.out_dir(job_id), Path(folder).expanduser(), stem)
        except OSError as exc:
            self.store.append_event(job_id, {"type": "save_failed", "dir": folder,
                                             "error": str(exc)})
            return None
        self.store.append_event(job_id, {"type": "saved", "dir": folder, "written": written})
        return folder
