"""REST API + SSE for jobs and settings."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Body, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from n2lh.config import Settings
from n2lh.export import safe_stem, save_outputs

router = APIRouter(prefix="/api")

ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _manager(request: Request):
    return request.app.state.manager


def _settings(request: Request) -> Settings:
    return request.app.state.settings


# ---------------------------------------------------------------------- jobs
@router.post("/jobs")
async def create_job(request: Request, files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "no files uploaded")
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(400, f"unsupported file type: {f.filename}")
    problems = _settings(request).validate()
    if problems:
        raise HTTPException(400, "; ".join(problems))

    tmp = Path(tempfile.mkdtemp(prefix="n2lh-upload-"))
    saved: List[Path] = []
    names: List[str] = []
    for i, f in enumerate(files):
        target = tmp / f"upload-{i:02d}{Path(f.filename or 'x').suffix.lower()}"
        target.write_bytes(await f.read())
        saved.append(target)
        names.append(f.filename or target.name)
    job = _manager(request).start_job(saved, names)
    return {"id": job["id"], "status": job["status"]}


@router.get("/jobs")
def list_jobs(request: Request):
    return _manager(request).store.list_jobs()


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str):
    job = _manager(request).store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@router.post("/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str):
    manager = _manager(request)
    if manager.store.get(job_id) is None:
        raise HTTPException(404, "job not found")
    if not manager.cancel(job_id):
        raise HTTPException(409, "job is not running")
    return {"id": job_id, "status": "cancelling"}


@router.get("/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str, after: int = 0):
    store = _manager(request).store
    if store.get(job_id) is None:
        raise HTTPException(404, "job not found")
    manager = _manager(request)

    async def stream():
        cursor = after
        sent_done = False
        while True:
            events = store.read_events(job_id, after=cursor)
            for ev in events:
                cursor += 1
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("type") in ("job_done", "job_error"):
                    sent_done = True
            if sent_done:
                yield "event: end\ndata: {}\n\n"
                return
            if not manager.is_running(job_id) and cursor > 0 and not events:
                # Job crashed without a terminal event.
                job = store.get(job_id) or {}
                if job.get("status") in ("done", "error", "cancelled"):
                    yield f"data: {json.dumps({'type': 'job_done', 'ok': False})}\n\n"
                    yield "event: end\ndata: {}\n\n"
                    return
            await asyncio.sleep(0.4)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/jobs/{job_id}/pages/{index}/image")
def page_image(request: Request, job_id: str, index: int):
    path = _manager(request).store.job_dir(job_id) / "pages" / f"{index:04d}.png"
    if not path.exists():
        raise HTTPException(404, "page image not found")
    return FileResponse(path, media_type="image/png")


@router.get("/jobs/{job_id}/download/{kind}")
def download(request: Request, job_id: str, kind: str):
    if kind not in ("tex", "pdf"):
        raise HTTPException(400, "kind must be tex or pdf")
    fname = "document.tex" if kind == "tex" else "document.pdf"
    path = _manager(request).store.out_dir(job_id) / fname
    if not path.exists():
        raise HTTPException(404, f"{kind} not ready")
    # Content-Type matters: served as text/html a .tex opens in the window instead
    # of downloading, and the desktop window has no address bar to recover from that.
    media = "application/x-tex" if kind == "tex" else "application/pdf"
    return FileResponse(path, media_type=media, filename=f"{_stem(request, job_id)}.{kind}")


def _stem(request: Request, job_id: str) -> str:
    """A filename based on what the user uploaded, not on the job id."""
    job = _manager(request).store.get(job_id) or {}
    names = job.get("filenames") or []
    return safe_stem(names[0] if names else "", f"notes-{job_id}")


@router.post("/jobs/{job_id}/save")
def save_to_folder(request: Request, job_id: str, payload: dict = Body(...)):
    """Copy the finished document (and its figures) into a folder the user picked."""
    store = _manager(request).store
    if store.get(job_id) is None:
        raise HTTPException(404, "job not found")
    raw = str(payload.get("dir") or "").strip()
    if not raw:
        raise HTTPException(400, "no folder given")
    target = Path(raw).expanduser()
    if not target.is_absolute():
        raise HTTPException(400, "folder must be an absolute path")
    out = store.out_dir(job_id)
    if not (out / "document.tex").exists():
        raise HTTPException(404, "nothing to save yet")
    try:
        written = save_outputs(out, target, _stem(request, job_id))
    except OSError as exc:
        raise HTTPException(400, f"could not write to that folder: {exc}") from exc
    return {"dir": str(target), "written": written}


# ----------------------------------------------------------------- preflight
@router.get("/preflight")
def preflight(request: Request):
    import shutil
    from n2lh import __version__
    from n2lh.recognition.vlm import VLMError, probe_vlm

    settings = _settings(request)
    result = {
        "version": __version__,
        "latex_toolchain": bool(shutil.which("latexmk") or shutil.which("pdflatex")),
        "engine": settings.engine,
        "vlm_configured": settings.vlm_ready,
        "vlm_ok": None,
        "vlm_error": None,
    }
    # Live reachability probe, but only when a VLM is actually in play.
    if settings.vlm_ready and settings.engine in ("vlm", "hybrid"):
        try:
            # A reachability check must stay quick even when vlm_timeout (the
            # per-page cap) is many minutes; the Convert page calls this on load.
            probe_vlm(settings.vlm_base_url, settings.vlm_model,
                      settings.vlm_api_key, min(settings.vlm_timeout, 15),
                      use_proxy=settings.vlm_use_proxy)
            result["vlm_ok"] = True
        except VLMError as exc:
            result["vlm_ok"] = False
            result["vlm_error"] = str(exc)[:400]
    return result


# ------------------------------------------------------------------ settings
@router.get("/settings")
def get_settings(request: Request):
    return _settings(request).masked()


@router.put("/settings")
async def put_settings(request: Request):
    values = await request.json()
    settings = _settings(request)
    if "vlm_api_key" in values and values["vlm_api_key"] in ("", None, "__keep__"):
        values["vlm_api_key"] = "__keep__"
    settings.apply_dict(values)
    problems = settings.validate()
    if problems:
        raise HTTPException(400, "; ".join(problems))
    settings.save(Path(settings.data_dir))
    return settings.masked()
