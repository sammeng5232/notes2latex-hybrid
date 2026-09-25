"""The agentic generate-compile-fix document pipeline.

Design lineage:
- advaypakhale/notes2latex: per-page compile verification, error-log-guided
  repair retries, sequential processing with carried context, unlimited length.
- wz-ml/Math2LaTeX: offline primary recognition (segmentation + local model),
  so pages that compile never touch an API at all.

Engines are injected; the loop itself is pure Python and fully testable with
fakes (see tests/test_graph.py).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image

from n2lh.compiler.latex import CompileResult, LatexCompiler
from n2lh.pipeline.assembler import build_document
from n2lh.pipeline.colors import (apply_color_runs, color_margin_labels, drop_absent_colors,
                                  extract_color_runs, reconcile_colors)
from n2lh.pipeline.context import ContextWindow, EnvironmentTracker
from n2lh.pipeline.autofix import autofix_latex
from n2lh.pipeline.cjk import add_cjk, engine_for, has_cjk
from n2lh.pipeline.figures import render_figures
from n2lh.pipeline.ingest import colored_ink_regions, color_only_crop
from n2lh.pipeline.layout import tidy_layout
from n2lh.pipeline.sanitize import comment_out, sanitize_body
from n2lh.pipeline.style import bold_labels
from n2lh.pipeline.tables import (crop_band, find_table_band, insert_tabular,
                                  replace_tabular, wrap_corner_table)
from n2lh.pipeline.tiles import (STRIP_EDGE, STRIP_VARIANTS, count_lines, join_strips,
                                 judge_strip, line_pitch, needs_strips, plan_strips,
                                 planning_mask, protected_bands, strip_variant, tidy_strip,
                                 unpad_figbox)
from n2lh.recognition.base import PageImage, Recognizer, TranscribeResult
from n2lh.recognition.prompts import PREAMBLE_TEX, fix_user_prompt

log = logging.getLogger("n2lh.pipeline")

EventHandler = Callable[[dict], None]


class RecognizerHung(RuntimeError):
    """Raised when a recognizer call is abandoned after exceeding its hard
    wall-clock deadline without returning at all."""


def _call_with_deadline(fn: Callable[[], TranscribeResult], budget: float) -> TranscribeResult:
    """Run fn() with a hard wall-clock deadline, independent of whatever
    timeout the recognizer itself is (or isn't) configured with.

    Why this exists: a recognizer's own timeout can fail to fire. If the
    underlying network read never returns so much as one byte -- observed in
    practice against a heavily/oddly loaded shared VLM endpoint, where a
    stuck request outlived its configured timeout by well over an hour under
    concurrent load -- the calling Python code never regains control to even
    check a deadline, no matter how the recognizer's own client is
    configured. Calling fn() directly, or waiting on a future with no
    timeout, can therefore hang forever.

    fn() runs in a throwaway daemon thread; we join with a timeout and give
    up on it if it doesn't return in time. Python cannot forcibly stop a
    blocked thread, so a genuinely stuck call is ABANDONED, not killed -- it
    may keep running in the background (and its eventual result, if any, is
    simply discarded) for as long as the underlying I/O stays stuck. That is
    a deliberate, bounded resource leak traded for the caller never hanging
    forever.
    """
    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(budget)
    if t.is_alive():
        raise RecognizerHung(
            f"recognizer call exceeded its {budget:.0f}s deadline and was "
            "abandoned (it may still be running in the background, but "
            "nothing is waiting on it anymore)")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _clean(result: TranscribeResult) -> TranscribeResult:
    """Reduce a recognizer's output to body-only LaTeX (see sanitize_body)."""
    return TranscribeResult(latex=sanitize_body(result.latex), engine=result.engine,
                            confidence=result.confidence, notes=result.notes)


@dataclass
class PageOutcome:
    index: int
    latex: str
    compile_ok: bool
    attempts: int
    engine: str
    status: str                 # ok | fixed | failed
    errors: List[str] = field(default_factory=list)
    engine_failed: bool = False     # page ended because the recognizer itself failed
    engine_succeeded: bool = False  # at least one live recognizer call answered


@dataclass
class DocumentResult:
    pages: List[PageOutcome]
    tex: str
    pdf_path: Optional[Path]
    ok: bool
    aborted: Optional[str] = None   # why the run stopped early, if it did

    @property
    def n_ok(self) -> int:
        return sum(1 for p in self.pages if p.status != "failed")


class DocumentPipeline:
    def __init__(self, recognizer: Recognizer, compiler: LatexCompiler,
                 fixer: Optional[Recognizer] = None,
                 max_retries: int = 3, context_lines: int = 40,
                 parallel_workers: int = 1,
                 cancel_event: Optional[threading.Event] = None,
                 max_consecutive_engine_failures: int = 3,
                 preamble: Optional[str] = None) -> None:
        self.recognizer = recognizer
        self.preamble = preamble or PREAMBLE_TEX
        self._figures_dir: Optional[Path] = None   # set per run(): <outdir>/figures
        self._tables_read: set = set()   # a page's table is re-read at most once
        self._color_runs: Dict = {}      # page index -> colored runs read from crops
        self._cjk = False                # switched on by the first page of CJK
        # Dense pages read as full-resolution strips (see pipeline/tiles.py).
        self._strip_plans: Dict[int, Optional[tuple]] = {}   # page -> plan, or None
        self._strip_text: Dict[int, Dict[int, str]] = {}     # page -> strip -> tidied LaTeX
        self._strip_lock = threading.Lock()
        self._ink_families: Dict[int, Optional[List[str]]] = {}   # page -> colored ink measured
        self._margin_colors: Dict[int, Optional[str]] = {}    # page -> left-margin pen color
        self._strip_failed: Dict[int, set] = {}                # page -> strips that yielded nothing
        self._tiled: set = set()                             # pages whose latest read was strips
        self.compiler = compiler
        self.fixer = fixer          # engine used for log-guided repair passes
        self.max_retries = max(1, max_retries + 1)  # total attempts per page
        self.context_lines = context_lines
        # >1 prefetches every page's FIRST-attempt transcription concurrently
        # (the VLM round-trip, not the local compile step, is the bottleneck --
        # see _prefetch). 1 (default) is fully sequential, matching the
        # original notes2latex design exactly.
        self.parallel_workers = max(1, parallel_workers)
        self.cancel_event = cancel_event
        # Stop the job once this many pages IN A ROW ended because the
        # recognizer itself failed (as opposed to the LaTeX not compiling).
        # An endpoint that is down will fail every remaining page too; without
        # this a 43-page job kept grinding for 8 hours producing nothing.
        self.max_consecutive_engine_failures = max(1, max_consecutive_engine_failures)

    def _cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    # ------------------------------------------------------------------ public
    def run(self, pages: List[PageImage], outdir: Path,
            on_event: Optional[EventHandler] = None) -> DocumentResult:
        outdir.mkdir(parents=True, exist_ok=True)
        # Cropped figures live beside the output; the preamble's \graphicspath finds
        # them from the final compile (cwd = outdir) and from each per-page compile.
        self._figures_dir = outdir / "figures"
        ctx = ContextWindow(self.context_lines)
        envs = EnvironmentTracker()
        outcomes: List[PageOutcome] = []
        body_parts: List[str] = []

        self._emit(on_event, "job_start", total=len(pages))

        prefetched: Dict[int, TranscribeResult] = {}
        engine_streak = 0     # consecutive pages that failed on the recognizer
        if self.parallel_workers > 1 and len(pages) > 1:
            prefetched, engine_streak = self._prefetch(pages, on_event)

        aborted: Optional[str] = None
        cancelled = False
        endpoint_dead = False
        for page in pages:
            if not cancelled and self._cancelled():
                cancelled = True
                aborted = "cancelled by user"
                self._emit(on_event, "job_cancelled")
            if cancelled:
                outcomes.append(self._skipped_page(page, aborted, body_parts, on_event))
                continue

            has_result = page.index in prefetched
            if (not endpoint_dead
                    and engine_streak >= self.max_consecutive_engine_failures):
                # From here on: prefetched results are still compiled and kept,
                # but nothing waits on the recognizer again.
                endpoint_dead = True
                aborted = (f"stopped early: the recognizer failed {engine_streak} times "
                           "in a row (endpoint down or unresponsive); pages that "
                           "still needed it were not attempted")
                self._emit(on_event, "job_aborted", reason=aborted)
            if endpoint_dead and not has_result:
                outcomes.append(self._skipped_page(page, aborted, body_parts, on_event))
                continue

            outcome = self._process_page(page, ctx, envs, body_parts, on_event,
                                         prefetched.get(page.index),
                                         allow_engine=not endpoint_dead)
            outcomes.append(outcome)
            if outcome.engine_failed:
                engine_streak += 1
            elif outcome.engine_succeeded:
                engine_streak = 0
            # Only VERIFIED pages feed the rolling context and the open-
            # environment tracker. A failed page's LaTeX is not part of the
            # compiled document (see _unverified_block), so letting it in would
            # make the next pages' compile inputs disagree with what is really
            # in the body.
            if outcome.compile_ok:
                ctx.push(outcome.latex)
                envs.update(outcome.latex)

        # Final assembled document (closes any dangling environments).
        body = "\n\n".join(body_parts).strip()
        closed, _ = envs.close_for_compile(body)
        tex = build_document(body, preamble=self.preamble, closed_body=closed)
        tex_path = outdir / "document.tex"
        tex_path.write_text(tex, encoding="utf-8")

        final = self.compiler.compile(tex, outdir)
        # Success requires BOTH a compiling final document AND every page
        # verified (a placeholder skeleton would compile but is not success).
        all_verified = all(o.compile_ok for o in outcomes)
        result = DocumentResult(pages=outcomes, tex=tex,
                                pdf_path=final.pdf_path,
                                ok=final.ok and all_verified,
                                aborted=aborted)
        self._emit(on_event, "job_done", ok=result.ok,
                   pages_ok=result.n_ok, pages=len(outcomes),
                   pdf=str(final.pdf_path) if final.pdf_path else None,
                   tex=str(tex_path), aborted=aborted)
        return result

    # ----------------------------------------------------------------- private
    # Last-resort wall-clock budget for a SINGLE recognizer call (prefetch, the
    # first live attempt, every repair pass). The VLM client enforces its own
    # stall/total limits and aborts the connection itself; this only exists so
    # a recognizer that cannot (a custom engine, a wedged local model) can never
    # hang the whole job. It never undercuts a recognizer's own declared
    # ``hard_budget``, or it would abandon a call in the middle of its retries.
    RECOGNIZER_CALL_BUDGET = 900

    def _call_budget(self, engine: Recognizer) -> float:
        return max(float(self.RECOGNIZER_CALL_BUDGET),
                   float(getattr(engine, "hard_budget", 0) or 0))

    # ------------------------------------------------------------ dense pages
    def _strip_plan(self, engine: Recognizer, page: PageImage):
        """``(strips, strip_paths, expected_lines, page_height)`` for a dense page, else None.

        Only for an engine that reads strips. Computed once per page and kept, so
        a page whose prefetch failed is re-read on the same cuts, and the strips
        that did come back are not asked for again.
        """
        if not getattr(engine, "reads_strips", False):
            return None
        with self._strip_lock:
            if page.index in self._strip_plans:
                return self._strip_plans[page.index]
        plan = None
        try:
            with Image.open(page.path) as opened:
                image = opened.convert("RGB")
            mask = planning_mask(image)
            pitch = line_pitch(mask)
            if needs_strips(mask, pitch):
                strips = plan_strips(mask, pitch, keep_whole=protected_bands(image, mask, pitch))
                if len(strips) >= 2:
                    paths, expected = [], []
                    for s in strips:
                        path = page.path.parent / f".strip-p{page.index:04d}-{s.index:02d}.png"
                        crop = image.crop((0, s.top, image.width, s.bottom))
                        crop.save(path, "PNG")
                        paths.append(path)
                        expected.append(count_lines(planning_mask(crop), pitch))
                    plan = (strips, paths, expected, image.height)
        except Exception as exc:  # noqa: BLE001 - fall back to a whole-page read
            log.warning("page %d: could not plan strips: %s", page.index, exc)
            plan = None
        with self._strip_lock:
            self._strip_plans[page.index] = plan
        return plan

    def _read_page(self, engine: Recognizer, page: PageImage, tail: str,
                   open_envs: List[str], on_event: Optional[EventHandler],
                   stop: Optional[Callable[[], bool]] = None,
                   pooled: bool = False) -> TranscribeResult:
        """A page's first transcription: whole, or as full-resolution strips.

        A dense page is read as strips (see pipeline/tiles.py). One page deadline
        covers all of its strips -- the same bound a whole-page call has -- and a
        strip that would have to start with too little of it left fails instead
        of being sent to be abandoned. Each strip is sanitized on its own before
        the join: a strip that answers with a whole \\documentclass document would
        otherwise make sanitizing the joined page cut away every other strip.
        ``pooled`` reads strips concurrently (the live path, pages one at a time);
        without it they are read in order (inside a prefetch worker, where pages
        are already concurrent), so requests in flight stay at the worker count.
        """
        plan = self._strip_plan(engine, page)
        if plan is None:
            self._tiled.discard(page.index)
            return _call_with_deadline(lambda: engine.transcribe(page, tail, open_envs),
                                       self._call_budget(engine))

        strips, paths, expected, height = plan
        total = len(strips)
        deadline = time.monotonic() + self._call_budget(engine)
        with self._strip_lock:
            done = self._strip_text.setdefault(page.index, {})
            missing = [s for s in strips if s.index not in done]
        self._emit(on_event, "strips_planned", page=page.index, strips=total,
                   cached=total - len(missing))

        failed = threading.Event()      # one strip failed: the page fails, start no more
        first_error: List[BaseException] = []

        def read(strip) -> None:
            if failed.is_set():
                raise RuntimeError("strip %d/%d not started: another strip failed"
                                   % (strip.index, total))
            if (stop is not None and stop()) or self._cancelled():
                raise RuntimeError("stopped before strip %d/%d" % (strip.index, total))
            try:
                read_one(strip)
            except Exception as exc:
                with self._strip_lock:
                    first_error.append(exc)
                failed.set()
                raise

        def read_one(strip) -> None:
            """Read a strip, checking each answer and re-reading a bad one.

            An answer that fails judge_strip (lines skipped, garbage) is read
            again from a padded or rescaled copy of the strip -- the endpoint
            answers an identical request identically -- and the best answer
            is kept. A strip that yields no answer at all fails the page the
            first time (the retry pass asks for it alone) and is left as a
            visible gap the second time, rather than costing the whole page.
            """
            started = time.monotonic()
            source = paths[strip.index - 1]
            want = expected[strip.index - 1]
            best: Optional[tuple] = None          # (score, text, verdict)
            error: Optional[BaseException] = None
            for variant in range(len(STRIP_VARIANTS)):
                left = deadline - time.monotonic()
                if left < 30:
                    error = error or RecognizerHung(
                        f"no time left in the page budget for strip {strip.index}/{total}")
                    break
                if (stop is not None and stop()) or self._cancelled():
                    raise RuntimeError("stopped during strip %d/%d" % (strip.index, total))
                image = source
                if variant:
                    image = source.with_name(f"{source.stem}-v{variant}.png")
                    with Image.open(source) as opened:
                        strip_variant(opened.convert("RGB"), variant).save(image, "PNG")

                def retried(attempt: int, error: str, variant=variant) -> None:
                    self._emit(on_event, "strip_retry", page=page.index, strip=strip.index,
                               strips=total, attempt=attempt, variant=variant,
                               error=error[:200])

                try:
                    raw = _call_with_deadline(
                        lambda image=image, variant=variant: engine.transcribe_strip(
                            image, strip.index, total, max_edge=STRIP_EDGE,
                            context_tail=tail if strip.index == 1 else "",
                            open_environments=open_envs if strip.index == 1 else (),
                            on_retry=retried, variant=variant),
                        left)
                except RecognizerHung:
                    raise
                except Exception as exc:  # noqa: BLE001 - the next variant may read
                    if self._cancelled():
                        raise
                    error = exc
                    self._emit(on_event, "strip_reread", page=page.index, strip=strip.index,
                               strips=total, variant=variant, reason=str(exc)[:200])
                    continue
                raw = sanitize_body(raw or "")
                if variant:
                    with Image.open(source) as opened:
                        raw = unpad_figbox(raw, variant, *opened.size)
                ok, score, why = judge_strip(raw, want)
                # Garbage scores 0 and is never kept, even as a last resort: a
                # read that skipped lines is still the page's own text, garbage
                # is not.
                if score > 0 and (best is None or score > best[0]):
                    best = (score, raw, why)
                if ok:
                    break
                if score == 0:
                    error = RuntimeError(f"unusable answer: {why}")
                self._emit(on_event, "strip_reread", page=page.index, strip=strip.index,
                           strips=total, variant=variant, reason=why[:200])

            if best is None:
                with self._strip_lock:
                    again = strip.index in self._strip_failed.setdefault(page.index, set())
                    self._strip_failed[page.index].add(strip.index)
                if not again:
                    raise error or RuntimeError(f"strip {strip.index}/{total} not read")
                text = ("\\par\\noindent\\textit{[Rows %d--%d of this page could not be "
                        "read: the recognizer failed on them twice.]}" % (strip.top, strip.bottom))
                self._emit(on_event, "strip_unreadable", page=page.index, strip=strip.index,
                           strips=total, error=str(error)[:200])
            else:
                score, raw, why = best
                text = tidy_strip(raw, strip, height)
                if why != "ok":
                    log.warning("page %d strip %d/%d: kept the best of its reads, which %s",
                                page.index, strip.index, total, why)
                    self._emit(on_event, "strip_doubtful", page=page.index, strip=strip.index,
                               strips=total, reason=why[:200])
            with self._strip_lock:
                done[strip.index] = text
            self._emit(on_event, "strip_done", page=page.index, strip=strip.index,
                       strips=total, chars=len(text),
                       seconds=round(time.monotonic() - started, 1))

        if pooled and len(missing) > 1:
            pool = ThreadPoolExecutor(max_workers=min(len(missing), self.parallel_workers))
            try:
                for future in as_completed([pool.submit(read, s) for s in missing]):
                    if future.exception() is not None:    # the first failure ends the page
                        raise first_error[0] if first_error else future.exception()
            finally:
                # Strips already in flight are waited for (each is bounded by the
                # page deadline): their answers are kept, so the retry pass does
                # not send the same strip a second time while the first is running.
                pool.shutdown(wait=True, cancel_futures=True)
        else:
            for strip in missing:
                read(strip)

        with self._strip_lock:
            parts = [done[s.index] for s in strips]
            self._tiled.add(page.index)
        return TranscribeResult(latex=join_strips(parts, strips, height),
                                engine=getattr(engine, "name", "?"),
                                notes=f"read as {total} full-resolution strips")

    def _strips_pending(self, page: PageImage) -> bool:
        """True when some of the page's strips were read and some were not."""
        with self._strip_lock:
            plan = self._strip_plans.get(page.index)
            done = self._strip_text.get(page.index, {})
            return plan is not None and 0 < len(done) < len(plan[0])

    def _is_tiled(self, page: PageImage) -> bool:
        """Whether the page's latest transcription was read as strips."""
        return page.index in self._tiled

    def _prefetch(self, pages: List[PageImage], on_event: Optional[EventHandler]
                  ) -> "tuple[Dict[int, TranscribeResult], int]":
        """Fire off every page's first-attempt transcription concurrently.

        Returns ``(results, trailing_failure_streak)``.

        Trade-off: since pages are dispatched together, none of them can see
        the rolling context / open-environment state that earlier pages would
        normally have produced -- each is transcribed as if it opened the
        document (transcribe(page, "", [])). That mainly costs cross-page
        notation consistency and environments left open across a page break.
        If a page then fails to compile, the ordinary sequential repair pass
        in _process_page still runs with the REAL context and the actual
        compiler errors, so most such issues self-heal there. A page whose
        prefetch call raised -- or never returns at all within its call
        budget (see _call_with_deadline) -- is simply left out of the returned
        dict, and _process_page transparently falls back to a live call.

        If the recognizer keeps failing (``max_consecutive_engine_failures`` in
        a row, in completion order) the remaining not-yet-started pages are
        skipped instead of each burning its own full retry budget against a
        dead endpoint.
        """
        results: Dict[int, TranscribeResult] = {}
        total = len(pages)
        self._emit(on_event, "prefetch_start", total=total,
                   workers=self.parallel_workers)
        give_up = threading.Event()
        skipped = "skipped: the recognizer is failing or the job was cancelled"

        def work(page: PageImage):
            if give_up.is_set() or self._cancelled():
                return page.index, None, skipped
            try:
                result = self._read_page(
                    self.recognizer, page, "", [], on_event,
                    stop=lambda: give_up.is_set() or self._cancelled(), pooled=False)
                # Crop the figures here, in the worker, so the extra "locate the
                # figures" request runs in parallel with the other pages.
                return page.index, self._finish(page, result), None
            except Exception as exc:  # noqa: BLE001 - surfaced as a per-page event
                return page.index, None, str(exc)

        streak = 0
        # Every work() call is internally bounded (raises rather than blocking
        # forever), so plain as_completed() with no polling logic is safe here.
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            futures = [pool.submit(work, page) for page in pages]
            done = 0
            for future in as_completed(futures):
                index, result, error = future.result()
                done += 1
                if result is not None:
                    results[index] = result
                    streak = 0
                elif error != skipped:
                    streak += 1
                    if streak >= self.max_consecutive_engine_failures:
                        give_up.set()
                self._emit(on_event, "prefetched", page=index, ok=result is not None,
                           error=error, done=done, total=total)
        return results, streak

    def _process_page(self, page: PageImage, ctx: ContextWindow,
                      envs: EnvironmentTracker, body_parts: List[str],
                      on_event: Optional[EventHandler],
                      prefetched: Optional[TranscribeResult] = None,
                      allow_engine: bool = True) -> PageOutcome:
        """Transcribe + compile one page, repairing on failure.

        ``allow_engine=False`` (recognizer declared dead) still lets a
        prefetched result be compiled, but makes no further engine calls.
        """
        self._emit(on_event, "page_start", page=page.index)
        tail = ctx.tail(self.context_lines)
        open_envs = envs.open_environments

        result: Optional[TranscribeResult] = None
        errors: List[str] = []
        compile_ok = False
        engine_failed = False
        engine_succeeded = False
        attempt = 0
        appended = False  # is result.latex currently in body_parts?
        last_signature: Optional[str] = None   # first compile error of the previous attempt
        repeats = 0                            # consecutive attempts that hit that same error
        auto_fixed = False                     # a deterministic fix (autofix) made it compile

        while attempt < self.max_retries:
            if self._cancelled():
                errors = ["cancelled"]
                break
            use_prefetch = attempt == 0 and prefetched is not None
            if not use_prefetch and not allow_engine:
                break  # recognizer is known-dead: don't wait on it again
            attempt += 1
            engine_used = "?"

            try:
                if use_prefetch:
                    result = self._finish(page, prefetched)
                    engine_used = prefetched.engine
                elif result is None:
                    # No usable LaTeX yet (first live call, or an earlier engine
                    # failure): ask for a plain transcription. Sending a "fix
                    # this LaTeX" prompt with NOTHING to fix made the model
                    # answer with a whole standalone document.
                    # Strips already read are kept, so a page left half read goes
                    # back to the engine that read them for the rest.
                    engine = (self.recognizer if attempt == 1 or self._strips_pending(page)
                              else (self.fixer or self.recognizer))
                    result = self._finish(page, self._read_page(
                        engine, page, tail, open_envs, on_event, pooled=True))
                    engine_used = engine.name
                    engine_succeeded = True
                else:
                    fixer = self.fixer or self.recognizer
                    # A page read as strips is repaired from its LaTeX alone: the
                    # only image a repair could send is the whole page at the
                    # resolution that made the model write from memory.
                    tiled = self._is_tiled(page)
                    guidance = fix_user_prompt(result.latex, errors, repeats, text_only=tiled)

                    def repair(fixer=fixer, guidance=guidance, tiled=tiled):
                        if tiled:
                            fixed = fixer.repair_text(guidance)
                            if fixed is not None:
                                return fixed
                        return fixer.transcribe(page, tail, open_envs, guidance=guidance)

                    result = self._finish(page, _call_with_deadline(
                        repair, self._call_budget(fixer)))
                    engine_used = fixer.name
                    engine_succeeded = True
                engine_failed = False
            except Exception as exc:  # engine failure (e.g. VLM unreachable, or a call
                                       # abandoned after its budget -- _call_with_deadline)
                errors = [f"engine error: {exc}"]
                engine_failed = True
                self._emit(on_event, "engine_error", page=page.index,
                           attempt=attempt, error=str(exc))
                # Escalate to a *different* engine once. Re-asking the very same
                # engine (engine=vlm) is pointless: its client already retried --
                # unless the page was read as strips and only some came back: then
                # a second pass asks for the missing strips alone.
                if attempt == 1 and (self._strips_pending(page) or (
                        self.fixer is not None and self.fixer is not self.recognizer)):
                    continue
                break

            self._emit(on_event, "transcribed", page=page.index,
                       attempt=attempt, engine=engine_used)

            body_parts.append(result.latex)
            appended = True
            cres: CompileResult = self._compile_with(page, body_parts, envs)

            if not cres.ok:
                # Mechanical faults (a bare &, an unclosed environment, aligned in
                # text mode) don't need a model: fix them here and recompile.
                fixed_latex, changes = autofix_latex(result.latex)
                if changes and fixed_latex != result.latex:
                    body_parts[-1] = fixed_latex
                    cres2 = self._compile_with(page, body_parts, envs)
                    self._emit(on_event, "autofix", page=page.index, attempt=attempt,
                               changes=changes, ok=cres2.ok)
                    progressed = (not cres2.ok and cres2.errors and cres.errors
                                  and self._signature(cres2.errors[0]) != self._signature(cres.errors[0]))
                    if cres2.ok or progressed:
                        # Keep it: it builds, or it got past the first error so the
                        # model repair starts from a better base.
                        result = TranscribeResult(latex=fixed_latex, engine=result.engine,
                                                  confidence=result.confidence, notes=result.notes)
                        cres = cres2
                        auto_fixed = auto_fixed or cres2.ok
                    else:
                        body_parts[-1] = result.latex      # no help: put the original back

            if cres.ok:
                compile_ok = True
                errors = []
                break

            # Roll back this attempt's contribution before repairing.
            body_parts.pop()
            appended = False
            errors = [str(e) for e in cres.errors] or ["unknown compile error"]
            # The same error again after a repair means "fix it" isn't working:
            # count it so the next repair prompt changes approach (fix_user_prompt).
            signature = self._signature(errors[0])   # same error, moved line -> same
            repeats = repeats + 1 if signature == last_signature else 0
            last_signature = signature
            self._emit(on_event, "compile_fail", page=page.index,
                       attempt=attempt, errors=errors[:3])

        had_result = result is not None
        if not compile_ok and result is None:
            result = TranscribeResult(latex=f"% page {page.index}: recognition failed",
                                      engine="none")
        if compile_ok and not appended:
            body_parts.append(result.latex)
        elif not compile_ok:
            # Keep the unverified output reviewable, but NEVER as live LaTeX:
            # it is known not to compile, and everything after it is compiled
            # on top of it.
            body_parts.append(self._unverified_block(
                page.index, result.latex,
                "LaTeX did not compile" if had_result else "no transcription"))

        status = ("ok" if (compile_ok and attempt == 1 and not auto_fixed)
                  else ("fixed" if compile_ok else "failed"))
        outcome = PageOutcome(index=page.index, latex=result.latex,
                              compile_ok=compile_ok, attempts=attempt,
                              engine=result.engine, status=status, errors=errors,
                              engine_failed=engine_failed and not compile_ok,
                              engine_succeeded=engine_succeeded)
        self._emit(on_event, "page_done", page=page.index, status=status,
                   attempts=attempt, engine=result.engine,
                   compile_ok=compile_ok)
        return outcome

    def _compile_with(self, page: PageImage, body_parts: List[str],
                      envs: EnvironmentTracker) -> CompileResult:
        """Compile the accumulated body (as it stands) in this page's work dir."""
        body = "\n\n".join(body_parts).strip()
        closed, _ = envs.close_for_compile(body)
        tex = build_document(body, preamble=self.preamble, closed_body=closed)
        return self.compiler.compile(tex, page.path.parent / f".compile-p{page.index:04d}")

    @staticmethod
    def _signature(error) -> str:
        """An error message without its line number, to recognise 'the same error'."""
        return re.sub(r"\s*\(line \d+\)", "", str(error))

    def _finish(self, page: PageImage, result: TranscribeResult) -> TranscribeResult:
        """Clean a recognizer result, bold its note labels and turn ``\\figbox`` into pictures."""
        result = _clean(result)
        latex = result.latex
        if "\\figbox" in latex and self._figures_dir is not None:
            latex = render_figures(latex, page.path, self._figures_dir, page.index,
                                   locator=lambda: self._locate(page))
        self._use_cjk_if_needed(latex)
        latex = self._reread_table(page, latex)
        latex = self._apply_colors(page, latex)
        latex, bolded = bold_labels(latex)
        if bolded:
            log.debug("page %d: bolded %d label(s)", page.index, bolded)
        latex, tidied = tidy_layout(latex)
        if tidied:
            log.debug("page %d: %s", page.index, "; ".join(tidied))
        if latex != result.latex:
            result = TranscribeResult(latex=latex, engine=result.engine,
                                      confidence=result.confidence, notes=result.notes)
        return result

    def _use_cjk_if_needed(self, latex: str) -> None:
        """Move the whole job to XeLaTeX once any page turns out to be CJK.

        pdflatex cannot draw a Chinese character, so a page of Chinese notes
        fails every repair attempt for a reason no repair can address; see
        n2lh/pipeline/cjk.py. Switching is idempotent and one-way.
        """
        if self._cjk or not has_cjk(latex):
            return
        self._cjk = True
        self.preamble = add_cjk(self.preamble)
        engine = engine_for(getattr(self.compiler, "engine", "pdflatex"))
        if getattr(self.compiler, "engine", None) != engine:
            self.compiler.engine = engine
        log.info("CJK detected: compiling with %s and ctex", engine)

    def _reread_table(self, page: PageImage, latex: str) -> str:
        """Read a ruled table again from a crop of it (best-effort).

        A table is the one thing a whole-page transcription reliably gets wrong,
        because the page is downscaled before it is sent and the cells stop being
        legible; see n2lh/pipeline/tables.py. Two cases:

        - the transcription has the table: swap it for the crop's (more pixels);
        - the transcription DROPPED it (real case: a corner vocabulary table
          was missing entirely from one pass, present from another): read it
          from the crop and put it back.

        Each page is re-read at most once. A page read as strips already had its
        tables read at full resolution -- a strip never cuts through one -- so
        there a crop (sent at the whole-page size limit) could only be worse;
        only a table the strips dropped is fetched.
        """
        reread = getattr(self.recognizer, "transcribe_table", None)
        if (reread is None or page.index in self._tables_read or self._cancelled()):
            return latex
        if self._is_tiled(page) and "begin{tabular}" in latex:
            self._tables_read.add(page.index)
            return latex
        try:
            with Image.open(page.path) as opened:
                image = opened.convert("RGB")
            dark = image.convert("L").point(lambda v: 255 if v < 145 else 0)
            band = find_table_band(dark)
            if band is None:
                return latex
            self._tables_read.add(page.index)
            crop = crop_band(image, band, page.path.parent / f".table-p{page.index:04d}.png")
            table = _call_with_deadline(lambda: reread(crop),
                                        self._call_budget(self.recognizer))
            if not table:
                return latex
            if "begin{tabular}" in latex:
                latex, swapped = replace_tabular(latex, table)
                if swapped:
                    log.info("page %d: table re-read from a crop", page.index)
            else:
                latex, inserted = insert_tabular(latex, table, band, dark.size[1])
                if inserted:
                    log.info("page %d: table missing from the transcription; "
                             "re-read from a crop and inserted", page.index)
        except Exception as exc:  # noqa: BLE001 - the page's own table still stands
            log.warning("page %d: could not re-read the table: %s", page.index, exc)
        return latex

    def _apply_colors(self, page: PageImage, latex: str) -> str:
        """Put colored ink back via a focused re-read of the colored regions.

        The whole-page transcription is unreliable about color in BOTH
        directions (see n2lh/pipeline/colors.py): it drops the colors that
        are there, and it invents ones that are not. So each colored region
        is cropped and re-read with a color-only prompt whatever the
        transcription did: the read's runs are merged into this page's own
        LaTeX, and the transcription's own \\textcolor runs of a color whose
        region read out are kept only when the read corroborates them. The
        reads happen once per page and are cached: a repair pass re-merges
        them against its own latex for free. Best-effort throughout -- a
        failed or empty re-read leaves the page exactly as it was.
        """
        if self._cancelled():
            return latex
        reread = getattr(self.recognizer, "transcribe_colors", None)
        if reread is None:
            return latex
        try:
            cached = self._color_runs.get(page.index)
            if cached is None:
                runs, corner_texts, reads = [], [], {}
                with Image.open(page.path) as opened:
                    image = opened.convert("RGB")
                for i, (box, names) in enumerate(colored_ink_regions(image)):
                    # Each region is best-effort on its own: one flaky crop
                    # read (the endpoint sometimes answers nothing but a
                    # tool_calls finish) must not cost the other regions.
                    try:
                        # Only the colored ink: black words run through the
                        # region and must not be readable, or the model marks
                        # them colored (see color_only_crop).
                        crop = color_only_crop(image, box)
                        # Enlarged so the crop survives the client's 1600px
                        # downscale, exactly like the table crop.
                        if crop.width < 2400:
                            scale = 2400 / crop.width
                            crop = crop.resize((2400, max(1, int(crop.height * scale))),
                                               Image.LANCZOS)
                        crop_path = page.path.parent / f".colors-p{page.index:04d}-{i}.png"
                        crop.save(crop_path, "PNG")
                        read = _call_with_deadline(
                            lambda cp=crop_path, ns=names: reread(cp, ns),
                            self._call_budget(self.recognizer))
                        if read:
                            region_runs = extract_color_runs(read)
                            for color, plain in region_runs:
                                reads.setdefault(color, []).append(plain)
                            # The read succeeded, so it adjudicates these
                            # colors even when it found no runs of its own.
                            for name in names:
                                reads.setdefault(name, [])
                            cx = (box[0] + box[2]) / 2 / max(image.width, 1)
                            cy = (box[1] + box[3]) / 2 / max(image.height, 1)
                            # Runs from a left-margin region are not applied:
                            # margin marks are single glyphs beside the body,
                            # and the focused read misreads them as short
                            # words ("预备", "测度") that would color body
                            # text. The region still adjudicates colors and
                            # counts as evidence -- its runs just color
                            # nothing.
                            if cx >= 0.15:
                                runs.extend(region_runs)
                            # A colored block in the top-right corner is the
                            # page's corner table (or its marks): remember its
                            # texts so the table can be floated back to the
                            # corner whatever the transcription did with it.
                            if cx > 0.6 and cy < 0.25:
                                corner_texts.extend(t for _, t in region_runs)
                    except Exception as exc:  # noqa: BLE001 - one region only
                        log.warning("page %d: colored region %d not re-read: %s",
                                    page.index, i, exc)
                self._color_runs[page.index] = (runs, corner_texts, reads)
            else:
                runs, corner_texts, reads = cached
            if "\\textcolor" in latex and reads:
                latex, n_unwrapped = reconcile_colors(latex, reads)
                if n_unwrapped:
                    log.info("page %d: %d invented colored run(s) unwrapped",
                             page.index, n_unwrapped)
            if runs:
                latex, n = apply_color_runs(latex, runs)
                if n:
                    log.info("page %d: %d colored run(s) applied from crop re-reads",
                             page.index, n)
            present = self._page_colors(page)
            if present is not None and "\\textcolor" in latex:
                latex, n = drop_absent_colors(latex, present)
                if n:
                    log.info("page %d: %d run(s) in a color the page does not have unwrapped",
                             page.index, n)
            margin = self._margin_color(page)
            if margin:
                latex, n = color_margin_labels(latex, margin)
                if n:
                    log.info("page %d: %d margin label(s) colored %s", page.index, n, margin)
            if corner_texts:
                # The corner table's own text tells us which tabular it is; the
                # repositioning itself is deterministic (see wrap_corner_table).
                latex, moved = wrap_corner_table(latex, corner_texts)
                if moved:
                    log.info("page %d: corner table floated beside the body text",
                             page.index)
        except Exception as exc:  # noqa: BLE001 - colors are best-effort
            log.warning("page %d: could not re-read the colored ink: %s",
                        page.index, exc)
        return latex

    def _page_colors(self, page: PageImage) -> Optional[List[str]]:
        """The page's colored-ink families, measured once; None if unmeasurable."""
        if page.index not in self._ink_families:
            try:
                from n2lh.pipeline.ingest import color_ink_names
                with Image.open(page.path) as opened:
                    self._ink_families[page.index] = color_ink_names(opened.convert("RGB"))
            except Exception as exc:  # noqa: BLE001 - colors are best-effort
                log.warning("page %d: could not measure its ink colors: %s", page.index, exc)
                self._ink_families[page.index] = None
        return self._ink_families[page.index]

    def _margin_color(self, page: PageImage) -> Optional[str]:
        """The one pen color of the page's left-margin marks, measured; or None.

        Only when every colored block in the left margin is the same color --
        on the real page, six green section labels. Two colors there, or none,
        and nothing is guessed.
        """
        if page.index in self._margin_colors:
            return self._margin_colors[page.index]
        color = None
        try:
            with Image.open(page.path) as opened:
                image = opened.convert("RGB")
            names = set()
            for box, found in colored_ink_regions(image, max_regions=12):
                if (box[0] + box[2]) / 2 / max(image.width, 1) < 0.15:
                    names.update(found)
            if len(names) == 1:
                color = names.pop()
        except Exception as exc:  # noqa: BLE001 - colors are best-effort
            log.warning("page %d: could not measure the margin color: %s", page.index, exc)
        self._margin_colors[page.index] = color
        return color

    def _locate(self, page: PageImage):
        """Accurate figure boxes from the recognizer, or None (best-effort)."""
        locate = getattr(self.recognizer, "locate_figures", None)
        if locate is None or self._cancelled():
            return None
        try:
            return _call_with_deadline(lambda: locate(page), self._call_budget(self.recognizer))
        except Exception as exc:  # noqa: BLE001 - the coarse inline boxes still work
            log.warning("page %d: could not refine figure boxes: %s", page.index, exc)
            return None

    def _skipped_page(self, page: PageImage, reason: str, body_parts: List[str],
                      on_event: Optional[EventHandler]) -> PageOutcome:
        """Record a page that was deliberately not attempted (cancel / breaker)."""
        latex = f"% page {page.index}: not attempted ({reason})"
        label = ("not attempted - job cancelled" if "cancel" in reason
                 else "not attempted - job stopped early")
        body_parts.append(self._unverified_block(page.index, latex, label))
        self._emit(on_event, "page_done", page=page.index, status="failed",
                   attempts=0, engine="none", compile_ok=False)
        return PageOutcome(index=page.index, latex=latex, compile_ok=False,
                           attempts=0, engine="none", status="failed",
                           errors=[reason])

    @staticmethod
    def _unverified_block(index: int, latex: str, reason: str) -> str:
        """Visible gray note + the raw LaTeX as a COMMENT.

        A page that failed must not stay in the body as live LaTeX: it is known
        not to compile, and every later page (and the final document) is
        compiled on top of the accumulated body -- so one bad page used to
        break all the pages after it, and the whole PDF. ``reason`` must be
        LaTeX-safe text.
        """
        note = ("\\par\\noindent{\\color{gray}\\small [page " + str(index)
                + ": unverified - " + reason
                + "; source kept as a comment in the .tex, see the review pane]}\\par")
        return note + "\n" + comment_out(latex)

    @staticmethod
    def _emit(on_event: Optional[EventHandler], type_: str, **payload) -> None:
        if on_event is None:
            return
        event = {"type": type_, "ts": time.time()}
        event.update(payload)
        on_event(event)
