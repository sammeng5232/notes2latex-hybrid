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

from n2lh.compiler.latex import CompileResult, LatexCompiler
from n2lh.pipeline.assembler import build_document
from n2lh.pipeline.context import ContextWindow, EnvironmentTracker
from n2lh.pipeline.autofix import autofix_latex
from n2lh.pipeline.figures import render_figures
from n2lh.pipeline.sanitize import comment_out, sanitize_body
from n2lh.pipeline.style import bold_labels
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
                result = _call_with_deadline(
                    lambda: self.recognizer.transcribe(page, "", []),
                    self._call_budget(self.recognizer))
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
                    engine = self.recognizer if attempt == 1 else (self.fixer or self.recognizer)
                    result = self._finish(page, _call_with_deadline(
                        lambda engine=engine: engine.transcribe(page, tail, open_envs),
                        self._call_budget(engine)))
                    engine_used = engine.name
                    engine_succeeded = True
                else:
                    fixer = self.fixer or self.recognizer
                    guidance = fix_user_prompt(result.latex, errors, repeats)
                    result = self._finish(page, _call_with_deadline(
                        lambda fixer=fixer, guidance=guidance: fixer.transcribe(
                            page, tail, open_envs, guidance=guidance),
                        self._call_budget(fixer)))
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
                # engine (engine=vlm) is pointless: its client already retried.
                if (self.fixer is not None and self.fixer is not self.recognizer
                        and attempt == 1):
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
        latex, bolded = bold_labels(latex)
        if bolded:
            log.debug("page %d: bolded %d label(s)", page.index, bolded)
        if latex != result.latex:
            result = TranscribeResult(latex=latex, engine=result.engine,
                                      confidence=result.confidence, notes=result.notes)
        return result

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
