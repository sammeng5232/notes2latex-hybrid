from __future__ import annotations

import threading
import time
from typing import List, Optional

from n2lh.compiler.latex import CompileResult, LatexError
from n2lh.pipeline.graph import DocumentPipeline
from n2lh.recognition.base import PageImage, Recognizer, TranscribeResult


class ScriptedRecognizer(Recognizer):
    """Pops queued outputs; records every call for assertions."""

    def __init__(self, name: str = "scripted",
                 outputs: Optional[List[str]] = None,
                 fail: bool = False) -> None:
        self.name = name
        self.outputs = list(outputs or [])
        self.fail = fail
        self.calls: List[dict] = []

    def transcribe(self, page, context_tail, open_environments, guidance=None):
        self.calls.append({
            "page": page.index,
            "tail": context_tail,
            "open": list(open_environments),
            "guidance": guidance,
        })
        if self.fail:
            raise RuntimeError("engine down")
        latex = self.outputs.pop(0) if self.outputs else "\\text{fallback}"
        return TranscribeResult(latex=latex, engine=self.name)


class ConcurrentRecognizer(Recognizer):
    """Thread-safe fake: returns per-page canned output, tracks concurrency
    actually observed (peak simultaneous calls in flight) and per-page
    context/open-environment as seen by each call."""

    def __init__(self, name: str = "concurrent", delay: float = 0.05,
                 fail_pages: Optional[set] = None) -> None:
        self.name = name
        self.delay = delay
        self.fail_pages = fail_pages or set()
        self._lock = threading.Lock()
        self.calls: List[dict] = []
        self._in_flight = 0
        self.peak_in_flight = 0

    def transcribe(self, page, context_tail, open_environments, guidance=None):
        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            if page.index in self.fail_pages and guidance is None:
                raise RuntimeError(f"page {page.index} engine down")
            time.sleep(self.delay)
            with self._lock:
                self.calls.append({
                    "page": page.index, "tail": context_tail,
                    "open": list(open_environments), "guidance": guidance,
                })
            return TranscribeResult(latex=f"\\text{{page {page.index}}}", engine=self.name)
        finally:
            with self._lock:
                self._in_flight -= 1


class FakeCompiler:
    """Fails whenever the document contains a BROKEN marker."""

    def __init__(self):
        self.compile_count = 0

    def compile(self, tex: str, workdir) -> CompileResult:
        import shutil
        from pathlib import Path
        self.compile_count += 1
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        if "BROKEN" in tex:
            return CompileResult(ok=False, errors=[
                LatexError("Undefined control sequence. \\BROKEN", line=7)])
        (workdir / "document.pdf").write_bytes(b"%pdf")
        return CompileResult(ok=True, pdf_path=workdir / "document.pdf")


def make_pages(tmp_path, n: int) -> List[PageImage]:
    pages = []
    for i in range(1, n + 1):
        p = tmp_path / f"p{i:04d}.png"
        p.write_bytes(b"fake-image")
        pages.append(PageImage(index=i, path=p))
    return pages


def collect(events):
    return [e["type"] for e in events]


# ------------------------------------------------------------------- tests
def test_first_try_success_no_fixer(tmp_path):
    rec = ScriptedRecognizer(outputs=["\\text{hello page one}",
                                      "\\text{hello page two}"])
    pipeline = DocumentPipeline(rec, FakeCompiler(), fixer=None, max_retries=3)
    events = []
    result = pipeline.run(make_pages(tmp_path, 2), tmp_path / "out", events.append)

    assert [p.status for p in result.pages] == ["ok", "ok"]
    assert all(p.attempts == 1 for p in result.pages)
    assert result.ok
    assert "job_done" in collect(events)
    assert "\\text{hello page one}" in result.tex
    assert "\\text{hello page two}" in result.tex
    # tex written to disk
    assert (tmp_path / "out" / "document.tex").exists()


def test_fixer_repairs_broken_page_and_gets_guidance(tmp_path):
    rec = ScriptedRecognizer(outputs=["\\BROKEN", "\\text{page two}"])
    fixer = ScriptedRecognizer(name="fixer", outputs=["\\text{repaired page one}"])
    pipeline = DocumentPipeline(rec, FakeCompiler(), fixer=fixer, max_retries=3)

    result = pipeline.run(make_pages(tmp_path, 2), tmp_path / "out")

    assert result.pages[0].status == "fixed"
    assert result.pages[0].attempts == 2
    assert result.pages[0].engine == "fixer"
    assert result.ok
    # fixer received the compile error as guidance
    fix_call = fixer.calls[0]
    assert fix_call["guidance"] is not None
    assert "Undefined control sequence" in fix_call["guidance"]
    assert "\\BROKEN" in fix_call["guidance"]
    # page 1 body was repaired, not duplicated
    assert result.tex.count("\\text{repaired page one}") == 1
    assert "BROKEN" not in result.tex


def test_context_carries_to_next_page(tmp_path):
    rec = ScriptedRecognizer(outputs=[
        "\\begin{align*}\nx &= 1 \\\\",           # page 1: dangling align*
        "\\text{page two continues}",              # page 2
    ])
    pipeline = DocumentPipeline(rec, FakeCompiler(), fixer=None, max_retries=1)
    pipeline.run(make_pages(tmp_path, 2), tmp_path / "out")

    page2_call = rec.calls[1]
    assert "x &= 1" in page2_call["tail"]                 # rolling context window
    assert page2_call["open"] == ["align*"]               # open env forwarded
    # final document must close the dangling environment
    assert result_tex_closes_align(tmp_path)


def result_tex_closes_align(tmp_path) -> bool:
    tex = (tmp_path / "out" / "document.tex").read_text("utf-8")
    return tex.count("\\begin{align*}") == tex.count("\\end{align*}")


def test_engine_failure_escalates_to_fixer(tmp_path):
    rec = ScriptedRecognizer(fail=True)
    fixer = ScriptedRecognizer(name="fixer", outputs=["\\text{rescued}"])
    pipeline = DocumentPipeline(rec, FakeCompiler(), fixer=fixer, max_retries=3)

    result = pipeline.run(make_pages(tmp_path, 1), tmp_path / "out")

    assert result.pages[0].status == "fixed"
    assert result.ok
    # An engine failure leaves nothing to "fix": the fixer must be asked for a
    # plain transcription. Handing it a repair prompt about EMPTY LaTeX made
    # the model answer with a whole standalone document (see the cascade test).
    assert fixer.calls and fixer.calls[0]["guidance"] is None


def test_engine_failure_without_fixer_marks_failed_and_continues(tmp_path):
    rec = ScriptedRecognizer(fail=True)
    pipeline = DocumentPipeline(rec, FakeCompiler(), fixer=None, max_retries=3)
    events = []
    result = pipeline.run(make_pages(tmp_path, 2), tmp_path / "out", events.append)

    assert all(p.status == "failed" for p in result.pages)
    assert not result.ok
    assert "engine_error" in collect(events)
    assert "unverified" in result.tex   # isolated placeholder kept for review


def test_retries_exhausted_keeps_unverified_output(tmp_path):
    rec = ScriptedRecognizer(outputs=["\\BROKEN", "\\BROKEN", "\\BROKEN", "\\BROKEN"])
    pipeline = DocumentPipeline(rec, FakeCompiler(), fixer=None, max_retries=3)
    result = pipeline.run(make_pages(tmp_path, 1), tmp_path / "out")

    assert result.pages[0].status == "failed"
    assert result.pages[0].attempts == 4          # 1 primary + 3 repairs (max_retries)
    assert "unverified" in result.tex
    assert result.pages[0].errors                 # compile errors surfaced


# ------------------------------------------------------------ parallel prefetch
def test_default_is_sequential_single_worker(tmp_path):
    """parallel_workers defaults to 1: no concurrency, no prefetch events."""
    rec = ConcurrentRecognizer(delay=0.02)
    pipeline = DocumentPipeline(rec, FakeCompiler(), fixer=None, max_retries=1)
    events = []
    result = pipeline.run(make_pages(tmp_path, 5), tmp_path / "out", events.append)

    assert result.ok
    assert rec.peak_in_flight == 1                       # never overlapped
    assert "prefetch_start" not in collect(events)
    assert "prefetched" not in collect(events)


def test_parallel_prefetch_runs_pages_concurrently(tmp_path):
    rec = ConcurrentRecognizer(delay=0.15)
    pipeline = DocumentPipeline(rec, FakeCompiler(), fixer=None, max_retries=1,
                                parallel_workers=4)
    events = []
    result = pipeline.run(make_pages(tmp_path, 6), tmp_path / "out", events.append)

    assert result.ok
    assert all(p.status == "ok" for p in result.pages)
    assert rec.peak_in_flight > 1                         # calls actually overlapped
    assert rec.peak_in_flight <= 4                         # respected worker cap
    kinds = collect(events)
    assert "prefetch_start" in kinds
    assert kinds.count("prefetched") == 6
    # every page still gets exactly one transcribe call (prefetch result reused,
    # not re-fetched live)
    assert len(rec.calls) == 6
    assert sorted(c["page"] for c in rec.calls) == [1, 2, 3, 4, 5, 6]
    # prefetched calls see no cross-page context (dispatched together)
    assert all(c["tail"] == "" and c["open"] == [] for c in rec.calls)


def test_parallel_prefetch_failure_falls_back_to_live_sequential_call(tmp_path):
    """A page whose prefetch call raises is retried live in the normal
    sequential pass, where it succeeds (fail_pages only fails when
    guidance is None on the very first prefetch dispatch -- the live
    fallback call is also guidance=None, so make the fake only fail once
    via a mutable counter instead."""
    class FlakyOnce(Recognizer):
        name = "flaky"

        def __init__(self):
            self.attempts_page2 = 0
            self.calls = []

        def transcribe(self, page, context_tail, open_environments, guidance=None):
            self.calls.append(page.index)
            if page.index == 2:
                self.attempts_page2 += 1
                if self.attempts_page2 == 1:
                    raise RuntimeError("transient prefetch failure")
            return TranscribeResult(latex=f"\\text{{page {page.index}}}", engine=self.name)

    rec = FlakyOnce()
    pipeline = DocumentPipeline(rec, FakeCompiler(), fixer=None, max_retries=1,
                                parallel_workers=3)
    events = []
    result = pipeline.run(make_pages(tmp_path, 3), tmp_path / "out", events.append)

    assert result.ok
    assert all(p.status == "ok" for p in result.pages)
    # page 2 was attempted twice: once in prefetch (failed), once live (succeeded)
    assert rec.calls.count(2) == 2
    prefetch_events = [e for e in events if e["type"] == "prefetched" and e["page"] == 2]
    assert prefetch_events and prefetch_events[0]["ok"] is False


def test_parallel_prefetch_skipped_for_single_page(tmp_path):
    """No point spinning up a thread pool for a 1-page job."""
    rec = ConcurrentRecognizer(delay=0.01)
    pipeline = DocumentPipeline(rec, FakeCompiler(), fixer=None, max_retries=1,
                                parallel_workers=4)
    events = []
    pipeline.run(make_pages(tmp_path, 1), tmp_path / "out", events.append)
    assert "prefetch_start" not in collect(events)


def test_parallel_prefetch_abandons_permanently_stuck_page(tmp_path, monkeypatch):
    """A recognizer call that never returns at all (the real failure mode we
    hit in production -- a blocking network read that outlives even its own
    configured timeout) must not hang the whole job forever. _prefetch has to
    give up on it and move on; the stuck page falls back to live processing
    later, which will itself surface as a normal engine failure since the
    fake never succeeds."""
    class HangsForeverOnOnePage(Recognizer):
        name = "hangs"

        def __init__(self, stuck_page: int):
            self.stuck_page = stuck_page
            self.calls: List[int] = []

        def transcribe(self, page, context_tail, open_environments, guidance=None):
            self.calls.append(page.index)
            if page.index == self.stuck_page:
                # Simulate a network read that never returns by blocking on
                # an event that's never set -- NOT a sleep with a bound, a
                # genuinely-unbounded wait, matching the real failure mode.
                threading.Event().wait()
                raise AssertionError("unreachable")
            return TranscribeResult(latex=f"\\text{{page {page.index}}}", engine=self.name)

    rec = HangsForeverOnOnePage(stuck_page=2)
    pipeline = DocumentPipeline(rec, FakeCompiler(), fixer=None, max_retries=1,
                                parallel_workers=3)
    pipeline.RECOGNIZER_CALL_BUDGET = 0.3  # test-sized budget, not the real 900s
    events = []

    start = time.time()
    result = pipeline.run(make_pages(tmp_path, 3), tmp_path / "out", events.append)
    elapsed = time.time() - start

    # The whole run must finish quickly -- nowhere near "forever" -- despite
    # one page's transcribe() call that will literally never return.
    assert elapsed < 5.0
    assert result.pages[0].status == "ok"
    assert result.pages[2].status == "ok"
    # page 2 was abandoned in prefetch, then failed live too (the fake hangs
    # unconditionally on that page, so it can never succeed) -- it's the one
    # page allowed to end up "failed" here.
    assert result.pages[1].status == "failed"
    prefetch_fail = [e for e in events if e["type"] == "prefetched" and e["page"] == 2]
    assert prefetch_fail and prefetch_fail[0]["ok"] is False
    assert "abandoned" in prefetch_fail[0]["error"]


# ------------------------------------------------- failure isolation / breaker
class CommentAwareCompiler(FakeCompiler):
    """Like FakeCompiler but, like real LaTeX, ignores %-commented lines."""

    def compile(self, tex: str, workdir) -> CompileResult:
        live = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))
        return super().compile(live, workdir)


class PreambleAwareCompiler(CommentAwareCompiler):
    """Fails exactly like LaTeX when a preamble command appears in the body."""

    def compile(self, tex: str, workdir) -> CompileResult:
        body = tex.split("\\begin{document}", 1)[-1]
        live = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("%"))
        if "\\usepackage" in live or "\\documentclass" in live:
            return CompileResult(ok=False, errors=[
                LatexError("LaTeX Error: Can be used only in preamble.", line=9)])
        return super().compile(tex, workdir)


def test_failed_page_does_not_poison_the_pages_after_it(tmp_path):
    """The production incident: page 3 failed, its raw LaTeX stayed in the running
    document, and EVERY later page then failed with the same error whatever it
    contained -- each one burning its whole retry budget."""
    rec = ScriptedRecognizer(outputs=["\\BROKEN"] * 4 + ["\\text{page two}", "\\text{page three}"])
    pipeline = DocumentPipeline(rec, CommentAwareCompiler(), fixer=None, max_retries=3)
    result = pipeline.run(make_pages(tmp_path, 3), tmp_path / "out")

    assert [p.status for p in result.pages] == ["failed", "ok", "ok"]
    assert [p.attempts for p in result.pages] == [4, 1, 1]      # later pages: first try
    assert "\\text{page two}" in result.tex and "\\text{page three}" in result.tex
    # The failed page is kept only as a comment, so the document still builds.
    assert "% \\BROKEN" in result.tex
    assert "unverified" in result.tex
    assert result.pdf_path is not None and not result.ok


def test_stray_full_document_from_the_model_is_sanitized(tmp_path):
    rogue = ("\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n"
             "\\text{real content}\n\\end{document}")
    rec = ScriptedRecognizer(outputs=[rogue, "\\text{page two}"])
    pipeline = DocumentPipeline(rec, PreambleAwareCompiler(), fixer=None, max_retries=0)
    result = pipeline.run(make_pages(tmp_path, 2), tmp_path / "out")

    assert [p.status for p in result.pages] == ["ok", "ok"]
    assert result.ok
    assert "\\text{real content}" in result.tex
    assert result.tex.count("\\documentclass") == 1      # only the real preamble
    assert result.tex.count("\\begin{document}") == 1


def test_failed_pages_do_not_feed_context_or_environment_tracking(tmp_path):
    rec = ScriptedRecognizer(outputs=[
        "\\begin{align*}\nBROKEN &= 1",          # fails: leaves an env open
        "\\begin{align*}\nBROKEN &= 1",
        "\\text{page two}",
    ])
    pipeline = DocumentPipeline(rec, CommentAwareCompiler(), fixer=None, max_retries=1)
    pipeline.run(make_pages(tmp_path, 2), tmp_path / "out")

    page2 = rec.calls[-1]
    assert page2["open"] == []                    # failed page's align* is not "open"
    assert "BROKEN" not in page2["tail"]


def test_circuit_breaker_stops_a_job_whose_recognizer_is_down(tmp_path):
    rec = ScriptedRecognizer(fail=True)
    pipeline = DocumentPipeline(rec, CommentAwareCompiler(), fixer=None, max_retries=3,
                                max_consecutive_engine_failures=3)
    events = []
    result = pipeline.run(make_pages(tmp_path, 8), tmp_path / "out", events.append)

    assert len(rec.calls) == 3                     # not 8: the rest were not attempted
    assert [p.attempts for p in result.pages] == [1, 1, 1, 0, 0, 0, 0, 0]
    assert all(p.status == "failed" for p in result.pages)
    assert result.aborted and "stopped early" in result.aborted
    assert "job_aborted" in collect(events)
    assert not result.ok
    assert result.pdf_path is not None              # placeholders still build a PDF


def test_a_successful_page_resets_the_engine_failure_streak(tmp_path):
    class FailEveryOther(Recognizer):
        name = "flaky"

        def __init__(self):
            self.n = 0

        def transcribe(self, page, context_tail, open_environments, guidance=None):
            self.n += 1
            if self.n % 2 == 1:
                raise RuntimeError("blip")
            return TranscribeResult(latex=f"\\text{{page {page.index}}}", engine=self.name)

    pipeline = DocumentPipeline(FailEveryOther(), CommentAwareCompiler(), fixer=None,
                                max_retries=0, max_consecutive_engine_failures=2)
    result = pipeline.run(make_pages(tmp_path, 6), tmp_path / "out")
    assert result.aborted is None                  # never 2 failures in a row
    assert [p.status for p in result.pages] == ["failed", "ok"] * 3 \
        or sum(p.engine_failed for p in result.pages) == 3


def test_breaker_keeps_pages_that_were_already_prefetched(tmp_path):
    """Endpoint dies after some pages were prefetched: those good results must
    still be compiled and kept -- only pages needing a live call are skipped."""
    class DiesAfterThree(Recognizer):
        name = "dies"

        def __init__(self):
            self.calls: List[int] = []
            self._lock = threading.Lock()

        def transcribe(self, page, context_tail, open_environments, guidance=None):
            time.sleep(0.05)
            with self._lock:
                self.calls.append(page.index)
            if page.index > 3:
                raise RuntimeError("endpoint down")
            return TranscribeResult(latex=f"\\text{{page {page.index}}}", engine=self.name)

    rec = DiesAfterThree()
    pipeline = DocumentPipeline(rec, CommentAwareCompiler(), fixer=None, max_retries=0,
                                parallel_workers=2, max_consecutive_engine_failures=3)
    events = []
    result = pipeline.run(make_pages(tmp_path, 9), tmp_path / "out", events.append)

    assert [p.status for p in result.pages[:3]] == ["ok", "ok", "ok"]
    assert all(p.status == "failed" and p.attempts == 0 for p in result.pages[3:])
    for i in (1, 2, 3):
        assert f"\\text{{page {i}}}" in result.tex
    assert result.aborted and "job_aborted" in collect(events)


def test_cancel_stops_the_job_and_skips_remaining_pages(tmp_path):
    cancel = threading.Event()

    class CancelsOnSecondCall(Recognizer):
        name = "c"

        def __init__(self):
            self.n = 0

        def transcribe(self, page, context_tail, open_environments, guidance=None):
            self.n += 1
            if self.n == 2:
                cancel.set()
            return TranscribeResult(latex=f"\\text{{page {page.index}}}", engine=self.name)

    pipeline = DocumentPipeline(CancelsOnSecondCall(), CommentAwareCompiler(), fixer=None,
                                max_retries=0, cancel_event=cancel)
    events = []
    result = pipeline.run(make_pages(tmp_path, 5), tmp_path / "out", events.append)

    assert result.aborted == "cancelled by user"
    assert "job_cancelled" in collect(events)
    assert [p.attempts for p in result.pages[2:]] == [0, 0, 0]
    assert result.pdf_path is not None

def test_repair_prompt_escalates_when_the_same_error_keeps_coming_back(tmp_path):
    """A real job got the identical broken diagram back on four repairs in a row.
    From the second identical failure the repair prompt must say so and change tack."""
    rec = ScriptedRecognizer(outputs=["\\BROKEN"] * 4)
    pipeline = DocumentPipeline(rec, FakeCompiler(), fixer=None, max_retries=3)
    pipeline.run(make_pages(tmp_path, 1), tmp_path / "out")

    g = [c["guidance"] for c in rec.calls]
    assert g[0] is None                                   # first attempt: plain transcription
    assert "SAME error" not in g[1]                       # first repair: ordinary prompt
    assert "SAME error" in g[2] and "previous 1 fix" in g[2]
    assert "previous 2 fix" in g[3]

# ------------------------------------------------------------------ autofix
class AmpersandCompiler(FakeCompiler):
    """Fails on a bare (unescaped) ampersand, like LaTeX outside an alignment env."""

    def compile(self, tex: str, workdir) -> CompileResult:
        import re
        if re.search(r"(?<!\\)&", tex.split("\\begin{document}", 1)[-1]):
            self.compile_count += 1
            return CompileResult(ok=False, errors=[
                LatexError("Misplaced alignment tab character &.", line=12)])
        return super().compile(tex, workdir)


def test_mechanical_fault_is_repaired_without_calling_the_model_again(tmp_path):
    rec = ScriptedRecognizer(outputs=["Hausdorff & second countable."])
    fixer = ScriptedRecognizer(name="fixer", outputs=["should never be used"])
    events = []
    pipeline = DocumentPipeline(rec, AmpersandCompiler(), fixer=fixer, max_retries=3)
    result = pipeline.run(make_pages(tmp_path, 1), tmp_path / "out", events.append)

    assert result.ok and result.pages[0].status == "fixed"
    assert fixer.calls == [] and len(rec.calls) == 1
    assert "Hausdorff \\& second countable." in result.tex
    auto = [e for e in events if e["type"] == "autofix"]
    assert auto and auto[0]["ok"] and "bare" in auto[0]["changes"][0]


def test_autofix_is_not_tried_on_a_page_that_already_compiles(tmp_path):
    rec = ScriptedRecognizer(outputs=["fine $a \\\\ b$ text"])
    events = []
    DocumentPipeline(rec, AmpersandCompiler(), fixer=None).run(
        make_pages(tmp_path, 1), tmp_path / "out", events.append)
    assert not [e for e in events if e["type"] == "autofix"]


def test_when_autofix_cannot_help_the_model_repair_still_runs(tmp_path):
    rec = ScriptedRecognizer(outputs=["\\BROKEN"])
    fixer = ScriptedRecognizer(name="fixer", outputs=["\\text{repaired}"])
    result = DocumentPipeline(rec, FakeCompiler(), fixer=fixer, max_retries=3).run(
        make_pages(tmp_path, 1), tmp_path / "out")
    assert result.ok and len(fixer.calls) == 1