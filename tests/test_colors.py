"""Putting the colored ink back into a transcription that dropped it.

Real case: a vocabulary table's English column was written in pink pen; three
prompt shapes could not get the whole-page transcription to mark it, while the
same model on a crop of just that column returned seven correct runs. The
merge here must therefore be conservative: it may only wrap text that is
already in the page LaTeX.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PIL import Image

from n2lh.pipeline.colors import apply_color_runs, extract_color_runs
from n2lh.pipeline.graph import DocumentPipeline
from n2lh.recognition.base import PageImage, Recognizer, TranscribeResult


# ------------------------------------------------------------- extracting
def test_runs_are_extracted_with_their_plain_text():
    crop = ("\\textcolor{pink}{Closed nested sets thm} and "
            "\\textcolor{green}{\\underline{有界变差}} plus \\textcolor{red}{\\cancel{②}}")
    runs = extract_color_runs(crop)
    assert ("pink", "Closed nested sets thm") in runs
    assert ("green", "有界变差") in runs
    # a two-character Latin run is too short to match safely
    assert all(len(text) >= 3 or any("\u3400" <= c <= "\u9fff" for c in text)
               for _, text in runs)


def test_a_short_chinese_word_is_a_run_but_a_latin_pair_is_not():
    """振幅 is a complete word; 'ab' could match anywhere."""
    assert extract_color_runs("\\textcolor{pink}{振幅}") == [("pink", "振幅")]
    assert extract_color_runs("\\textcolor{pink}{ab}") == []


def test_short_runs_and_empty_input_are_dropped():
    assert extract_color_runs("\\textcolor{red}{①}") == []
    assert extract_color_runs(None) == []
    assert extract_color_runs("no colors here") == []


# ---------------------------------------------------------------- merging
def test_a_run_wraps_the_matching_text_exactly():
    out, n = apply_color_runs(
        "Cantor 闭集套定理 & Closed nested sets theorem \\\\",
        [("pink", "Closed nested sets theorem")])
    assert n == 1
    assert "\\textcolor{pink}{Closed nested sets theorem}" in out
    assert "Cantor 闭集套定理 &" in out          # the rest is untouched


def test_a_slightly_misread_run_still_matches_fuzzily():
    """Two readings of the same handwriting differ: thm vs theorem."""
    out, n = apply_color_runs(
        "The theorem says: Closed nested sets theorem \\\\",
        [("pink", "Closed nested sets thm")])
    assert n == 1
    assert "\\textcolor{pink}{Closed nested sets theorem}" in out


def test_a_run_found_nowhere_is_skipped_not_guessed():
    out, n = apply_color_runs("Some unrelated body text.",
                              [("pink", "Epigraph Supgraph")])
    assert n == 0 and out == "Some unrelated body text."


def test_overlapping_runs_are_applied_once():
    out, n = apply_color_runs("Total variation matters",
                              [("pink", "Total variation"),
                               ("green", "variation matters")])
    assert n == 1
    assert out.count("\\textcolor") == 1


def test_the_wrap_respects_token_boundaries():
    out, n = apply_color_runs("he \\underline{Oscillation} was",
                              [("pink", "Oscillation")])
    assert n == 1
    assert "\\textcolor{pink}{\\underline{Oscillation}}" in out


def test_applying_no_runs_changes_nothing():
    assert apply_color_runs("x & y", []) == ("x & y", 0)


# --------------------------------------------------------- through the pipeline
def colored_page(tmp_path: Path, name="page.png") -> Path:
    """A page whose top-right corner carries a block of pink pen."""
    im = Image.new("RGB", (1000, 1400), (255, 255, 255))
    px = im.load()
    for x in range(700, 950):           # pink block, top right
        for y in range(100, 300):
            px[x, y] = (255, 78, 225)
    for x in range(80, 600):            # black 'body text' strokes
        for y in range(500, 700):
            px[x, y] = (40, 40, 40)
    path = tmp_path / name
    im.save(path)
    return path


class ColorDroppingRecognizer(Recognizer):
    """Transcribes the page without colors, but reads a color crop well."""

    name = "cdr"

    def __init__(self, runs: Optional[List] = None) -> None:
        self.calls: List = []
        self._runs = runs or [("pink", "Epigraph and Supgraph")]

    def transcribe(self, page, context_tail, open_environments, guidance=None):
        return TranscribeResult(
            latex="A vocabulary table:\n\nEpigraph and Supgraph, Hypograph.", engine=self.name)

    def transcribe_colors(self, image_path, names=None):
        self.calls.append((str(image_path), names))
        return "\\textcolor{pink}{Epigraph and Supgraph}"


class OkCompiler:
    def compile(self, tex, workdir):
        from n2lh.compiler.latex import CompileResult
        Path(workdir).mkdir(parents=True, exist_ok=True)
        (Path(workdir) / "document.pdf").write_bytes(b"%pdf")
        return CompileResult(ok=True, pdf_path=Path(workdir) / "document.pdf")


def test_a_colored_page_gets_its_colors_back(tmp_path):
    rec = ColorDroppingRecognizer()
    pipe = DocumentPipeline(rec, OkCompiler(), fixer=None, max_retries=0)
    result = pipe.run([PageImage(1, colored_page(tmp_path))], tmp_path / "out")
    assert "\\textcolor{pink}{Epigraph and Supgraph}" in result.tex
    assert len(rec.calls) == 1, "one focused call per colored region"
    names = rec.calls[0][1]
    assert names == ["pink"]


def test_a_transcription_that_already_has_colors_costs_no_call(tmp_path):
    class AlreadyColored(ColorDroppingRecognizer):
        def transcribe(self, page, context_tail, open_environments, guidance=None):
            return TranscribeResult(
                latex="\\textcolor{pink}{Epigraph and Supgraph} and more.",
                engine=self.name)

    rec = AlreadyColored()
    pipe = DocumentPipeline(rec, OkCompiler(), fixer=None, max_retries=0)
    pipe.run([PageImage(1, colored_page(tmp_path))], tmp_path / "out")
    assert rec.calls == []


def test_a_grayscale_page_costs_no_color_call(tmp_path):
    im = Image.new("RGB", (1000, 1400), (255, 255, 255))
    px = im.load()
    for x in range(80, 600):
        for y in range(500, 700):
            px[x, y] = (40, 40, 40)
    page = tmp_path / "gray.png"
    im.save(page)

    rec = ColorDroppingRecognizer()
    pipe = DocumentPipeline(rec, OkCompiler(), fixer=None, max_retries=0)
    pipe.run([PageImage(1, page)], tmp_path / "out")
    assert rec.calls == []


def test_a_failing_color_read_leaves_the_page_as_it_was(tmp_path):
    class Boom(ColorDroppingRecognizer):
        def transcribe_colors(self, image_path, names=None):
            raise RuntimeError("endpoint down")

    result = DocumentPipeline(Boom(), OkCompiler(), fixer=None, max_retries=0).run(
        [PageImage(1, colored_page(tmp_path))], tmp_path / "out")
    assert "Epigraph and Supgraph" in result.tex
    assert "\\textcolor" not in result.tex


def test_one_flaky_region_does_not_cost_the_others(tmp_path):
    """The endpoint sometimes answers nothing but a tool_calls finish; the
    region that fails must not take down the regions that answer."""
    # Two colored regions far apart: pink top right, green top left.
    im = Image.new("RGB", (1000, 1400), (255, 255, 255))
    px = im.load()
    for x in range(700, 950):
        for y in range(100, 300):
            px[x, y] = (255, 78, 225)
    for x in range(50, 300):
        for y in range(100, 400):
            px[x, y] = (91, 164, 128)
    page = tmp_path / "two.png"
    im.save(page)

    class FlakyPink(ColorDroppingRecognizer):
        def transcribe_colors(self, image_path, names=None):
            self.calls.append(names)
            if names == ["pink"]:
                raise RuntimeError("model returned no answer content")
            return "\\textcolor{green}{Epigraph and Supgraph}"

    rec = FlakyPink()
    result = DocumentPipeline(rec, OkCompiler(), fixer=None, max_retries=0).run(
        [PageImage(1, page)], tmp_path / "out")
    assert "\\textcolor{green}{Epigraph and Supgraph}" in result.tex
    assert len(rec.calls) == 2, "both regions were attempted"
