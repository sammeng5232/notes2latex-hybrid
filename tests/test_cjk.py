"""Chinese notes: pdflatex cannot draw the characters, so the job moves engine."""

from __future__ import annotations

import shutil

import pytest

from n2lh.compiler.latex import CompileResult, LatexCompiler
from n2lh.pipeline.assembler import build_document
from n2lh.pipeline.cjk import add_cjk, engine_for, has_cjk
from n2lh.pipeline.graph import DocumentPipeline
from n2lh.recognition.base import PageImage, Recognizer, TranscribeResult
from n2lh.recognition.prompts import make_preamble

HAS_XETEX = shutil.which("xelatex") is not None

CHINESE = "\\textbf{定理 1.20 (Heine-Borel)} $R^n$ 中有界闭集 $\\subset$ 紧。"


# -------------------------------------------------------------- detection
@pytest.mark.parametrize("text", [
    "实变函数",                       # Han
    "设 $f$ 可测",                    # Han mixed with maths
    "，。、《》：",                     # CJK punctuation on its own
    "カタカナ", "ひらがな", "한글",      # kana and hangul
])
def test_text_pdflatex_cannot_typeset_is_detected(text):
    assert has_cjk(text)


@pytest.mark.parametrize("text", [
    "", "Def. a measurable set", "$\\forall \\varepsilon > 0$",
    "Cauchy, Borel, Lebesgue", "naive resume cafe",
])
def test_ordinary_notes_do_not_trigger_the_switch(text):
    assert not has_cjk(text)


# ---------------------------------------------------------------- preamble
def test_ctex_goes_in_right_after_documentclass():
    out = add_cjk(make_preamble()).splitlines()
    assert out[0].startswith("\\documentclass")
    assert "ctex" in out[1]


def test_adding_it_twice_changes_nothing():
    once = add_cjk(make_preamble())
    assert add_cjk(once) == once


def test_a_preamble_without_a_documentclass_still_gets_it():
    assert "ctex" in add_cjk("\\usepackage{amsmath}\n")


def test_the_page_setup_survives_the_switch():
    """Paper size and font size are chosen in Settings and must not be lost."""
    pre = add_cjk(make_preamble(font_pt=12, paper="a3paper"))
    assert "\\documentclass[12pt,a3paper]{article}" in pre
    assert "a3paper" in pre and "ctex" in pre


@pytest.mark.parametrize("current,expected", [
    ("pdflatex", "xelatex"), ("", "xelatex"), ("xelatex", "xelatex"), ("lualatex", "lualatex"),
])
def test_an_engine_that_can_already_do_it_is_kept(current, expected):
    assert engine_for(current) == expected


# ------------------------------------------------------- through the pipeline
class ChineseRecognizer(Recognizer):
    name = "cjk"

    def __init__(self, latex=CHINESE):
        self._latex = latex

    def transcribe(self, page, context_tail, open_environments, guidance=None):
        return TranscribeResult(latex=self._latex, engine=self.name)


class RecordingCompiler:
    """Accepts anything, but remembers the engine it was asked for."""

    def __init__(self):
        self.engine = "pdflatex"
        self.engines_used = []

    def compile(self, tex, workdir):
        from pathlib import Path
        self.engines_used.append(self.engine)
        Path(workdir).mkdir(parents=True, exist_ok=True)
        (Path(workdir) / "document.pdf").write_bytes(b"%pdf")
        return CompileResult(ok=True, pdf_path=Path(workdir) / "document.pdf")


def run(tmp_path, latex):
    page = tmp_path / "p.png"
    page.write_bytes(b"not really a png")
    compiler = RecordingCompiler()
    pipe = DocumentPipeline(ChineseRecognizer(latex), compiler, fixer=None, max_retries=0,
                            preamble=make_preamble())
    result = pipe.run([PageImage(1, page)], tmp_path / "out")
    return pipe, compiler, result


def test_a_chinese_page_moves_the_job_to_xelatex(tmp_path):
    pipe, compiler, result = run(tmp_path, CHINESE)
    assert compiler.engine == "xelatex"
    assert "ctex" in pipe.preamble
    assert "ctex" in result.tex
    assert "定理" in result.tex               # transcribed, not translated


def test_an_english_page_is_left_on_pdflatex(tmp_path):
    pipe, compiler, _ = run(tmp_path, "Def. a measurable set $E$.")
    assert compiler.engine == "pdflatex"
    assert "ctex" not in pipe.preamble


def test_one_chinese_page_is_enough_for_the_whole_document(tmp_path):
    """A mixed set of notes must not compile half of itself with pdflatex."""
    pages = []
    for i, text in enumerate(["Def. an English page.", CHINESE, "Thm. another English page."], 1):
        p = tmp_path / f"p{i}.png"
        p.write_bytes(b"x")
        pages.append((p, text))

    class Mixed(Recognizer):
        name = "mixed"

        def transcribe(self, page, context_tail, open_environments, guidance=None):
            return TranscribeResult(latex=pages[page.index - 1][1], engine=self.name)

    compiler = RecordingCompiler()
    pipe = DocumentPipeline(Mixed(), compiler, fixer=None, max_retries=0,
                            preamble=make_preamble())
    pipe.run([PageImage(i + 1, p) for i, (p, _) in enumerate(pages)], tmp_path / "out")
    assert compiler.engine == "xelatex"
    assert compiler.engines_used[-1] == "xelatex"   # the final document too


# --------------------------------------------------------- the real toolchain
@pytest.mark.skipif(not HAS_XETEX, reason="no XeLaTeX installed")
def test_a_chinese_page_really_compiles(tmp_path):
    """pdflatex answers 'Unicode character 实 (U+5B9E) not set up for use with
    LaTeX' and no repair can help it; XeLaTeX with ctex builds the same page."""
    body = ("\\textbf{定理 6.16 (Bessel-Fischer)} 在 $L^2$ 中任一规范正交系 "
            "$\\{h_n\\}$ 均有极限点。\n\n\\textbf{定义 2.2} 设 $E \\subset R^n$ 可测。")
    tex = build_document(body, preamble=add_cjk(make_preamble()))
    result = LatexCompiler(engine="xelatex", timeout=420).compile(tex, tmp_path / "cjk")
    assert result.ok, result.errors


@pytest.mark.skipif(not HAS_XETEX, reason="no XeLaTeX installed")
def test_pdflatex_really_cannot_do_it(tmp_path):
    """The premise of the whole switch, asserted rather than assumed."""
    tex = build_document("定理 1.1", preamble=make_preamble())
    result = LatexCompiler(engine="pdflatex", timeout=300).compile(tex, tmp_path / "pdf")
    assert not result.ok
    assert any("Unicode character" in str(e) for e in result.errors)
