from __future__ import annotations

import shutil

import pytest

from n2lh.compiler.latex import LatexCompiler
from n2lh.pipeline.assembler import build_document
from n2lh.pipeline.style import bold_labels

HAS_TEX = shutil.which("latexmk") is not None or shutil.which("pdflatex") is not None


def bolded(text):
    return bold_labels(text)[0]


@pytest.mark.parametrize("line,expected", [
    ("Def. A k-dim distribution.", "\\textbf{Def.} A k-dim distribution."),
    ("Thm. $f$ is smooth.", "\\textbf{Thm.} $f$ is smooth."),
    ("Cor. $H^k(M) = 0$.", "\\textbf{Cor.} $H^k(M) = 0$."),
    ("Eg. the sphere.", "\\textbf{Eg.} the sphere."),
    ("Pf. induct on $n$.", "\\textbf{Pf.} induct on $n$."),
    ("Rmk. 1-dim dist.", "\\textbf{Rmk.} 1-dim dist."),
    ("Def: a chart.", "\\textbf{Def:} a chart."),
    ("Lem (Five lem). the rows are exact.", "\\textbf{Lem (Five lem).} the rows are exact."),
    ("Thm (Poincare duality). $P_M^k$ is iso.", "\\textbf{Thm (Poincare duality).} $P_M^k$ is iso."),
])
def test_the_label_that_opens_a_statement_is_bolded(line, expected):
    assert bolded(line) == expected


@pytest.mark.parametrize("line", [
    "Lecture 9 2025/11/9 Week 12",
    "Lecture 3 20250917 Week 3",
    "Lecture 1 20250903 Week 1",
])
def test_a_lecture_heading_is_bolded_whole(line):
    assert bolded(line) == "\\textbf{%s}" % line


def test_the_count_says_how_many_labels_were_bolded():
    text = "Def. one.\nsome prose\nThm. two.\nPf. three."
    out, n = bold_labels(text)
    assert n == 3 and out.count("\\textbf{") == 3


@pytest.mark.parametrize("line", [
    "Defined on $U$ only.",                 # a word that merely starts the same way
    "Corollaries follow.",
    "the Def. is in the margin",            # not at the start of the line
    "% Def. in a comment",
    "Note that $x > 0$.",                   # "Note" without punctuation is prose
])
def test_running_prose_is_left_alone(line):
    assert bolded(line) == line


def test_a_label_the_model_already_bolded_is_not_bolded_twice():
    line = "\\textbf{Def.} A smooth map."
    assert bolded(line) == line


def test_labels_inside_math_are_not_touched():
    src = ("\\[\nQ. = \\int_M \\omega\n\\]\n"
           "$$\nEx. = 1\n$$\n"
           "\\begin{align*}\nQ. &= 2\n\\end{align*}")
    assert bolded(src) == src


def test_a_label_after_the_math_block_is_still_found():
    out = bolded("\\[\nx = 1\n\\]\nThm. it converges.")
    assert out.endswith("\\textbf{Thm.} it converges.")


def test_indentation_and_line_endings_survive():
    assert bolded("  Def. indented.") == "  \\textbf{Def.} indented."
    assert bold_labels("Def. a.\r\nThm. b.\r\n")[0] == "\\textbf{Def.} a.\r\n\\textbf{Thm.} b.\r\n"


def test_a_page_without_labels_is_returned_unchanged():
    src = "just some prose\n\\begin{center}\n\\includegraphics{p0001_f1.png}\n\\end{center}"
    assert bold_labels(src) == (src, 0)


def test_an_overlong_parenthetical_is_not_swallowed():
    """`Thm (...)` names a result; a whole sentence in brackets is not a name."""
    line = "Eg (" + "x" * 80 + "). body."
    assert bolded(line) == line


@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
def test_a_bolded_page_still_compiles(tmp_path):
    page = ("Lecture 9 20251109 Week 12\n\n"
            "Def. A dist $\\nu$ on $M$ is \\underline{smth}.\n\n"
            "Thm (Frobenius). $\\nu$ is integrable iff involutive.\n\n"
            "Pf. $\\forall p \\in M$, take a chart.\n")
    out, n = bold_labels(page)
    assert n == 4
    result = LatexCompiler(timeout=120).compile(build_document(out), tmp_path / "out")
    assert result.ok, result.errors


# ------------------------------------------------------ numbered Chinese labels
@pytest.mark.parametrize("line,expected", [
    ("定理 1.16 (Bolzano-Weierstrass). 有界无限点集均有极限点。",
     r"\textbf{定理 1.16 (Bolzano-Weierstrass).} 有界无限点集均有极限点。"),
    ("定义 2.2 (Carathéodory 条件). 可测集。",
     r"\textbf{定义 2.2 (Carathéodory 条件).} 可测集。"),
    ("定理4.6（逐项积分）．非负可测函数列",
     r"\textbf{定理4.6（逐项积分）．}非负可测函数列"),
    ("推论 3.10. 上述之每个均可取成。", r"\textbf{推论 3.10.} 上述之每个均可取成。"),
])
def test_a_numbered_chinese_label_is_bolded(line, expected):
    assert bolded(line) == expected


def test_a_second_statement_on_the_same_line_is_bolded_too():
    """A dense summary sheet writes two statements on one handwritten line."""
    out, n = bold_labels("定理 2.1. 外测度次可加。 推论 2.2. 对可数点集有 m*E=0.")
    assert out == r"\textbf{定理 2.1.} 外测度次可加。 \textbf{推论 2.2.} 对可数点集有 m*E=0."
    assert n == 2


@pytest.mark.parametrize("line,expected", [
    ("① 定理 1.16. 有极限点。", r"① \textbf{定理 1.16.} 有极限点。"),
    ("预 定理 1.18. 闭集套。", r"预 \textbf{定理 1.18.} 闭集套。"),
])
def test_a_margin_mark_before_the_label_stays_outside_the_bold(line, expected):
    assert bolded(line) == expected


@pytest.mark.parametrize("line", [
    "这是连续函数延拓定理的推论。",          # 定理 inside a word, no number
    "(Cantor 闭集套定理) is a name",         # in a parenthetical, no number
    r"$\text{定理 1.2}$ inside math",
])
def test_an_unnumbered_or_math_label_is_left_alone(line):
    assert bolded(line) == line


def test_a_chinese_label_the_model_already_bolded_is_not_bolded_twice():
    line = r"\textbf{定理 6.16 (Bessel-Fischer)} 在 $L^2$ 中"
    assert bolded(line) == line


# ------------------------------------------------ display math, followed exactly
def test_two_inline_formulas_side_by_side_are_not_a_display():
    """`$\big($$f$` on a real page made every later line look like display math,
    so nothing after it was bolded."""
    from n2lh.pipeline.style import display_math_depth
    assert display_math_depth(r"定理 5.12. $\big($$f$在$[a,b]$上 $\big)$.", 0) == 0
    src = "\\textbf{定理 5.12.} $\\big($$f$ a.e.$\\big)$.\n定理 5.13 (FTC). 绝对连续.\n"
    out, _ = bold_labels(src)
    assert "\\textbf{定理 5.13 (FTC).}" in out


def test_real_displays_and_spaced_breaks_are_still_tracked():
    from n2lh.pipeline.style import display_math_depth
    assert display_math_depth("$$", 0) == 1
    assert display_math_depth("$$", 1) == 0
    assert display_math_depth("$$ x = 1 $$", 0) == 0
    assert display_math_depth(r"\[", 0) == 1
    assert display_math_depth(r"a \\[2pt] b", 0) == 0
    assert display_math_depth(r"cost \$5 and $x$", 0) == 0

