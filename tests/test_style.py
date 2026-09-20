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
