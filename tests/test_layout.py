"""Faults that compile cleanly and are still wrong on the printed page."""

from __future__ import annotations

import re
import shutil

import pytest

from n2lh.compiler.latex import LatexCompiler
from n2lh.pipeline.assembler import build_document
from n2lh.pipeline.layout import fit_wide_tables, fix_indicator, tidy_layout
from n2lh.recognition.prompts import make_preamble

HAS_TEX = shutil.which("latexmk") is not None or shutil.which("pdflatex") is not None


def overfull_inches(tmp_path, body, name="doc"):
    """Compile and return how far the worst line sticks out of the text block."""
    out = tmp_path / name
    result = LatexCompiler(timeout=240, cleanup_aux=False).compile(
        build_document(body, preamble=make_preamble()), out)
    assert result.ok, result.errors
    log = next(out.glob("*.log"), None)
    text = log.read_text("utf-8", errors="replace") if log else ""
    over = [float(x) for x in re.findall(r"Overfull .hbox \(([\d.]+)pt too wide\)", text)]
    return max(over) / 72.27 if over else 0.0


# ------------------------------------------------------------------ indicator
@pytest.mark.parametrize("written", [
    r"\mathbb{1}", r"\mathbb {1}", r"\mathbb1", r"\mathbb 1", r"\mathds{1}",
])
def test_every_spelling_of_the_indicator_becomes_mathbbm(written):
    out, n = fix_indicator(f"$f(x) = p{written}_{{x>0}}$")
    assert out == r"$f(x) = p\mathbbm{1}_{x>0}$"
    assert n == 1 or written.startswith(r"\mathds")


def test_blackboard_letters_are_left_alone():
    src = r"$\mathbb{R}^n, \mathbb{Z}, \mathbb{1}_{A}, \mathbb{N}$"
    out, n = fix_indicator(src)
    assert out == r"$\mathbb{R}^n, \mathbb{Z}, \mathbbm{1}_{A}, \mathbb{N}$"
    assert n == 1


def test_a_page_with_no_indicator_is_untouched():
    assert fix_indicator(r"$\mathbb{R}$") == (r"$\mathbb{R}$", 0)


# --------------------------------------------------------------- wide tables
def test_a_table_is_wrapped_so_it_cannot_overflow():
    out, n = fit_wide_tables("text\n\\begin{tabular}{ll} a & b \\\\ \\end{tabular}\ntext")
    assert n == 1
    assert out == "text\n\\fitpage{\\begin{tabular}{ll} a & b \\\\ \\end{tabular}}\ntext"


def test_a_nested_table_is_covered_by_the_outer_wrap_only():
    src = ("\\begin{tabular}{ll} a & \\begin{tabular}{c} x \\end{tabular} \\\\ "
           "\\end{tabular}")
    out, n = fit_wide_tables(src)
    assert n == 1 and out.count("\\fitpage") == 1
    assert out.startswith("\\fitpage{\\begin{tabular}")


def test_running_twice_does_not_wrap_twice():
    once, _ = fit_wide_tables("\\begin{tabular}{l} a \\end{tabular}")
    twice, n = fit_wide_tables(once)
    assert n == 0 and twice == once


def test_several_tables_are_each_wrapped():
    src = ("\\begin{tabular}{l} a \\end{tabular} and \\begin{tabular}{l} b \\end{tabular}")
    out, n = fit_wide_tables(src)
    assert n == 2 and out.count("\\fitpage{") == 2


def test_an_unclosed_table_is_left_for_the_autofix_pass():
    src = "\\begin{tabular}{ll} a & b"
    assert fit_wide_tables(src) == (src, 0)


def test_prose_and_math_are_not_touched():
    src = "Def. $x \\in \\mathbb{R}$\n\\[ a = b \\]"
    assert fit_wide_tables(src) == (src, 0)


# ------------------------------------------------------------------ together
def test_tidy_layout_reports_what_it_changed():
    src = "\\begin{tabular}{l} $\\mathbb{1}_{x>0}$ \\end{tabular}"
    out, changes = tidy_layout(src)
    assert "\\mathbbm{1}" in out and "\\fitpage{" in out
    assert changes == ["indicator: 1 \\mathbb{1} -> \\mathbbm{1}",
                       "fitted 1 table(s) to the text width"]


def test_a_clean_page_reports_nothing():
    assert tidy_layout("Def. a map $f$.")[1] == []


# ------------------------------------------------- what it looks like on paper
@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
def test_the_indicator_is_a_different_glyph_after_the_fix(tmp_path):
    """\\mathbb{1} compiles happily and prints \\nVdash, so only the glyph tells us."""
    import pypdfium2 as pdfium

    def render(body):
        out = tmp_path / str(abs(hash(body)))
        r = LatexCompiler(timeout=240).compile(
            build_document(body, preamble=make_preamble()), out)
        assert r.ok, r.errors
        return pdfium.PdfDocument(r.pdf_path)[0].render(scale=2).to_pil().tobytes()

    wrong = render(r"$\mathbb{1}_{\{x \ge 0\}}$")
    right = render(r"$\mathbbm{1}_{\{x \ge 0\}}$")
    assert wrong != right                       # the page really does change
    assert render(fix_indicator(r"$\mathbb{1}_{\{x \ge 0\}}$")[0]) == right


@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
def test_a_table_too_wide_for_the_page_is_brought_back_inside(tmp_path):
    """The real one: six columns of dense maths, 9.36in past a 6.27in text block."""
    cols = 10
    row = " & ".join([r"$f(x) = \frac{1}{\theta} e^{-x/\theta}\mathbbm{1}_{x \ge 0}$"] * cols)
    table = "\\begin{tabular}{|" + "l|" * cols + "}\n" + row + " \\\\\n\\end{tabular}"
    before = overfull_inches(tmp_path, table, "before")
    after = overfull_inches(tmp_path, fit_wide_tables(table)[0], "after")
    assert before > 3, f"fixture is not wide enough to test ({before:.2f}in)"
    assert after < 0.05, f"still overflowing by {after:.2f}in"


@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
def test_a_table_that_already_fits_is_not_shrunk(tmp_path):
    """\\fitpage scales down only; a narrow table must come out its natural size."""
    import pypdfium2 as pdfium

    def width_of(body):
        out = tmp_path / str(abs(hash(body)))
        r = LatexCompiler(timeout=240).compile(
            build_document(body, preamble=make_preamble()), out)
        assert r.ok, r.errors
        page = pdfium.PdfDocument(r.pdf_path)[0]
        # the rendered ink width tells us whether it was scaled
        im = page.render(scale=2).to_pil().convert("L").point(lambda v: 255 if v < 200 else 0)
        box = im.getbbox()
        return box[2] - box[0]

    table = "\\begin{tabular}{ll} a & b \\\\ c & d \\\\ \\end{tabular}"
    assert abs(width_of(table) - width_of(fit_wide_tables(table)[0])) <= 2
