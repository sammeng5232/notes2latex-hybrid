"""Page setup chosen in Settings, not by editing the .tex afterwards."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from n2lh.compiler.latex import LatexCompiler
from n2lh.config import Settings
from n2lh.pipeline.assembler import build_document
from n2lh.recognition.prompts import FONT_SIZES, PAPER_SIZES, make_preamble

HAS_TEX = shutil.which("latexmk") is not None or shutil.which("pdflatex") is not None


def first_lines(**kw):
    return make_preamble(**kw).splitlines()[:2]


def test_the_default_is_a4_at_11pt_with_one_inch_margins():
    cls, geom = first_lines()
    assert cls == "\\documentclass[11pt,a4paper]{article}"
    assert geom == "\\usepackage[a4paper,margin=1in]{geometry}"


@pytest.mark.parametrize("paper", PAPER_SIZES)
def test_every_offered_paper_size_reaches_both_the_class_and_geometry(paper):
    """geometry lays out for its own default unless it is told the paper too."""
    cls, geom = first_lines(paper=paper)
    assert paper in cls and paper in geom


@pytest.mark.parametrize("pt", FONT_SIZES)
def test_every_offered_font_size_is_applied(pt):
    assert first_lines(font_pt=pt)[0].startswith(f"\\documentclass[{pt}pt,")


def test_margin_landscape_and_two_columns():
    cls, geom = first_lines(paper="a3paper", margin_in=0.5, landscape=True, two_column=True)
    assert cls == "\\documentclass[11pt,a3paper,landscape,twocolumn]{article}"
    assert geom == "\\usepackage[a3paper,margin=0.5in,landscape]{geometry}"


def test_a_margin_is_written_without_a_trailing_zero():
    assert "margin=1.25in" in first_lines(margin_in=1.25)[1]
    assert "margin=2in" in first_lines(margin_in=2.0)[1]


@pytest.mark.parametrize("bad,expected", [
    ({"font_pt": 13}, "11pt"),
    ({"font_pt": "big"}, "11pt"),
    ({"paper": "a9paper"}, "a4paper"),
    ({"paper": ""}, "a4paper"),
])
def test_a_value_that_would_not_compile_falls_back_to_the_default(bad, expected):
    """These come from a settings file a user can edit by hand; a bad one must
    not produce a preamble that fails to compile."""
    assert expected in make_preamble(**bad).splitlines()[0]


@pytest.mark.parametrize("margin,shown", [(0.01, "0.25in"), (99, "4in"), ("wide", "1in")])
def test_an_impossible_margin_is_clamped(margin, shown):
    assert f"margin={shown}" in make_preamble(margin_in=margin).splitlines()[1]


# ------------------------------------------------------------------ settings
def test_settings_carry_the_page_setup_into_the_preamble():
    s = Settings(doc_paper="a3paper", doc_font_pt=12, doc_margin_in=0.75, doc_landscape=True)
    assert s.preamble().splitlines()[0] == "\\documentclass[12pt,a3paper,landscape]{article}"


def test_settings_reject_a_page_setup_that_is_not_offered():
    assert Settings(doc_paper="a9paper").validate()
    assert Settings(doc_margin_in=99).validate()
    assert Settings(doc_font_pt=13).validate()
    assert not Settings(doc_paper="a3paper", doc_margin_in=0.5, doc_font_pt=12).validate()


def test_values_arriving_as_strings_from_the_form_are_coerced():
    s = Settings()
    s.apply_dict({"doc_font_pt": "12", "doc_margin_in": "0.75",
                  "doc_landscape": "true", "doc_two_column": "on"})
    assert (s.doc_font_pt, s.doc_margin_in) == (12, 0.75)
    assert s.doc_landscape is True and s.doc_two_column is True
    assert not s.validate()


# -------------------------------------------------------- the real page size
@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
@pytest.mark.parametrize("paper,w_mm,h_mm", [("a4paper", 210, 297), ("a3paper", 297, 420)])
def test_the_pdf_really_comes_out_at_that_size(tmp_path, paper, w_mm, h_mm):
    tex = build_document("Def. a page.", preamble=make_preamble(paper=paper))
    result = LatexCompiler(timeout=120).compile(tex, tmp_path / paper)
    assert result.ok, result.errors
    import pypdfium2 as pdfium
    page = pdfium.PdfDocument(result.pdf_path)[0]
    # points -> mm, allowing a little rounding in the driver
    assert abs(page.get_width() / 72 * 25.4 - w_mm) < 2
    assert abs(page.get_height() / 72 * 25.4 - h_mm) < 2


@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
def test_landscape_swaps_the_page_dimensions(tmp_path):
    tex = build_document("Def. a wide page.", preamble=make_preamble(landscape=True))
    result = LatexCompiler(timeout=120).compile(tex, tmp_path / "land")
    assert result.ok, result.errors
    import pypdfium2 as pdfium
    page = pdfium.PdfDocument(result.pdf_path)[0]
    assert page.get_width() > page.get_height()
