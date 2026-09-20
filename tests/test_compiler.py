from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from n2lh.compiler.latex import LatexCompiler, clean_aux, parse_log
from n2lh.pipeline.assembler import build_document

HAS_TEX = shutil.which("latexmk") is not None or shutil.which("pdflatex") is not None


SAMPLE_LOG = """This is pdfTeX, Version 3.141592653
entering extended mode
(./document.tex
LaTeX2e <2023-11-01>
! Undefined control sequence.
l.42 \\badcmd

! LaTeX Error: \\begin{equation*} on input line 8 ended by \\end{document}.
See the LaTeX manual for further explanation.
==> Fatal error occurred, no output PDF file produced!
"""


def test_parse_log_extracts_errors_with_lines():
    errors = parse_log(SAMPLE_LOG)
    assert len(errors) == 2
    assert "Undefined control sequence" in errors[0].message
    assert errors[0].line == 42
    assert "ended by" in errors[1].message


def test_parse_log_empty():
    assert parse_log("no errors here\njust noise") == []


def test_clean_aux_removes_generated_files_only(tmp_path: Path):
    keep = tmp_path / "document.tex"
    keep.write_text("x")
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"%pdf")
    for name in ("document.aux", "document.log", "document.out",
                 "document.synctex.gz", "document.fdb_latexmk", "document.fls"):
        (tmp_path / name).write_text("junk")
    minted = tmp_path / "_minted-document"
    minted.mkdir()
    (minted / "abc.pyg").write_text("x")

    clean_aux(tmp_path)

    assert keep.exists() and pdf.exists()
    for name in ("document.aux", "document.log", "document.out",
                 "document.synctex.gz", "document.fdb_latexmk", "document.fls"):
        assert not (tmp_path / name).exists(), name
    assert not minted.exists()


@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
def test_real_compile_roundtrip_with_cleanup(tmp_path: Path):
    compiler = LatexCompiler(timeout=90)
    tex = build_document("Hello world. $x^2 + y^2 = z^2$")
    result = compiler.compile(tex, tmp_path)
    assert result.ok, f"expected ok, errors: {result.errors}"
    assert result.pdf_path is not None and result.pdf_path.exists()
    # Strict cleanup: no aux/log/out beside the source.
    assert not (tmp_path / "document.aux").exists()
    assert not (tmp_path / "document.log").exists()
    assert not (tmp_path / "document.out").exists()
    assert (tmp_path / "document.tex").exists()


@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
def test_real_compile_failure_reports_errors(tmp_path: Path):
    compiler = LatexCompiler(timeout=90)
    tex = build_document("this is broken: \\badcmdxyz")
    result = compiler.compile(tex, tmp_path)
    assert not result.ok
    assert result.errors
    assert any("Undefined" in e.message for e in result.errors)
    assert not (tmp_path / "document.aux").exists()


@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
def test_preamble_supports_commutative_diagrams_and_common_math_macros(tmp_path: Path):
    """The models emit tikzcd / \\mathscr / \\bm / \\cancel for topology notes. A real
    43-page job lost a page for good because tikz-cd was not in the preamble."""
    body = (
        "$\\mathscr{F}$ and $\\bm{v}$ and $\\cancel{x}$.\n\n"
        "\\begin{tikzcd}\n"
        "M_1 \\arrow[r, \"\\psi\"] \\arrow[d] & M_2 \\arrow[d] \\\\\n"
        "\\mathbb{R} \\arrow[r] & \\mathbb{R}\n"
        "\\end{tikzcd}\n"
    )
    result = LatexCompiler(timeout=120).compile(build_document(body), tmp_path)
    assert result.ok, f"preamble is missing a package: {result.errors}"