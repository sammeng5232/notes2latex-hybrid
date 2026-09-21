from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from n2lh.compiler.latex import LatexCompiler
from n2lh.pipeline.assembler import build_document
from n2lh.pipeline.autofix import autofix_latex

HAS_TEX = shutil.which("latexmk") is not None or shutil.which("pdflatex") is not None
DATA = Path(__file__).parent / "data"


def fix(text):
    return autofix_latex(text)


# ------------------------------------------------------------------ ampersands
def test_bare_ampersand_in_prose_is_escaped():
    out, changes = fix("Hausdorff & 2nd countable, so $M$ & $N$.")
    assert out == "Hausdorff \\& 2nd countable, so $M$ \\& $N$."
    assert changes == ["escaped 2 bare '&'"]


def test_ampersand_is_left_alone_inside_alignment_environments():
    src = ("\\begin{align*}\nx &= 1 \\\\\ny &= 2\n\\end{align*}\n"
           "$$\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$$\n"
           "\\begin{tabular}{ll} a & b \\end{tabular}\n"
           "\\begin{tikzpicture}\\matrix{ a & b \\\\ }; \\end{tikzpicture}")
    assert fix(src) == (src, [])


def test_escaped_ampersand_and_comments_are_untouched():
    src = "already \\& fine % and a & in a comment\nnext line"
    assert fix(src) == (src, [])


def test_ampersand_outside_the_alignment_env_but_inside_another_env_is_escaped():
    out, _ = fix("\\begin{center}rock & roll\\end{center}")
    assert "rock \\& roll" in out


# ---------------------------------------------------------------- environments
def test_unclosed_environment_is_closed_at_the_end():
    out, changes = fix("\\begin{proof}\nSince $x$ is small.")
    assert out.rstrip().endswith("\\end{proof}")
    assert changes == ["closed 1 unclosed environment(s)"]


def test_crossed_environments_are_repaired_in_nesting_order():
    out, _ = fix("\\begin{center}\\begin{itemize}\\item a\\end{center}")
    assert out.index("\\end{itemize}") < out.index("\\end{center}")


def test_stray_end_is_removed():
    out, changes = fix("text \\end{proof} more")
    assert "\\end{proof}" not in out and changes == ["removed 1 stray \\end"]


# ------------------------------------------------------------------- math mode
def test_aligned_in_text_mode_is_wrapped_in_display_math():
    out, changes = fix("So\n\\begin{aligned} a &= b \\\\ c &= d \\end{aligned}\nDone.")
    assert "\\[\n\\begin{aligned}" in out and "\\end{aligned}\n\\]" in out
    assert changes == ["wrapped 1 math-only block(s) in \\[ \\]"]


@pytest.mark.parametrize("wrapper_open,wrapper_close", [
    ("$$", "$$"), ("\\[", "\\]"), ("\\begin{equation*}", "\\end{equation*}"), ("$", "$")])
def test_aligned_already_in_math_is_not_wrapped_again(wrapper_open, wrapper_close):
    src = f"{wrapper_open}\\begin{{aligned}} a &= b \\end{{aligned}}{wrapper_close}"
    assert fix(src) == (src, [])


def test_line_break_with_optional_length_is_not_mistaken_for_display_math():
    src = "$$\\begin{aligned} a &= b \\\\[2pt] c &= d \\end{aligned}$$"
    assert fix(src) == (src, [])


def test_valid_document_body_comes_back_unchanged():
    src = ("Def. $f: M \\to N$ is \\underline{smooth}.\n\n\\begin{itemize}\n\\item a\n\\end{itemize}\n\n"
           "\\[ \\forall p \\in M, \\exists U \\]")
    assert fix(src) == (src, [])


# ------------------------------------------------------- real LaTeX, real faults
@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
def test_autofix_makes_the_three_real_failure_modes_compile(tmp_path):
    comp = LatexCompiler(timeout=120)
    faulty = ("Hausdorff & 2nd countable.\n\n"
              "\\begin{aligned} a &= b \\\\ c &= d \\end{aligned}\n\n"
              "\\begin{proof}\nSince it is small.")
    assert not comp.compile(build_document(faulty), tmp_path / "before").ok
    fixed, changes = fix(faulty)
    assert len(changes) == 3
    result = comp.compile(build_document(fixed), tmp_path / "after")
    assert result.ok, result.errors


@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
def test_real_page_that_four_model_repairs_could_not_fix_now_compiles(tmp_path):
    """Page 29 of the user's notes: the model bounced between a stray &, `aligned`
    outside math and an unclosed `proof` through every repair."""
    latex = (DATA / "page29_failing.tex").read_text("utf-8")
    comp = LatexCompiler(timeout=120)
    assert not comp.compile(build_document(latex), tmp_path / "before").ok
    fixed, changes = fix(latex)
    assert changes
    result = comp.compile(build_document(fixed), tmp_path / "after")
    assert result.ok, result.errors

# ------------------------------------------------------------- column specs
def test_array_spec_is_widened_to_the_widest_row():
    out, changes = fix("$$\\begin{array}{cc} a & b & c \\\\ d & e \\end{array}$$")
    assert "\\begin{array}{ccc}" in out
    assert changes == ["widened 1 array/tabular column spec(s)"]


def test_tabular_spec_keeps_rules_and_paragraph_columns_when_padding():
    out, _ = fix("\\begin{tabular}{|l|p{3cm}|} a & b & c \\\\ \\end{tabular}")
    assert "\\begin{tabular}{|l|p{3cm}|c}" in out


def test_nested_tables_are_counted_separately():
    src = ("$$\\begin{array}{cc} \\begin{array}{cc} 1 & 2 & 3 \\end{array} & x \\\\ y & z "
           "\\end{array}$$")
    out, changes = fix(src)
    assert out.startswith("$$\\begin{array}{cc} \\begin{array}{ccc}")
    assert changes == ["widened 1 array/tabular column spec(s)"]


@pytest.mark.parametrize("src", [
    "$$\\begin{array}{cc} a & b \\\\ c & d \\end{array}$$",                        # fits
    "$$\\begin{array}{c} \\multicolumn{2}{c}{a} & b \\end{array}$$",                 # spans
    "\\begin{tabular}{*{3}{c}} a & b & c & d \\end{tabular}",                      # exotic spec
])
def test_column_specs_that_fit_or_cannot_be_counted_are_left_alone(src):
    assert fix(src) == (src, [])


# ---------------------------------------------------------------------- hash
def test_bare_hash_is_escaped_but_an_escaped_one_is_not_doubled():
    out, changes = fix("Induct on # charts, then \\# charts.")
    assert out == "Induct on \\# charts, then \\# charts."
    assert changes == ["escaped 1 bare '#'"]


def test_hash_is_left_alone_in_a_document_that_defines_macros():
    src = "\\newcommand{\\twice}[1]{#1#1} \\twice{a}"
    assert fix(src) == (src, [])


# ------------------------------------------------------------------ unicode
def test_unicode_symbol_in_text_mode_becomes_an_inline_formula():
    out, changes = fix("So p ∤ n.")
    assert out == "So p $\\nmid$ n."
    assert changes == ["replaced 1 unsupported Unicode symbol(s)"]


def test_unicode_symbols_inside_math_become_bare_macros():
    out, _ = fix("$x ∈ ℝ, ∀ε > 0$ and \\[ a ≤ b ⇒ c \\]")
    assert out == "$x \\in  \\mathbb{R} , \\forall \\varepsilon  > 0$ and \\[ a \\le  b \\Rightarrow  c \\]"


def test_latin1_symbols_are_only_touched_inside_math():
    assert fix("2 × 3 · 4")[0] == "2 × 3 · 4"
    assert fix("$2 × 3$")[0] == "$2 \\times  3$"


def test_unicode_in_comments_and_ordinary_text_is_untouched():
    src = "% ∀ is fine in a comment\nCafé — naïve “quotes”."
    assert fix(src) == (src, [])


# -------------------------------------------- the real pages of the 2026-09-20 run
@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
def test_page_with_a_too_narrow_array_now_compiles(tmp_path):
    """Page 42: `array{ccccccc}` holding rows of 9 cells; four model repairs never widened it."""
    latex = (DATA / "page42_failing.tex").read_text("utf-8")
    comp = LatexCompiler(timeout=120)
    assert not comp.compile(build_document(latex), tmp_path / "before").ok
    fixed, changes = fix(latex)
    assert "widened 2 array/tabular column spec(s)" in changes
    result = comp.compile(build_document(fixed), tmp_path / "after")
    assert result.ok, result.errors


@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
def test_page_with_a_raw_unicode_symbol_and_a_bare_hash_compiles(tmp_path):
    faulty = "Sps $p ∤ n$ and $\\forall x ∈ ℝ$. Induct on # charts."
    comp = LatexCompiler(timeout=120)
    assert not comp.compile(build_document(faulty), tmp_path / "before").ok
    fixed, _ = fix(faulty)
    result = comp.compile(build_document(fixed), tmp_path / "after")
    assert result.ok, result.errors

# ------------------------------------------- sub/superscripts in text mode
def test_a_superscript_in_text_mode_is_put_into_math():
    r"""Real page: \textbf{定理 6.6 (L^p 完备性)} -> Missing $ inserted."""
    out, changes = fix(r"\textbf{Thm 6.6 (L^p completeness)} holds.")
    assert out == r"\textbf{Thm 6.6 ($L^p$ completeness)} holds."
    assert changes == ["put 1 text-mode sub/superscript(s) into math"]


def test_a_subscript_in_text_mode_too():
    assert fix("the sequence x_n converges")[0] == "the sequence $x_n$ converges"


def test_a_braced_operand_is_taken_whole():
    assert fix("norm L^{p+1} here")[0] == "norm $L^{p+1}$ here"


def test_a_macro_operand_is_taken_whole():
    assert fix(r"see \alpha_1 there")[0] == r"see $\alpha_1$ there"


def test_the_chinese_page_that_failed_after_the_engine_switch():
    r"""Once XeLaTeX could draw the characters, this was all that still broke."""
    src = r"\textbf{定理 6.6 (L^p 完备性)} $\|f\|_p$ 为 Banach 空间。"
    out, changes = fix(src)
    assert r"($L^p$ 完备性)" in out
    assert changes == ["put 1 text-mode sub/superscript(s) into math"]


@pytest.mark.parametrize("src", [
    "$L^p$ is complete",
    r"\[ x_n \to x \]",
    r"\begin{align*} a^2 &= b \end{align*}",
    r"a \^{} b",                      # an escaped caret is a real character
    r"file\_name in prose",           # an escaped underscore likewise
])
def test_scripts_already_in_math_or_escaped_are_untouched(src):
    assert fix(src) == (src, [])


# ---------------------------------------------------- multi-paragraph textcolor
def test_a_textcolor_spanning_a_blank_line_becomes_a_color_group():
    src = ("\\textcolor{magenta}{\n中英词汇对照表\n\nCantor 闭集套定理 \\hfill Closed nested "
           "sets theorem\n}")
    out, changes = fix(src)
    assert out == ("{\\color{magenta}\n中英词汇对照表\n\nCantor 闭集套定理 \\hfill Closed nested "
                   "sets theorem\n}")
    assert changes and "multi-paragraph" in changes[0]


def test_a_single_paragraph_textcolor_is_left_alone():
    src = "keep \\textcolor{red}{this} exactly"
    out, changes = fix(src)
    assert out == src and changes == []


def test_the_rewrite_keeps_braces_balanced_with_nested_textcolor():
    src = "\\textcolor{pink}{a\n\nb \\textcolor{blue}{c} d\n\ne}"
    out, _ = fix(src)
    assert out.startswith("{\\color{pink}")
    assert out.endswith("e}")
    assert "\\textcolor{blue}{c}" in out
    assert out.count("{") == out.count("}")


def test_an_unterminated_textcolor_is_left_to_the_general_pass():
    # No matching close brace: nothing safe to rewrite, the token pass will
    # report the imbalance instead.
    src = "\\textcolor{red}{unterminated..."
    out, changes = fix(src)
    assert "{\\color" not in out or "unterminated" in out
