from n2lh.pipeline.sanitize import comment_out, sanitize_body


def test_plain_body_is_unchanged():
    body = "Def. An $n$-dim mfd.\n\n\\begin{align*}\nx &= 1\n\\end{align*}"
    assert sanitize_body(body) == body


def test_full_document_reduced_to_its_body():
    doc = ("\\documentclass{article}\n\\usepackage{amsmath, tikz}\n\\begin{document}\n"
           "Hello $x$.\n\\end{document}\n")
    assert sanitize_body(doc) == "Hello $x$."


def test_stray_preamble_lines_dropped_without_a_document_wrapper():
    text = "\\usepackage[utf8]{inputenc}\n\\documentclass[11pt]{article}\nBody text."
    assert sanitize_body(text) == "Body text."


def test_code_fence_lines_dropped_and_empty_stays_empty():
    assert sanitize_body("```latex\nx = 1\n```") == "x = 1"
    assert sanitize_body("") == ""
    assert sanitize_body("\\documentclass{article}") == ""


def test_comment_out_prefixes_every_line_and_neutralises_latex():
    out = comment_out("\\begin{align*}\n\nx &= 1")
    assert out.splitlines() == ["% \\begin{align*}", "%", "% x &= 1"]