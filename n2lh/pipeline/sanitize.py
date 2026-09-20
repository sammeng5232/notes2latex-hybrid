"""Clean model output so it is safe to splice into the shared document body.

The prompts tell the VLM to return body content only, but models routinely
ignore that -- especially on repair passes, where a "fix this LaTeX" request can
come back as a whole standalone document. A stray ``\\documentclass`` or
``\\usepackage`` inside the body is fatal (``LaTeX Error: Can be used only in
preamble``), and because every page is compiled as part of the accumulated
document, one such page used to break every page after it.
"""

from __future__ import annotations

import re

_BEGIN_DOC = re.compile(r"\\begin\{document\}")
_END_DOC = re.compile(r"\\end\{document\}")
# One-line preamble commands (optionally with [opts]{args}); anything the
# preamble in prompts.PREAMBLE_TEX already provides.
_PREAMBLE_LINE = re.compile(
    r"^[ \t]*\\(?:documentclass|usepackage|RequirePackage|pgfplotsset)\b[^\n]*\n?",
    re.MULTILINE,
)
_FENCE = re.compile(r"^```[a-zA-Z]*[ \t]*\n?", re.MULTILINE)


def sanitize_body(latex: str) -> str:
    """Return ``latex`` reduced to body content.

    - If a full document was returned, keep only what is between
      ``\\begin{document}`` and ``\\end{document}``.
    - Drop ``\\documentclass`` / ``\\usepackage`` / ``\\RequirePackage`` lines and
      any stray ``\\begin{document}`` / ``\\end{document}`` markers.
    - Drop leftover markdown code-fence lines.
    """
    if not latex:
        return ""
    text = latex
    begin = _BEGIN_DOC.search(text)
    if begin:
        text = text[begin.end():]
    end = _END_DOC.search(text)
    if end:
        text = text[:end.start()]
    text = _PREAMBLE_LINE.sub("", text)
    text = _BEGIN_DOC.sub("", text)
    text = _END_DOC.sub("", text)
    text = _FENCE.sub("", text)
    return text.strip("\n").rstrip() if text.strip() else ""


def comment_out(latex: str) -> str:
    """Prefix every line with ``% `` so raw text can ride along in the .tex
    (visible to someone reading the source) without being compiled."""
    return "\n".join("% " + line if line.strip() else "%" for line in latex.splitlines())
