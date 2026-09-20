"""Chinese (and other CJK) notes need a different engine, chosen automatically.

pdflatex cannot typeset a Chinese character at all: a page of real-analysis
notes headed 实变函数 fails with

    LaTeX Error: Unicode character 实 (U+5B9E) not set up for use with LaTeX

and burns every repair attempt, because no amount of rewriting the LaTeX makes
pdflatex able to draw the glyph. XeLaTeX with ctex compiles the same page.

So the pipeline watches the transcription: the first page carrying CJK switches
the job to XeLaTeX and adds ctex to the preamble. Detecting it from the text
rather than asking the user means a mixed set of notes, or one Chinese page in
an English document, is handled without anyone having to notice.
"""

from __future__ import annotations

import re

# Han, the CJK punctuation block (，。、《》), fullwidth forms, kana and hangul.
# Kana and hangul are included so such a page is at least sent to an engine that
# can draw them; ctex targets Chinese, and a CJK font supplies the rest.
_CJK = re.compile(
    "[　-〿぀-ヿ㐀-䶿一-鿿"
    "가-힯豈-﫿︰-﹏＀-￯]")

# ctex loads a CJK font and the punctuation/spacing rules that go with it.
CJK_PACKAGE = "\\usepackage[UTF8]{ctex}   % Chinese: needs XeLaTeX or LuaLaTeX"
CJK_ENGINES = ("xelatex", "lualatex")


def has_cjk(text: str) -> bool:
    """True when the text contains characters pdflatex cannot typeset."""
    return bool(_CJK.search(text or ""))


def add_cjk(preamble: str) -> str:
    """Put ctex into a preamble, right after \\documentclass."""
    if "ctex" in preamble or "xeCJK" in preamble:
        return preamble
    lines = preamble.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("\\documentclass"):
            lines.insert(i + 1, CJK_PACKAGE + "\n")
            return "".join(lines)
    return CJK_PACKAGE + "\n" + preamble


def engine_for(current: str) -> str:
    """The engine to compile CJK with, keeping the user's if it already works."""
    return current if current in CJK_ENGINES else "xelatex"
