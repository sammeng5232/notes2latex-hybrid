"""Prompts (shared by the VLM engine) and the default preamble."""

from __future__ import annotations

import re
from typing import List, Optional

_PREAMBLE_TEMPLATE = r"""\documentclass[__CLASSOPTS__]{article}
\usepackage[__GEOMETRY__]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{mathrsfs}   % \mathscr
\usepackage{bm}
\usepackage{bbm}        % \mathbbm{1}: amssymb has no blackboard digit, and
                        % \mathbb{1} silently renders as the symbol \nVdash
\usepackage{cancel}
\usepackage{xcolor}
\usepackage{graphicx}   % hand-drawn figures cropped from the page image
% Figures live in <output>/figures. Per-page compiles run one or two levels deeper
% (pages/.compile-pNNNN for jobs, .pages/.compile-pNNNN for the CLI), so search all.
\graphicspath{{figures/}{../../figures/}{../../out/figures/}}
\usepackage{tikz}
\usepackage{tikz-cd}    % commutative diagrams: models routinely emit tikzcd for these notes
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
% Notes are a run of short items (Def./Eg./Thm./Pf.): a gap and no indent reads best.
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
% Fallback so a \figbox the app did not turn into a picture still compiles.
\newcommand{\figbox}[4]{\par\noindent\fbox{[figure]}\par}
% Shrink a block to the text width, but only when it is wider: a summary table of
% distributions ran 9 inches past the margin and simply vanished off the page.
\newcommand{\fitpage}[1]{\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{#1}}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{definition}{Definition}
\newtheorem{proposition}{Proposition}
\newtheorem{remark}{Remark}
\newtheorem{example}{Example}
"""


# Paper sizes the article class understands, plus the two-column and landscape
# switches, all chosen in Settings rather than by editing the .tex afterwards.
PAPER_SIZES = ("a4paper", "a3paper", "a5paper", "b5paper",
               "letterpaper", "legalpaper", "executivepaper")
FONT_SIZES = (10, 11, 12)


def make_preamble(font_pt: int = 11, paper: str = "a4paper", margin_in: float = 1.0,
                  landscape: bool = False, two_column: bool = False) -> str:
    """The document preamble for the chosen page setup.

    Anything invalid falls back to the default rather than producing a preamble
    that will not compile: these come from settings a user can edit by hand.
    """
    if font_pt not in FONT_SIZES:
        font_pt = 11
    if paper not in PAPER_SIZES:
        paper = "a4paper"
    try:
        margin = float(margin_in)
    except (TypeError, ValueError):
        margin = 1.0
    margin = min(4.0, max(0.25, margin))

    opts = [f"{font_pt}pt", paper]
    if landscape:
        opts.append("landscape")
    if two_column:
        opts.append("twocolumn")
    # geometry needs the paper size too, or it lays out for its own default.
    geometry = [paper, f"margin={margin:g}in"]
    if landscape:
        geometry.append("landscape")
    return (_PREAMBLE_TEMPLATE
            .replace("__CLASSOPTS__", ",".join(opts), 1)
            .replace("__GEOMETRY__", ",".join(geometry), 1))


PREAMBLE_TEX = make_preamble()

TRANSCRIBE_SYSTEM = r"""You transcribe a scanned page of HANDWRITTEN mathematics lecture notes into LaTeX.
Output ONLY the LaTeX body for this page: no preamble, no \documentclass, no \begin{document}, no commentary, no code fences.

CONTENT
- Copy the words and formulas exactly as written. Keep the author's abbreviations ("mfd", "Eg.", "Def.", "Thm.", "Pf.", "iff", "s.t.", "w/").
- TRANSCRIBE, DO NOT NORMALISE. Never replace what is written with the version you know. Keep the author's own letters, parameters and convention even when another is more common, and never convert a formula into an equivalent one. If the page writes the exponential distribution as Exp(theta) with density (1/theta)e^{-x/theta}, mean theta and variance theta^2, write exactly that - NOT Exp(lambda) with mean 1/lambda. If it writes Gamma(alpha, theta) with mean alpha*theta, do not turn it into Gamma(alpha, beta) with mean alpha/beta. The same holds for every other choice on the page: a reciprocal, a sign, an index range, a normalising constant, the side a transpose sits on. Read the symbol that is there and copy it.
- The notes may be written in Chinese, or mix Chinese with English and mathematics. Transcribe the Chinese exactly as it stands, in Chinese: never translate it, transliterate it, or replace a Chinese term with its English name. Keep Chinese punctuation (，。、《》：) as written.
- Do not correct, complete or re-derive anything. If a step looks wrong, unfinished or unconventional, transcribe it as written: these are someone's notes, and a silent "fix" is indistinguishable from a misreading and cannot be caught by compiling.
- Codes and identifiers - course codes, dates, years, reference numbers - are copied character by character, and every character is counted before you write it. Do not drop or add one to make a more familiar-looking code. A bare vertical stroke inside a code is the digit 1, not a separator: "CUHK/STAT2001A/2324/1" is easily misread as "STAT200A" when the 1 is written as a plain stroke.
- Inline math in $...$, display math in \[ ... \] or equation*/align*. Start a new paragraph (blank line) for every new item (Def., Eg., Thm., Pf., Rmk.) and every new handwritten paragraph.
- A handwritten "&" between words means "and": write \& (a bare & breaks LaTeX outside tables and align).
- An indicator (a bold/blackboard 1, often written 1 with a doubled stroke, as in 1_{x>0}) is \mathbbm{1}, never \mathbb{1}: the blackboard alphabet has no digits and \mathbb{1} comes out as a struck-through turnstile.
- An upside-down A is \forall. It is very often mistaken for the letter v or V: "Vp∈M", "Vx≠0", "Va>0", "V chart", "Vw" all mean \forall p\in M, \forall x\neq 0, \forall a>0, \forall \text{ chart}, \forall\omega. A backwards E is \exists. Never output a lone v or V where a quantifier is meant.

TABLES - a summary table is where content goes missing
- Transcribe every column the table has, including a repetitive one (an F(x) column reading "has no closed form" for half the rows is still a column). Count the columns in the header before you start, and give every row that many cells.
- A cell contains exactly what is written in it, even when the column heading leads you to expect something else. If the M(t) column of one row actually holds a covariance and a correlation, transcribe those - do NOT supply the moment generating function that "should" be there. Writing a formula the page does not contain is the worst thing you can do: it is correct-looking and undetectable.
- A cell often holds several formulas separated by commas. Read to the right-hand edge of every cell and transcribe all of them, not just the first.
- Keep the author's subscripts: x_i, x_j stays x_i, x_j and must not be renumbered to x_1, x_2.
- A table can sit in a CORNER or the margin of the page, beside or above the body text (a vocabulary list, a summary box). It is still a table and still content: transcribe all of its rows and columns where it appears in reading order -- never drop it for being off to the side, and never wrap it in \begin{center} unless it is centered on the page.

FORMATTING - reproduce how the page looks
- Underlined text -> \underline{...}. Underlined words, terms being defined and headings are common; do not drop underlines.
- COLORED INK is part of the note: red and blue pens mark corrections, emphasis and status. When text, an underline, a strike, a box or a margin mark is written in colored ink, wrap exactly that content in \textcolor{<color>}{...} with a standard color name (red, green, blue, cyan, magenta, yellow, brown, lime, olive, orange, pink, purple, teal, violet, gray). A colored underline is still an underline: \textcolor{red}{\underline{Riemann integrable}}. Black ink gets no \textcolor. Never invent a color that is not on the page, and never drop one that is: the color is information (red often marks corrections and crossed-out work, blue the terms being defined). A \textcolor argument must NOT contain a blank line: color a multi-paragraph block as {\color{<name>} ... } or as one \textcolor per paragraph.
- Text visibly centered on the page (titles, cover-page lines) -> \begin{center} ... \end{center}, one handwritten line per line, separated by \\.
- Text written larger than the body text -> a size command: \LARGE for a page title, \Large for a heading, \large for a subheading, e.g. {\Large Topology of Manifolds}. Never use size commands for ordinary text.
- A standalone heading such as "Manifolds with Boundaries" -> \subsection*{...} (keep \underline{} if it is underlined).
- \textbf{...} for bold, \textit{...} for italic, itemize/enumerate for bulleted or numbered lists.
- A label that opens a statement - Def. Defn. Thm. Cor. Lem. Prop. Pf. Eg. Ex. Rmk. Note. Claim. - and the parenthetical naming it -> bold, including the punctuation: \textbf{Def.}, \textbf{Thm (Poincare duality).}, \textbf{Pf.}. A lecture heading such as "Lecture 9 20251109 Week 12" -> \textbf{Lecture 9 20251109 Week 12} on its own line.

LAYOUT - keep content where the page puts it
- A margin column - a narrow band down one side of the page holding circled numbers, single status characters, ticks or crosses beside the body text - is content, not decoration. Transcribe each margin mark inline at the point of the body it stands beside, e.g. \textbf{② 难} where its section starts. Never drop margin marks, and never gather them into a separate list of their own.
- Transcribe only the margin marks that are actually there. If the page numbers six sections ① to ⑥, the items after the sixth carry no circled number: do not continue the numbering pattern yourself, and do not renumber the body's own labels to match.
- Text struck through by a stroke -> \cancel{...}, colored as the ink that struck it: \textcolor{red}{\cancel{②}}.
- A table or boxed block sitting in a corner of the page beside body text is NOT centered: transcribe it without \begin{center}, at the point in reading order where the surrounding text reaches it, and keep transcribing the body text that runs beside it - do not move that text all above or below the table.

FIGURES - do NOT redraw them
- Never draw a figure with TikZ or an array. For every hand-drawn picture, graph, sketch or arrow/commutative diagram, put one line \figbox{x0}{y0}{x1}{y1} where the figure appears in reading order.
- x0,y0,x1,y1 are integers from 0 to 1000 (origin TOP-LEFT, x right, y down): roughly where the figure is. An approximate box is fine - what matters is that there is exactly one \figbox per figure, in the right place in the text; the exact crop is measured separately.
- Text lines, single equations and matrices are not figures. Transcribe any caption or text next to a figure as normal text.

Continue seamlessly from the context you are given."""

TABLE_SYSTEM = r"""You transcribe ONE ruled table from a crop of handwritten lecture notes into a LaTeX tabular.
Output ONLY the tabular environment: no preamble, no surrounding text, no commentary, no code fences.

- Count the columns in the header row first, and give every row exactly that many cells. Transcribe that header as the table's first row (its first cell is usually blank). Include a column that is repetitive or often empty.
- Copy each cell exactly as written. Do not normalise notation, do not convert a formula to an equivalent one, and do not supply a formula the cell does not contain because the column heading suggests it.
- A cell often holds two formulas separated by a comma; read to the right-hand edge and transcribe both. Copy each function name letter for letter: Cov, Cor and Corr are three different spellings and the number of r's matters, so a Corr must not be shortened to Cor, and neither must become Cov. A handwritten r is easily taken for a v; a covariance is a plain product, a correlation carries a ratio or a square root.
- Keep the author's subscripts (x_i, x_j stay x_i, x_j) and their parameterisation.
- An indicator 1 is \mathbbm{1}. Escape a literal & as \&.
"""

TABLE_USER = (
    "This is a crop of one table from a page of handwritten mathematics notes. "
    "Transcribe the whole table as a single LaTeX tabular, every row and every column."
)

COLOR_SYSTEM = r"""You transcribe a crop of handwritten notes that contains COLORED ink.
Output ONLY the LaTeX for the crop's text: no preamble, no commentary, no code fences.
- Wrap each run of text written in a colored pen in \textcolor{<color>}{...} with a standard color name (red, green, blue, cyan, magenta, yellow, brown, lime, olive, orange, pink, purple, teal, violet).
- Content written in black or gray ink gets NO \textcolor, even inside this crop.
- Copy the text exactly as written; keep Chinese as Chinese.
- An underline stays \underline{...}, a strike-through is \cancel{...}, each colored as the ink that drew it.
- You have no tools and no functions: never emit function calls, XML or JSON. Answer in LaTeX only.
"""

COLOR_USER = (
    "This crop shows the part of a page that carries {names} ink. "
    "Transcribe the crop, marking exactly the colored content."
)

LOCATE_SYSTEM = (
    "You are a precise figure locator for scanned handwritten notes. "
    "Reply with the JSON list only, no commentary."
)

LOCATE_USER = (
    "This is a page of handwritten mathematics lecture notes. Find every FIGURE on the page: "
    "hand-drawn pictures, sketches, graphs and commutative/arrow diagrams. Ordinary lines of "
    "text, single equations and matrices are NOT figures. Return ONLY a JSON list, one object "
    'per figure, in reading order (top to bottom): [{"bbox": [x0, y0, x1, y1]}]. Coordinates '
    "are integers from 0 to 1000, relative to the image width and height, origin at the "
    "TOP-LEFT, x to the right, y downward. Each box must enclose the whole figure including "
    "all of its labels. If there are no figures return []."
)

FIX_SYSTEM = (
    "You are a LaTeX repair expert. You will receive a page image, the LaTeX "
    "that was generated for it, and compiler errors. Return a corrected full "
    "body for that page only. Output ONLY LaTeX, no commentary. Keep every "
    "\\includegraphics line and every \\figbox{..}{..}{..}{..} line exactly as it is "
    "(they are pictures cropped from the page), and keep underlines, centering and "
    "size commands. Fix only what the compiler complained about: do not restate the "
    "mathematics in a more familiar form, do not change the author's symbols or "
    "parameters (Exp(theta) with mean theta must not become Exp(lambda) with mean "
    "1/lambda), and do not correct anything you think is wrong."
)


def doc_hint_block(filenames: List[str]) -> str:
    """System-prompt addendum naming the uploaded files ('' when there are none).

    A handwritten name or title is small, stylised and genuinely ambiguous: on
    a real page the author 周民强 came out as 周兆庆 on one pass and 周美玲 on
    another. The file the user uploaded was named 实变函数_周民强_总结.pdf -- the
    correct spelling was available all along. Names, titles and subject terms
    that are hard to read should prefer the file name's reading.
    """
    if not filenames:
        return ""
    shown = ", ".join(f'"{n}"' for n in filenames[:5])
    return (
        "\n\nSOURCE FILE\n"
        f"- These notes were uploaded as {shown}. A name, title or subject written "
        "on the page (an author, a book, a course, a topic) is often the same words "
        "as in this file name. When such a word is hard to read in the handwriting, "
        "prefer the reading that matches the file name."
    )


def transcribe_user_prompt(context_tail: str, open_environments: List[str],
                           color_ink: Optional[List[str]] = None) -> str:
    parts = ["Transcribe this page to LaTeX body content."]
    if color_ink:
        names = " and ".join(color_ink)
        parts.append(
            f"COLORED INK IS PRESENT ON THIS PAGE: image analysis of the scan found "
            f"{names} ink. It is on the page right now - look for it: colored text, "
            "underlines, strikes, boxes and margin marks in those colors. Every one of "
            "them must reach your output as \\textcolor{<color>}{...} (a strike-through "
            "is \\textcolor{<color>}{\\cancel{...}}), as the system prompt describes. "
            "A page with colored ink whose transcription contains no \\textcolor has "
            "silently dropped the colors.")
    if context_tail.strip():
        parts.append(
            "The document so far ends with:\n```latex\n" + context_tail.strip() + "\n```"
            "\nKeep notation, numbering and style consistent with it."
        )
    if open_environments:
        envs = ", ".join(open_environments)
        parts.append(
            f"Note: the previous page left these LaTeX environments open: {envs}. "
            "Continue inside them (do not re-open them)."
        )
    parts.append("Return only the LaTeX for THIS page.")
    return "\n\n".join(parts)


PACKAGES = ("amsmath, amssymb, amsthm, mathtools, mathrsfs, bm, bbm, cancel, xcolor, "
            "graphicx, tikz, tikz-cd, pgfplots")

# Plain-language hints for the compile errors seen most in real runs. Handwritten
# notes use "&" for "and" all the time (16 of 25 first-attempt failures on a
# 43-page job), and a repair prompt that only quotes the error kept getting the
# same broken environment back.
_HINTS = [
    (re.compile(r"Misplaced alignment tab|Extra alignment tab"),
     "A bare `&` is only legal inside array/tabular/align/aligned/matrix/cases. In running "
     "text write it as `\\&` (handwritten \"&\" usually means \"and\"). Inside a display, "
     "put the aligned rows in a proper alignment environment. In an `array`/`tabular` the "
     "column spec must declare at least as many columns as the widest row has cells "
     "(a row with 9 cells needs `ccccccccc`, not `ccccccc`)."),
    (re.compile(r"allowed only in math mode|Missing \$ inserted"),
     "Math-only commands/environments (aligned, split, gathered, cases, matrices, \\frac, "
     "\\mathfrak, ^ and _) must be inside $...$ or \\[ ... \\]. Wrap an `aligned` block in "
     "\\[ ... \\]. The opposite mistake is just as fatal: `\\begin{center}`, "
     "`\\includegraphics` and `\\figbox` are text-mode and must NOT sit inside \\[ ... \\], "
     "$$ ... $$ or $ ... $ -- take the picture out of the math delimiters."),
    (re.compile(r"Unicode character"),
     "pdflatex cannot typeset raw Unicode symbols: write \\forall, \\exists, \\in, \\nmid, "
     "\\mathbb{R}, \\alpha ... as LaTeX macros inside math."),
    (re.compile(r"Paragraph ended before \\@?textcolor"),
     "\\textcolor{c}{...} may not contain a blank line. Color a multi-paragraph "
     "block with the group form {\\color{c} ... }, or give each paragraph its own "
     "\\textcolor."),
    (re.compile(r"macro parameter character #"),
     "A literal # must be written `\\#`."),
    (re.compile(r"Environment [\w*-]+ undefined"),
     "That environment is not available. Only these packages are loaded: " + PACKAGES +
     ". Rewrite the figure with plain tikz, an `array`, or a `tabular`."),
    (re.compile(r"No shape named|tikz@f@|Package tikz Error|Giving up on this path"),
     "A tikz / tikz-cd figure refers to a node or cell that is empty or was never defined "
     "(every node used in \\draw / \\arrow must be declared first; tikz-cd directions such as "
     "`dr`, `ur`, `l` must land on an existing, non-empty cell). Fix the references, or "
     "redraw the diagram as a simple `array`."),
    (re.compile(r"Can be used only in preamble"),
     "Never output \\documentclass, \\usepackage or \\begin{document}: return body content only."),
    (re.compile(r"Undefined control sequence"),
     "Use only standard amsmath/amssymb macros; replace an unknown macro with an equivalent "
     "standard one or plain text."),
]


def error_hints(errors: List[str]) -> List[str]:
    """Distinct hints that apply to the given compiler errors, in stable order."""
    text = "\n".join(errors)
    return [hint for pattern, hint in _HINTS if pattern.search(text)]


def fix_user_prompt(previous_latex: str, errors: List[str], repeats: int = 0) -> str:
    """Repair prompt. ``repeats`` = how many earlier repair attempts already failed
    with this very same error; from 2 on, plain "fix it" has demonstrably stopped
    working (a real job got the identical broken diagram back four times), so the
    prompt says so and asks for the smallest change that compiles."""
    err_text = "\n".join(f"- {e}" for e in errors) or "- unknown error"
    prev = previous_latex.strip()
    if len(prev) > 3000:
        prev = prev[-3000:]
    hints = error_hints(errors)
    hint_text = ("Hints:\n" + "\n".join(f"- {h}" for h in hints) + "\n\n") if hints else ""
    escalate = ""
    if repeats >= 1:
        escalate = (
            f"IMPORTANT: your previous {repeats} fix attempt(s) produced this SAME error, so "
            "repeating the same approach will fail again. Change the approach. If one figure or "
            "diagram cannot be made to compile, replace ONLY that figure with the text "
            "\\textit{[diagram omitted]} and keep everything else on the page exactly as "
            "transcribed.\n\n")
    return (
        "The following LaTeX was generated for this page but failed to compile.\n\n"
        f"Generated LaTeX:\n```latex\n{prev}\n```\n\n"
        f"Compiler errors:\n{err_text}\n\n"
        f"{hint_text}"
        f"{escalate}"
        "Fix the errors and return the corrected full page body."
    )
