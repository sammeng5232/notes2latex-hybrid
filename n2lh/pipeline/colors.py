"""Put the colored ink back into a transcription that dropped it.

The whole-page transcription does many jobs at once -- read the words, the
math, the layout, the tables -- and on the small models this pipeline runs
against, ink color is the first thing to fall off: three prompt shapes were
tried (a generic line in the system prompt, a per-page directive naming the
detected colors, a directive adding WHERE the colors are), and none produced a
single \\textcolor, while the very same model asked ONLY about colors, on a
crop of just the colored region, marked seven runs correctly.

So this is the same trick as the table re-read and the figure locator: one
job, enough pixels. ``n2lh.pipeline.ingest.colored_ink_regions`` finds WHERE
the colored ink is (density clustering, no model call); the recognizer is
asked to transcribe each region's crop with the colors marked; and the runs it
returns are merged into the page's own LaTeX here.

The merge is deliberately conservative: a run only ever wraps text that is
ALREADY in the page LaTeX (found fuzzily, because two readings of the same
handwriting differ: "Closed nested sets thm" vs "...theorem"). The crop read
contributes the COLOR, never the CONTENT -- a wrong span match is skipped, not
guessed.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

log = logging.getLogger("n2lh.colors")

# \textcolor{pink}{...} with one level of nested braces (for \cancel etc.).
_RUN = re.compile(r"\\textcolor\{(\w+)\}\{((?:[^{}]|\{[^{}]*\})*)\}")
# A latex command (for reducing a run to the words it covers).
_COMMAND = re.compile(r"\\[A-Za-z]+")
# A run shorter than this is not worth fuzzy-matching: a two-character match
# can be found anywhere. A Chinese character counts double -- 振幅 is a
# complete word at two characters, "ab" is not.
_MIN_RUN_WEIGHT = 3
# How much of the run's plain text must be found in the page LaTeX.
_MIN_MATCH = 0.7


def _weight(text: str) -> int:
    return sum(2 if "\u3400" <= c <= "\u9fff" else 1 for c in text)


def extract_color_runs(latex: Optional[str]) -> List[Tuple[str, str]]:
    """(color, plain text) for every \\textcolor run in the crop's LaTeX."""
    if not latex:
        return []
    runs = []
    for color, body in _RUN.findall(latex):
        plain = _plain(body)
        if _weight(plain) >= _MIN_RUN_WEIGHT:
            runs.append((color, plain))
    return runs


def _plain(text: str) -> str:
    """The words a run covers, without the latex commands inside it."""
    text = _COMMAND.sub(" ", text)
    return re.sub(r"[{}\s]+", " ", text).strip()


def _locate(page_latex: str, plain: str,
            placed: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Where (start, end) in page_latex the run's text best matches.

    SequenceMatcher against the whole page gives the longest common substring;
    covering >= _MIN_MATCH of the run means "this is that text, misread a
    little". The span is then widened to whole word/command boundaries so the
    wrap cannot cut a token in half, and to include a simple wrapping command
    (underline, bold) so a colored underline comes out as one nested command
    (color outside, underline inside) rather than a nesting that hides it.
    """
    matcher = SequenceMatcher(None, plain, page_latex, autojunk=False)
    block = max(matcher.get_matching_blocks(),
                key=lambda b: b.size, default=None)
    if block is None or block.size == 0:
        return None
    if block.size < _MIN_MATCH * len(plain):
        return None
    start, end = block.b, block.b + block.size
    # Widen to token boundaries: never wrap half a word or half a command.
    while start > 0 and page_latex[start - 1].isalnum():
        start -= 1
    while end < len(page_latex) and page_latex[end].isalnum():
        end += 1
    # A command the match landed inside gets wrapped whole.
    if start > 0 and page_latex[start - 1] == "\\":
        cs = start - 1
        while cs > 0 and page_latex[cs - 1] == "\\":
            cs -= 1
        start = cs
    # ... and so does a simple \command{match} around it.
    if start > 0 and page_latex[start - 1] == "{":
        cs = start - 2
        while cs > 0 and (page_latex[cs].isalnum()):
            cs -= 1
        if cs >= 0 and page_latex[cs] == "\\":
            start = cs
            if end < len(page_latex) and page_latex[end] == "}":
                end += 1
    for p0, p1 in placed:
        if start < p1 and end > p0:      # overlaps a run already colored
            return None
    return start, end


def apply_color_runs(page_latex: str,
                     runs: List[Tuple[str, str]]) -> Tuple[str, int]:
    """Wrap the page's own text in \\textcolor where the runs' text is found.

    Returns the new LaTeX and how many runs were applied. Unmatched runs are
    skipped (and logged): better a missing color than a color on the wrong
    words.
    """
    placed: List[Tuple[int, int]] = []
    applied = 0
    # Apply from the end so earlier spans keep their offsets.
    found: List[Tuple[int, int, str]] = []
    for color, plain in runs:
        span = _locate(page_latex, plain, placed)
        if span is None:
            log.info("color run not found in the page latex: %r", plain[:60])
            continue
        placed.append(span)
        found.append((span[0], span[1], color))
        applied += 1
    for start, end, color in sorted(found, reverse=True):
        page_latex = (page_latex[:start]
                      + "\\textcolor{" + color + "}{" + page_latex[start:end] + "}"
                      + page_latex[end:])
    return page_latex, applied
