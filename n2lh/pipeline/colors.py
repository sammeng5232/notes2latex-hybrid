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

The merge is deliberately conservative, in two ways:

* a run only ever wraps text that is ALREADY in the page LaTeX (found fuzzily,
  because two readings of the same handwriting differ: "Closed nested sets
  thm" vs "...theorem") -- the crop read contributes the COLOR, never the
  CONTENT;
* a run whose text occurs MORE THAN ONCE (a glossary term is exactly a word
  that also appears in the body) is only applied when other, unambiguous runs
  anchor where the colored block sits: it is then colored at the occurrence
  nearest that anchor. On the real page, "Cantor 闭集套定理" was pinked inside
  a body theorem because it also names a glossary cell; with nothing to anchor
  it, an ambiguous run is skipped -- a missing color beats a wrong one.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

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


def _widen(page_latex: str, start: int, end: int) -> Tuple[int, int]:
    """Widen a span to whole word/command boundaries.

    Never wrap half a word or half a command, and include a simple wrapping
    command (\\underline{...}, \\textbf{...}) so a colored underline comes out
    as one nested command (color outside, underline inside) rather than a
    nesting that hides it.
    """
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
        while cs > 0 and page_latex[cs].isalnum():
            cs -= 1
        if cs >= 0 and page_latex[cs] == "\\":
            start = cs
            if end < len(page_latex) and page_latex[end] == "}":
                end += 1
    return start, end


def _occurrences(page_latex: str, plain: str) -> List[Tuple[int, int]]:
    """All whitespace-flexible exact occurrences of the run's text."""
    words = plain.split()
    if not words:
        return []
    pattern = re.compile(r"\s+".join(re.escape(word) for word in words), re.S)
    return [_widen(page_latex, *m.span()) for m in pattern.finditer(page_latex)]


def _locate(page_latex: str, plain: str) -> Optional[Tuple[int, int]]:
    """The best fuzzy (SequenceMatcher) span for the run's text, or None.

    Used when the text does not occur verbatim: two readings of the same
    handwriting differ. Covering >= _MIN_MATCH of the run means "this is that
    text, misread a little".
    """
    matcher = SequenceMatcher(None, plain, page_latex, autojunk=False)
    block = max(matcher.get_matching_blocks(),
                key=lambda b: b.size, default=None)
    if block is None or block.size == 0:
        return None
    if block.size < _MIN_MATCH * len(plain):
        return None
    return _widen(page_latex, block.b, block.b + block.size)


def _overlaps(span: Tuple[int, int], placed: List[Tuple[int, int]]) -> bool:
    return any(span[0] < p1 and span[1] > p0 for p0, p1 in placed)


def _already_wrapped(page_latex: str, span: Tuple[int, int], color: str) -> bool:
    """Is this span already inside a \\textcolor of the same color?

    The transcription got the color right, the crop read corroborated it, and
    reconcile_colors kept it -- wrapping it again would nest the command.
    """
    return (page_latex[:span[0]].endswith("\\textcolor{" + color + "}{")
            and page_latex[span[1]:span[1] + 1] == "}")


def _similar(a: str, b: str) -> bool:
    """Do two readings refer to the same text? (Same fuzziness as _locate.)

    The common text must cover most of BOTH readings: corroboration by
    substring is how a real margin mark ("测度", green) vouched for a model's
    invented green on body text that merely contained it ("的外测度").
    """
    if not a or not b:
        return False
    longest = max(len(a), len(b))
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    block = max(matcher.get_matching_blocks(),
                key=lambda b: b.size, default=None)
    return block is not None and block.size >= 0.7 * longest


def reconcile_colors(page_latex: str,
                     reads: Dict[str, List[str]]) -> Tuple[str, int]:
    """Unwrap the transcription's own \\textcolor runs the crop reads refute.

    The whole-page transcription does not only DROP colors -- sometimes it
    invents them. Real case: the green pen on the page was single margin
    characters, and the model marked seven body statements green; every one
    compiled, looked deliberate, and was wrong. So when a focused crop read
    of a color's region SUCCEEDED, that read is the authority for the color:
    a model run of that color whose text matches none of the read's runs is
    unwrapped (the words stay, the color goes). Colors whose crop read failed
    are left exactly as the model wrote them -- a flaky endpoint must not
    cost colors that were right.

    ``reads`` maps color -> the plain texts the crop reads marked in that
    color; a color is only in the map when its region read out at all.
    """
    unwrapped = []

    def maybe_unwrap(m: "re.Match[str]") -> str:
        color, body = m.group(1), m.group(2)
        if color in reads and not any(_similar(_plain(body), t)
                                      for t in reads.get(color, [])):
            unwrapped.append(color)
            return body
        return m.group(0)

    out = _RUN.sub(maybe_unwrap, page_latex)
    return out, len(unwrapped)


def apply_color_runs(page_latex: str,
                     runs: List[Tuple[str, str]]) -> Tuple[str, int]:
    """Wrap the page's own text in \\textcolor where the runs' text is found.

    Returns the new LaTeX and how many runs were applied. Matching happens in
    the page's ORIGINAL coordinates and the wraps are applied from the end, so
    spans stay valid. Runs that cannot be placed are skipped (and logged):
    better a missing color than a color on the wrong words.
    """
    planned: List[Tuple[int, int, str]] = []   # (start, end, color)
    anchors: List[Tuple[int, int]] = []        # spans of unambiguous runs
    deferred: List[Tuple[str, str, List[Tuple[int, int]]]] = []

    for color, plain in runs:
        occ = _occurrences(page_latex, plain)
        if len(occ) == 1:
            span = occ[0]
        elif len(occ) > 1:
            deferred.append((color, plain, occ))
            continue
        else:
            span = _locate(page_latex, plain)
            if span is None:
                log.info("color run not found in the page latex: %r", plain[:60])
                continue
        if (_already_wrapped(page_latex, span, color)
                or _overlaps(span, [(p0, p1) for p0, p1, _ in planned])):
            continue
        planned.append((*span, color))
        anchors.append(span)

    if deferred:
        if anchors:
            # The unambiguous runs say where the colored block sits; a term
            # that also appears elsewhere (glossary words recur in the body)
            # is colored at the occurrence nearest that.
            centers = sorted((s + e) / 2 for s, e in anchors)
            center = centers[len(centers) // 2]
            for color, plain, occ in deferred:
                # Occurrences the transcription already colored (and the read
                # corroborated) are done; pick among the rest.
                candidates = [sp for sp in occ
                              if not _already_wrapped(page_latex, sp, color)]
                if not candidates:
                    continue
                span = min(candidates,
                           key=lambda sp: abs((sp[0] + sp[1]) / 2 - center))
                if _overlaps(span, [(p0, p1) for p0, p1, _ in planned]):
                    continue
                planned.append((*span, color))
        else:
            for color, plain, occ in deferred:
                log.info("color run is ambiguous with nothing to anchor it: %r",
                         plain[:60])

    for start, end, color in sorted(planned, key=lambda s: s[0], reverse=True):
        page_latex = (page_latex[:start]
                      + "\\textcolor{" + color + "}{" + page_latex[start:end] + "}"
                      + page_latex[end:])
    return page_latex, len(planned)
