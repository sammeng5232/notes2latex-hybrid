"""Re-read a ruled table from a crop of the page.

A dense summary table is the one thing a whole-page transcription reliably gets
wrong, and the cause is resolution rather than comprehension. The client sends
an image of at most 1600px on its longest edge, so on a 2550x3347 page the table
band is squeezed into a few hundred pixels of height and its cells stop being
legible. Asked again with a crop of just the table, the same model reads the
same cell correctly: on the page that prompted this it turned

    Cov(X_i, X_j) = -np_i p_j

(the second formula in the cell lost, or its ``Cor`` misread as another ``Cov``)
into

    Cov(X_i, X_j) = -np_i p_j,  Cor(X_i, X_j) = -sqrt(p_i p_j/((1-p_i)(1-p_j)))

which is what the page says. Three attempts at fixing this by instruction all
failed -- one made it worse -- because no wording buys the model more pixels.

It is the same trick as locating figures in a dedicated request: give the model
one job and enough resolution to do it. The band is found from the table's own
ruled lines, so locating it costs no model call.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

log = logging.getLogger("n2lh.tables")

# A ruled line is a row of the page whose ink spans most of the width.
_RULE_COVERAGE = 0.55
# Fewer rules than this is a box or an underline, not a table worth a second call.
_MIN_RULES = 4
# The crop is enlarged so that the client's downscale to 1600px still leaves the
# table bigger than it was inside the whole page.
_TARGET_WIDTH = 4000
_PAD_FRAC = 0.01

TABULAR = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.S)


def _row_coverage(dark: Image.Image) -> List[float]:
    """Fraction of each row of the page that is ink."""
    w, h = dark.size
    if w < 1 or h < 1:
        return []
    column = dark.resize((1, h), Image.BOX).convert("L").tobytes()
    return [v / 255 for v in column]


def find_table_band(dark: Image.Image) -> Optional[Tuple[int, int]]:
    """Top and bottom of the page's ruled table, or None if it has none."""
    coverage = _row_coverage(dark)
    inked = [y for y, c in enumerate(coverage) if c >= _RULE_COVERAGE]
    if not inked:
        return None

    # A drawn line is several pixel rows thick, so collapse touching rows into
    # one rule first. Counting pixel rows instead would let a single thick
    # underline pass for a four-row table.
    rules: List[Tuple[int, int]] = [(inked[0], inked[0])]
    for y in inked[1:]:
        if y - rules[-1][1] <= 3:
            rules[-1] = (rules[-1][0], y)
        else:
            rules.append((y, y))

    # Rules belonging to one table sit close together; a stray rule elsewhere on
    # the page (an underlined heading) starts its own group and loses.
    gap_limit = max(40, int(dark.size[1] * 0.12))
    groups: List[List[Tuple[int, int]]] = [[rules[0]]]
    for rule in rules[1:]:
        if rule[0] - groups[-1][-1][1] <= gap_limit:
            groups[-1].append(rule)
        else:
            groups.append([rule])

    biggest = max(groups, key=len)
    if len(biggest) < _MIN_RULES:
        return None
    biggest = [biggest[0][0], biggest[-1][1]]
    pad = int(dark.size[1] * _PAD_FRAC)
    return max(0, biggest[0] - pad), min(dark.size[1], biggest[-1] + pad)


def crop_band(page: Image.Image, band: Tuple[int, int], dest: Path) -> Path:
    """Write the table band, enlarged, so it survives the client's downscale."""
    crop = page.crop((0, band[0], page.size[0], band[1]))
    if crop.width < _TARGET_WIDTH:
        scale = _TARGET_WIDTH / crop.width
        crop = crop.resize((_TARGET_WIDTH, max(1, int(crop.height * scale))), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dest, "PNG")
    return dest


def _rows(tabular: str) -> int:
    return tabular.count("\\\\")


def replace_tabular(page_latex: str, table_latex: str) -> Tuple[str, bool]:
    """Swap the page's table for the one read from the crop.

    Only when each side holds exactly one table and the re-read is no shorter:
    a re-read that came back with fewer rows is a worse answer, not a better one.
    """
    if page_latex.count("\\begin{tabular}") != 1:
        return page_latex, False
    old = TABULAR.search(page_latex)
    new = TABULAR.search(table_latex)
    if not old or not new:
        return page_latex, False
    # One row of slack: the crop sometimes omits the header row, which is a
    # formatting difference rather than lost content. Losing more than that is
    # a truncated read and a worse answer than the page's own.
    if _rows(new.group(0)) < _rows(old.group(0)) - 1:
        log.info("table re-read has %d rows against the page's %d; keeping the page's",
                 _rows(new.group(0)), _rows(old.group(0)))
        return page_latex, False
    return page_latex[:old.start()] + new.group(0) + page_latex[old.end():], True


def insert_tabular(page_latex: str, table_latex: str, band: Tuple[int, int],
                   page_height: int) -> Tuple[str, bool]:
    """Put back a table the transcription dropped entirely.

    Real case: a vocabulary table in the top-right corner of a summary page
    came out missing on one pass and present on another -- the whole-page
    transcription is simply not reliable about side content. The crop re-read
    is. Position: a table whose band sits in the top third of the page goes
    right after the transcription's first block (that is where a corner table
    hangs, next to the title); anything lower joins the end of the page, in
    reading order. The table is wrapped in \\fitpage so a wide one cannot run
    off the page, and deliberately NOT centered: it sat in a corner, not in
    the middle of the page.
    """
    new = TABULAR.search(table_latex)
    if not new:
        return page_latex, False
    table = "\\fitpage{" + new.group(0) + "}"
    blocks = [b for b in re.split(r"\n\s*\n", page_latex.strip()) if b.strip()]
    if not blocks:
        return table, True
    center = (band[0] + band[1]) / 2 / max(page_height, 1)
    if center < 1 / 3:
        return "\n\n".join([blocks[0], table] + blocks[1:]), True
    return "\n\n".join(blocks + [table]), True


def wrap_corner_table(page_latex: str, run_texts: List[str]) -> Tuple[str, bool]:
    """Move a corner table into a right-floating wraptable beside the body.

    Color evidence says this table sat in a top corner of the page: its cells
    carry the colored ink of a corner region. Whole-page transcriptions place
    it anywhere -- stacked under the title on one pass, at the end of the page
    on another -- because a corner table has no natural slot in linear LaTeX.
    A wraptable right after the page's first block puts it back in the corner,
    with the body text flowing beside it the way the page reads. A centered
    title line directly above the table is taken along, un-centered.
    """
    if "wraptable" in page_latex:
        return page_latex, False
    texts = [t for t in run_texts if t]
    if not texts:
        return page_latex, False
    match = next((m for m in TABULAR.finditer(page_latex)
                  if any(t in m.group(0) for t in texts)), None)
    if match is None:
        return page_latex, False
    start, end = match.start(), match.end()
    # Absorb a \fitpage{...} wrapper around the tabular.
    if (page_latex[:start].rstrip().endswith("\\fitpage{")
            and page_latex[end:end + 1] == "}"):
        start = page_latex.rfind("\\fitpage{", 0, start)
        end += 1
    # Absorb a centered title line directly above the table: it travels along,
    # un-centered, but is not part of the table itself.
    cut = start
    header = ""
    above = re.search(r"\\begin\{center\}\s*(.+?)\s*\\end\{center\}\s*$",
                      page_latex[:cut], re.S)
    if above:
        header = above.group(1)
        cut = above.start()
    table = page_latex[start:end].strip()
    rest = (page_latex[:cut] + page_latex[end:]).strip()
    blocks = re.split(r"\n\s*\n", rest, maxsplit=1)
    first = blocks[0] if blocks and blocks[0].strip() else ""
    remainder = blocks[1] if len(blocks) > 1 else ""
    wrapped = ("\\begin{wraptable}{r}{0.6\\textwidth}\n"
               + (header + "\n" if header else "")
               + table + "\n\\end{wraptable}")
    out = ((first + "\n\n" if first else "") + wrapped
           + ("\n\n" + remainder if remainder.strip() else ""))
    return out, True
