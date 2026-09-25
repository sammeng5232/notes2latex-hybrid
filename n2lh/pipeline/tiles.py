"""Transcribe a dense page as full-resolution horizontal strips.

The client sends every image at most 1600px on its longest edge. On a sparse
page of notes that is plenty; on a dense summary sheet it is not. A 3508x4961
A3 page of real-analysis notes carries about a hundred lines of small Chinese
handwriting, and at 1131x1600 each line is a dozen pixels tall. The model
cannot read that, so it writes what such a page usually says instead: every
whole-page read called Bolzano-Weierstrass "Borel-Cantelli", and one invented
entire chapters -- product measures, Radon-Nikodym, the Fourier transform --
none of which is on the page. Read as six full-resolution strips, the same
page came back with every theorem it has, in order, and nothing it does not.

So a dense page is cut into horizontal strips, each sent without downscaling,
and the strip transcriptions are joined top to bottom. Cuts go through the
emptiest rows near evenly spaced targets, never through a ruled table, a
colored glossary or a margin label, so no line or block is split. Density is
judged from the line pitch measured in narrow vertical stripes of the page:
across whole rows, the lines of close handwriting touch and merge into
paragraphs, which is why the page-wide line height reads 117px on this page
when its true pitch is ~50px.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# The longest edge the client sends a whole page at.
PAGE_EDGE = 1600
# Below this many pixels per line at PAGE_EDGE, handwriting stops being legible
# enough to trust. Measured: the dense pages that fabricate sit at 28-30px, the
# sparse topology pages that read well at 65-75px.
MIN_LEGIBLE_PITCH = 45.0
# Each strip holds about this many lines as count_lines() counts them (a slight
# undercount: ~8 real lines). The model skips runs of lines inside long strips
# -- five theorems at a time from 16-line strips of the real page -- so strips
# are kept short; a short strip is also a fast request.
LINES_PER_STRIP = 6
MIN_STRIPS, MAX_STRIPS = 2, 16
# The longest edge a strip is sent at: the full width of a 300dpi A3 page.
STRIP_EDGE = 3600
# Ink darkness threshold shared with the table and figure finders.
_INK = 145


@dataclass(frozen=True)
class Strip:
    index: int          # 1-based, top to bottom
    top: int            # page pixel rows [top, bottom)
    bottom: int

    @property
    def height(self) -> int:
        return self.bottom - self.top


# ------------------------------------------------------------------- density
def _runs(profile: Sequence[int], gap: int = 2, min_h: int = 4) -> List[Tuple[int, int]]:
    """Runs of ink rows, split wherever `gap` blank rows in a row appear."""
    out: List[Tuple[int, int]] = []
    start: Optional[int] = None
    clear = 0
    for i, v in enumerate(profile):
        if v:
            if start is None:
                start = i
            clear = 0
        elif start is not None:
            clear += 1
            if clear >= gap:
                end = i - clear + 1
                if end - start >= min_h:
                    out.append((start, end))
                start, clear = None, 0
    if start is not None and len(profile) - start >= min_h:
        out.append((start, len(profile)))
    return out


def line_pitch(dark: Image.Image) -> Optional[float]:
    """Median distance between text lines, from narrow vertical stripes.

    Four stripes across the body of the page, each an eighth of its width. In a
    stripe that narrow, neighbouring lines rarely touch, so its runs of ink are
    lines rather than paragraphs.
    """
    W, H = dark.size
    pitches = []
    for left in (0.10, 0.25, 0.40, 0.55):
        stripe = dark.crop((int(W * left), 0, int(W * (left + 0.12)), H))
        profile = list(stripe.resize((1, H), Image.BOX).tobytes())
        runs = _runs(profile)
        if len(runs) >= 4:
            centres = [(a + b) / 2 for a, b in runs]
            pitches.append(statistics.median(b - a for a, b in zip(centres, centres[1:])))
    return statistics.median(pitches) if pitches else None


def needs_strips(dark: Image.Image, pitch: Optional[float] = None) -> bool:
    """True when the page is too dense to read at the whole-page resolution."""
    pitch = line_pitch(dark) if pitch is None else pitch
    if not pitch:
        return False
    return pitch * PAGE_EDGE / max(dark.size) < MIN_LEGIBLE_PITCH


# ------------------------------------------------------------------- planning
def planning_mask(page: Image.Image) -> Image.Image:
    """Ink for choosing cuts: dark strokes AND colored ones.

    Pink ink like the glossary's has a luminance of ~148, above the dark
    threshold, so a darkness mask alone saw its rows as blank paper and could
    cut straight through a colored line.
    """
    from n2lh.pipeline.ingest import _colored_mask

    rgb = page.convert("RGB")
    arr = np.asarray(rgb).astype(int)
    ink = (np.asarray(rgb.convert("L")) < _INK) | _colored_mask(arr)
    return Image.fromarray((ink * 255).astype(np.uint8), mode="L")


def protected_bands(page: Image.Image, dark: Image.Image,
                    pitch: Optional[float]) -> List[Tuple[int, int]]:
    """Row ranges a cut must not pass through.

    Every ruled table, and every colored block -- a corner glossary, a vertical
    section label in the margin. A cut through a table leaves two half tables;
    through a margin label, the circled number in one strip and its words in the
    next (it split "6 Lp" on the real page before this).
    """
    from n2lh.pipeline.ingest import colored_ink_regions
    from n2lh.pipeline.tables import find_table_bands

    H = dark.size[1]
    pad = int((pitch or H * 0.01) * 0.5)
    bands = [(a, b) for a, b, _ in find_table_bands(dark, min_rules=3)]
    try:
        regions = colored_ink_regions(page, max_regions=12)
    except Exception:  # noqa: BLE001 - protection is best-effort
        regions = []
    for box, _names in regions:
        bands.append((max(0, box[1] - pad), min(H, box[3] + pad)))
    # A band that tall would leave nowhere to cut, and a page read whole is the
    # failure strips exist to avoid; a strip boundary through it costs less.
    return [(a, b) for a, b in bands if b - a <= 0.4 * H]


def plan_strips(dark: Image.Image, pitch: Optional[float] = None,
                keep_whole: Sequence[Tuple[int, int]] = ()) -> List[Strip]:
    """Cut the page into strips at the emptiest rows near even targets.

    ``keep_whole`` lists page row ranges that must not be cut through -- a ruled
    table, a colored glossary, a margin label -- so each lands in one strip.
    """
    W, H = dark.size
    pitch = line_pitch(dark) if pitch is None else pitch
    if not pitch:
        return [Strip(1, 0, H)]
    lines = count_lines(dark, pitch) or round(H / pitch)
    n = max(MIN_STRIPS, min(MAX_STRIPS, round(lines / LINES_PER_STRIP)))
    ink = list(dark.resize((1, H), Image.BOX).tobytes())
    blocked = [False] * H
    for a, b in keep_whole:
        for y in range(max(0, a), min(H, b)):
            blocked[y] = True

    step = H / n
    window = int(step * 0.4)
    cuts: List[int] = []
    for k in range(1, n):
        target = int(k * step)
        best = None
        for y in range(max(1, target - window), min(H - 1, target + window)):
            if blocked[y]:
                continue
            # Emptiest row first, then the one closest to the target.
            key = (ink[y], abs(y - target))
            if best is None or key < best[0]:
                best = (key, y)
        if best is not None and (not cuts or best[1] - cuts[-1] > pitch * 2):
            cuts.append(best[1])

    edges = [0] + cuts + [H]
    return [Strip(i + 1, top, bottom)
            for i, (top, bottom) in enumerate(zip(edges, edges[1:]))
            if bottom - top > 0]


# ------------------------------------------------------------------ figures
def remap_figbox(latex: str, strip: Strip, page_height: int) -> str:
    """Turn a strip's \\figbox coordinates (0..1000 of the strip) into page ones.

    x is unchanged -- strips are full width -- and y is rescaled from the strip's
    height into its slice of the page.
    """
    from n2lh.pipeline.figures import FIGBOX

    def to_page(y: int) -> int:
        return round((strip.top + y / 1000 * strip.height) / page_height * 1000)

    def sub(m) -> str:
        x0, y0, x1, y1 = (int(g) for g in m.groups())
        return f"\\figbox{{{x0}}}{{{to_page(y0)}}}{{{x1}}}{{{to_page(y1)}}}"

    return FIGBOX.sub(sub, latex)


def _merge_seam_figures(parts: List[str], strips: Sequence[Strip], page_height: int) -> List[str]:
    """Join a figure that a cut split into one marker per strip.

    Cuts go through blank rows, and a drawing has blank rows inside it. Then
    strip k ends with a \\figbox touching its bottom edge and strip k+1 starts
    with one touching its top edge, over the same columns: that is one figure.
    The first marker is stretched over both and the second is removed.
    """
    from n2lh.pipeline.figures import FIGBOX

    near = 20                                  # 2% of the page, in 0..1000 units
    for k in range(len(parts) - 1):
        cut = round(strips[k].bottom / page_height * 1000)
        last = list(FIGBOX.finditer(parts[k]))
        first = list(FIGBOX.finditer(parts[k + 1]))
        if not last or not first:
            continue
        a, b = last[-1], first[0]
        ax0, ay0, ax1, ay1 = (int(g) for g in a.groups())
        bx0, by0, bx1, by1 = (int(g) for g in b.groups())
        overlap = min(ax1, bx1) - max(ax0, bx0)
        narrower = max(1, min(ax1 - ax0, bx1 - bx0))
        if abs(ay1 - cut) <= near and abs(by0 - cut) <= near and overlap >= 0.5 * narrower:
            merged = (f"\\figbox{{{min(ax0, bx0)}}}{{{ay0}}}"
                      f"{{{max(ax1, bx1)}}}{{{by1}}}")
            parts[k] = parts[k][:a.start()] + merged + parts[k][a.end():]
            parts[k + 1] = parts[k + 1][:b.start()] + parts[k + 1][b.end():]
    return parts


# ------------------------------------------------------------- line structure
_ENV = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")


def lines_as_paragraphs(latex: str) -> str:
    """One handwritten line per paragraph, outside environments and display math.

    A strip is transcribed one handwritten line per output line, and LaTeX joins
    lines separated by a single newline into one run-on paragraph: a section of
    six theorems came out as one block of text. Inside an environment (a table,
    a display, a list) and after an explicit ``\\\\`` a newline means something
    else and is left alone.
    """
    from n2lh.pipeline.style import display_math_depth

    out: List[str] = []
    depth = 0              # environment nesting
    display = 0            # \[ ... \] or $$ ... $$ open
    lines = latex.split("\n")
    for i, line in enumerate(lines):
        out.append(line)
        for m in _ENV.finditer(line):
            depth = max(0, depth + (1 if m.group(1) == "begin" else -1))
        display = display_math_depth(line, display)
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if (nxt is not None and line.strip() and nxt.strip()
                and depth == 0 and display == 0
                and not line.rstrip().endswith("\\\\")
                and not nxt.lstrip().startswith(("\\end", "\\item", "&", "\\\\"))):
            out.append("")
    return "\n".join(out)


# ------------------------------------------------------------- corner tables
_WRAPTABLE = re.compile(r"\\begin\{wraptable\}(\{[^}]*\}\{[^}]*\})(.*?)\\end\{wraptable\}", re.S)
_SEPARATOR = re.compile(r"\s*(?:\\quad|\\qquad|&|\\hfill)\s*")


def tabularize_wraptables(latex: str) -> str:
    """Give a corner table that was written as spaced lines a real tabular.

    A strip read of the page's glossary came back as a wraptable of lines like
    `振幅 \\quad Oscillation`, with no tabular: LaTeX sets those lines as one run
    of text, and the page-level corner handling only recognises a tabular.
    Lines with a separator become rows; a line without one before them is the
    table's heading and stays above it.
    """
    def fix(m: "re.Match[str]") -> str:
        size, body = m.group(1), m.group(2)
        if "\\begin{tabular}" in body:
            return m.group(0)
        head, rows = [], []
        for raw in body.strip().split("\n"):
            line = raw.strip().rstrip("\\").strip()
            if not line:
                continue
            cells = [c.strip() for c in _SEPARATOR.split(line) if c.strip()]
            if len(cells) >= 2:
                rows.append(cells)
            elif not rows:
                head.append(line)
            else:
                rows.append([line])
        if len(rows) < 2:
            return m.group(0)
        width = max(len(r) for r in rows)
        table = ("\\begin{tabular}{" + "l" * width + "}\n"
                 + "\n".join(" & ".join(r + [""] * (width - len(r))) + " \\\\" for r in rows)
                 + "\n\\end{tabular}")
        heading = ("\n".join(head) + "\n\n") if head else ""
        return f"\\begin{{wraptable}}{size}\n{heading}{table}\n\\end{{wraptable}}"

    return _WRAPTABLE.sub(fix, latex)


def wraptable_after_title(latex: str) -> str:
    """Move a corner wraptable up to just after the page's first block.

    A wraptable floats beside the paragraphs that FOLLOW it, so written after the
    section it sits next to (where a strip read put it) it would hang beside the
    wrong text. The page's first block is its title; the corner table belongs
    right after it, as wrap_corner_table places one too.
    """
    m = _WRAPTABLE.search(latex)
    if not m:
        return latex
    table = m.group(0)
    rest = (latex[:m.start()] + latex[m.end():]).strip()
    blocks = [b for b in re.split(r"\n\s*\n", rest) if b.strip()]
    if not blocks:
        return table
    return "\n\n".join([blocks[0], table] + blocks[1:])


# ------------------------------------------------------------------ re-reads
# How a strip is shown again after a read that failed the check. The endpoint
# answers an identical request identically -- the same five theorems skipped at
# temperature 0.1, 0.35 and 0.6 -- so a re-read has to change the image. White
# padding shifts the grid the vision encoder cuts the image into; a slight
# rescale changes it too. On the real page, strips that came back as garbage on
# every plain request read cleanly padded, and a strip whose full-size read
# kept only its last two lines came back whole at 60% size. Every variant is
# still at least 1.5x the resolution of a whole-page read.
STRIP_VARIANTS = ((0, 1.0), (0, 0.6), (40, 0.8), (96, 1.0), (0, 0.5), (64, 0.7))


def strip_variant(image: Image.Image, variant: int) -> Image.Image:
    pad, scale = STRIP_VARIANTS[variant % len(STRIP_VARIANTS)]
    if scale != 1.0:
        image = image.resize((max(1, int(image.width * scale)),
                              max(1, int(image.height * scale))), Image.LANCZOS)
    if pad:
        framed = Image.new(image.mode, (image.width + 2 * pad, image.height + 2 * pad), "white")
        framed.paste(image, (pad, pad))
        image = framed
    return image


def unpad_figbox(latex: str, variant: int, width: int, height: int) -> str:
    """Map \\figbox coordinates read from a padded variant back onto the strip."""
    from n2lh.pipeline.figures import FIGBOX

    pad, scale = STRIP_VARIANTS[variant % len(STRIP_VARIANTS)]
    if not pad:
        return latex
    w, h = width * scale, height * scale

    def back(v: int, size: float) -> int:
        return max(0, min(1000, round((v / 1000 * (size + 2 * pad) - pad) / size * 1000)))

    def sub(m) -> str:
        x0, y0, x1, y1 = (int(g) for g in m.groups())
        return (f"\\figbox{{{back(x0, w)}}}{{{back(y0, h)}}}"
                f"{{{back(x1, w)}}}{{{back(y1, h)}}}")

    return FIGBOX.sub(sub, latex)


# ------------------------------------------------------------ checking a read
def count_lines(dark: Image.Image, pitch: Optional[float] = None) -> int:
    """How many lines of handwriting a strip holds, from its ink alone.

    Runs of ink in stripes a thirty-third of the width wide, across the left
    60% of the strip (a corner glossary sits to the right): so narrow that
    neighbouring lines almost never touch. The median over the stripes is a
    slight undercount -- an indented continuation line misses the left stripes
    -- which is the safe side for a completeness check.
    """
    W, H = dark.size
    min_h = max(4, int((pitch or 0) * 0.12)) if pitch else max(4, H // 80)
    counts = []
    for i in range(15):
        left = 0.02 + i * 0.04
        stripe = dark.crop((int(W * left), 0, int(W * (left + 0.03)), H))
        runs = _runs(list(stripe.resize((1, H), Image.BOX).tobytes()), gap=3, min_h=min_h)
        if runs:
            counts.append(len(runs))
    return int(statistics.median(counts)) if counts else 0


_LABEL_ONLY = re.compile(r"^\s*(?:\\textcolor\{[^}]*\}\{)?\\textbf\{[^{}]{0,12}\}\}?\s*$")
_STRUCTURAL = re.compile(r"^\s*(?:\\(?:begin|end)\{[^}]*\}(?:\{[^}]*\})*|\\hline|\\centering"
                         r"|\\\\|\\par|\\medskip|\\smallskip|\\bigskip|\\noindent)\s*$")
_HAN = re.compile(r"[㐀-鿿]")


def _content_lines(latex: str) -> List[str]:
    return [l for l in latex.splitlines()
            if l.strip() and not _LABEL_ONLY.match(l) and not _STRUCTURAL.match(l)]


def judge_strip(latex: str, expected: int) -> Tuple[bool, float, str]:
    """Is a strip read believable? ``(ok, score, reason)``; higher score is better.

    Two failure modes seen on a real endpoint, neither caught by the loop guard:

    - lines silently skipped: of three reads of one 17-line strip, one had all
      17 and the others 10 and 12, each dropping a run of five or so lines;
      three identical requests for another strip all dropped the same five
      theorems. A read with clearly fewer lines than the strip holds is
      incomplete.
    - fluent garbage: "UL se五代 European European s ... Portuguese ULe",
      "SQLSQLiteSQL...", mostly blank lines -- the endpoint answering with no
      regard for the image. It has lines too short and too many, few of them
      with any math or Chinese, and a lot of white space.

    ``expected`` is count_lines() for the strip.
    """
    lines = _content_lines(latex)
    n = len(lines)
    text = "".join(lines)
    if n == 0 or not text.strip():
        return False, 0.0, "empty"
    blank = sum(1 for c in latex if c.isspace()) / max(1, len(latex))
    rich = sum(1 for l in lines if "$" in l or "\\" in l or len(_HAN.findall(l)) >= 2) / n
    mean_len = len(text) / n
    score = float(min(n, expected) if expected else n)
    if blank > 0.45:
        return False, 0.0, f"mostly white space ({blank:.0%})"
    # Strips are only cut from dense pages, whose lines are long (~150
    # characters on the real page); garbage comes in short fragments.
    if (rich < 0.3 and mean_len < 40) or (expected >= 4 and mean_len < 25):
        return False, 0.0, (f"does not look like notes: {n} short lines, "
                                    f"{rich:.0%} with any math or Chinese")
    if expected >= 4 and n > 3 * expected + 6:
        return False, 0.0, f"{n} lines where the strip holds about {expected}"
    if expected >= 5 and n < 0.85 * expected:
        return False, score, f"{n} lines where the strip holds about {expected}: lines skipped"
    return True, score, "ok"


# -------------------------------------------------------------------- joining
def join_strips(parts: Sequence[str], strips: Optional[Sequence[Strip]] = None,
                page_height: Optional[int] = None) -> str:
    """Strip transcriptions, top to bottom, as one page body.

    With the strips and page height, figures split by a cut are rejoined first.
    """
    parts = [p.strip() for p in parts]
    if strips is not None and page_height:
        parts = _merge_seam_figures(list(parts), strips, page_height)
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def tidy_strip(latex: str, strip: Strip, page_height: int) -> str:
    """Everything one strip's transcription needs before it is joined."""
    latex = remap_figbox(latex, strip, page_height)
    latex = tabularize_wraptables(latex)
    if strip.index == 1:
        latex = wraptable_after_title(latex)
    return lines_as_paragraphs(latex)
