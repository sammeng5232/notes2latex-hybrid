"""Turn ``\\figbox{x0}{y0}{x1}{y1}`` placeholders into real pictures.

Redrawing hand-drawn figures as TikZ does not work: on a 43-page set of topology
notes the models produced malformed diagrams (arrows into empty cells, nodes
that were never declared) that failed to compile no matter how many repairs were
asked for, and when they did compile they were a loose imitation. So the model
only *marks* each figure -- a bounding box, in reading order, inside the text --
and the app crops the real drawing out of the page image and embeds it.

Boxes are integers 0..1000 relative to the page image, origin top-left.

Where the coordinates come from matters. A model writing a transcription cannot
also measure: on a real page its four inline boxes were (200,200,400,350),
(200,450,400,600), (200,650,400,800), (200,850,400,1000) -- evenly spaced round
numbers to the left of the actual drawings. Asked *only* to locate figures, the
same model returned boxes that land on them. So the inline markers are used for
what they are good at (how many figures, and where each one belongs in the
reading order) and the dedicated request for the coordinates.

The result is still a model's estimate, so the box is finally snapped onto the
ink in the page image: grown over strokes it would have cut, then tightened to
what is actually there.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image

log = logging.getLogger("n2lh.figures")

_INT = r"\s*(-?\d+)\s*"
FIGBOX = re.compile(r"\\figbox\s*\{" + _INT + r"\}\s*\{" + _INT + r"\}\s*\{" + _INT + r"\}\s*\{" + _INT + r"\}")

# A figure has to be at least this big (in 1/1000ths of the page) to be worth a picture.
_MIN_SIDE = 15
# Padding around the snapped ink, as a fraction of the page's shorter side.
_PAD_FRAC = 0.008
# Widest crop kept, in pixels (~150 dpi across a 9in text block) to keep the PDF small.
_MAX_PX_WIDTH = 1400
# Fraction of the page width the notes' text block occupies (used to size the picture).
_TEXT_BLOCK = 0.86
# Hand-drawn figures read better a little smaller than their size on the page,
# and a printed figure never needs the full text width.
_SCALE = 0.82
_MAX_WIDTH_FRAC = 0.78
_MIN_WIDTH_FRAC = 0.20
# Ink darkness threshold (0-255) for snapping.
_INK = 145


_FIGS = r"(?:" + FIGBOX.pattern + r"\s*)+"
# The model sometimes puts the marker of a display drawing inside display math
# (`\[ \figbox{..} \]`). A picture in a `center` inside math mode does not compile
# ("Missing $ inserted") and no repair prompt can tell the model why, so a math
# wrapper around nothing but markers is dropped before they are replaced.
_MATH_WRAPPED = re.compile(
    r"(?:\\\[|\$\$)\s*(?P<a>" + _FIGS + r")(?:\\\]|\$\$)"
    r"|\\begin\{(?P<env>equation\*?|displaymath|align\*?|gather\*?)\}\s*(?P<b>" + _FIGS
    + r")\\end\{(?P=env)\}"
    r"|\$\s*(?P<c>" + _FIGS + r")\$")


def unwrap_math(latex: str) -> str:
    """Remove `\\[ \\]`, `$$ $$`, `$ $` or an equation environment that holds only figure markers."""
    def strip(m: "re.Match[str]") -> str:
        body = m.group("a") or m.group("b") or m.group("c")
        return "\n\n".join(f.group(0) for f in FIGBOX.finditer(body))
    return _MATH_WRAPPED.sub(strip, latex)


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def clamped(self) -> "Box":
        def c(v: int) -> int:
            return max(0, min(1000, v))
        x0, x1 = sorted((c(self.x0), c(self.x1)))
        y0, y1 = sorted((c(self.y0), c(self.y1)))
        return Box(x0, y0, x1, y1)

    def usable(self) -> bool:
        return self.w >= _MIN_SIDE and self.h >= _MIN_SIDE


def iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = a.area() + b.area() - inter
    return inter / union if union else 0.0


def reading_order(boxes: Sequence[Box]) -> List[Box]:
    """Top to bottom, then left to right -- the order the markers appear in the text."""
    return sorted(boxes, key=lambda b: (b.y0 + b.y1, b.x0 + b.x1))


def refine(inline: Sequence[Box], located: Sequence[Box], max_gap: int = 150) -> List[Box]:
    """Take the coordinates from ``located`` and the ordering from ``inline``.

    The inline markers sit in the transcription in reading order, so when the
    locator found the same number of figures the two lists describe the same
    drawings in the same order and are paired position by position. When the
    counts differ, each inline box takes the nearest unused located box whose
    vertical centre is within ``max_gap`` (1/1000ths of the page) and otherwise
    keeps its own coordinates.
    """
    if not located:
        return list(inline)
    ordered = reading_order(located)
    if len(ordered) == len(inline):
        return ordered

    def centre(b: Box) -> float:
        return (b.y0 + b.y1) / 2

    pairs = sorted((abs(centre(a) - centre(b)), i, j)
                   for i, a in enumerate(inline) for j, b in enumerate(ordered))
    out: List[Box] = list(inline)
    used_i, used_j = set(), set()
    for gap, i, j in pairs:
        if gap > max_gap:
            break
        if i in used_i or j in used_j:
            continue
        out[i] = ordered[j]
        used_i.add(i)
        used_j.add(j)
    return out


def parse_boxes(text: str) -> List[Box]:
    """Parse a model reply like ``[{"bbox": [x0,y0,x1,y1]}, ...]`` (fences allowed)."""
    import json
    if not text:
        return []
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return []
    boxes: List[Box] = []
    for item in data if isinstance(data, list) else []:
        raw = item.get("bbox") or item.get("bbox_2d") if isinstance(item, dict) else item
        try:
            x0, y0, x1, y1 = (int(round(float(v))) for v in raw)
        except (TypeError, ValueError):
            continue
        boxes.append(Box(x0, y0, x1, y1).clamped())
    return boxes


def _profile(dark: Image.Image, axis: int) -> List[int]:
    """Ink per row (``axis=0``) or per column (``axis=1``) of ``dark``."""
    w, h = dark.size
    if w < 1 or h < 1:
        return []
    line = dark.resize((1, h) if axis == 0 else (w, 1), Image.BOX)
    return list(line.convert("L").tobytes())


def _bands(profile: Sequence[int], gap: int) -> List[Tuple[int, int]]:
    """Runs of ink in ``profile``, merged across clear stretches shorter than ``gap``."""
    runs: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(profile):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(profile)))

    merged: List[Tuple[int, int]] = []
    for lo, hi in runs:
        if merged and lo - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))
    return merged


def line_height(dark: Image.Image) -> int:
    """The height of one line of this page's handwriting, in pixels.

    Measured from the page rather than assumed, so it follows the hand, the paper
    and the scan resolution. Only runs of ink that cross at least half the page
    are counted: those are lines of writing. Taking the median over *every* run
    instead gave 32 px on a page of sparse diagrams (it was measuring single
    strokes) and 168 px on a dense one (pairs of lines whose ascenders and
    descenders touch), neither of them a line.
    """
    W = dark.size[0]
    heights = []
    for lo, hi in _bands(_profile(dark, 0), 2):
        bbox = dark.crop((0, lo, W, hi)).getbbox()
        if bbox and (bbox[2] - bbox[0]) >= 0.5 * W:
            heights.append(hi - lo)
    if len(heights) < 3:
        return max(8, int(dark.size[1] * 0.022))         # too few lines to measure
    heights.sort()
    # The low quartile, not the median: in a close hand the descenders of one line
    # touch the ascenders of the next, so a run is usually a whole paragraph and
    # only the single-line paragraphs measure a line (257, 273, 290 px against the
    # 129 and 144 px of the one-line runs on the same page).
    return max(8, heights[len(heights) // 4])


def _trim_stray_bands(dark: Image.Image, px: Tuple[int, int, int, int],
                      line_h: int) -> Optional[Tuple[int, int, int, int]]:
    """Drop handwriting caught above or below the figure while the box was growing.

    Working inwards from each end, a band of ink is discarded when it looks like a
    line of writing: about as tall as this page's own line height, and running
    nearly the full width of the crop. The drawing -- the thickest band that is
    not writing -- is never dropped, and bands between kept ones stay, so a
    multi-part diagram survives. If every band is writing the locator pointed at
    prose, and None is returned so the figure is reported as omitted rather than
    embedding a picture of a sentence in the middle of the transcription.

    The height window is what makes this safe for drawings. A ruled line or an
    axis is far thinner than a line of writing, so the pair of long horizontal
    strokes that make up one figure here are kept even though each spans the whole
    crop; and bands are not merged across the small gaps inside a paragraph, so a
    two-line remark is two text lines rather than one thick block.

    Rules that looked reasonable and were not: thickness relative to the drawing (a
    two-line remark was 0.57 of one drawing's height), agreement with the located
    box (that box covered the remark too), and the size of the whitespace gap
    (which varies more between figures than between a figure and the next line).

    Only rows are trimmed. Writing runs horizontally, so a stray line is always a
    row; a column is one of the figure's own parts. Trimming columns by the same
    rule sheared the ``V`` and ``R^n`` labels off a chart diagram, because a column
    holding two labels at different heights is also thin and tall.
    """
    x0, y0, x1, y1 = px
    crop = dark.crop((x0, y0, x1, y1))
    width = crop.size[0]
    bands = _bands(_profile(crop, 0), max(2, int(line_h * 0.2)))
    if len(bands) < 2:
        return px

    def is_writing(i: int) -> bool:
        lo, hi = bands[i]
        # Measured on these notes: a drawn stroke spanning the crop (the long
        # horizontal lines of one figure) is 0.19-0.26 of a line, a line of
        # writing 0.42-1.52 even where the crop clips it, and a drawing 3.9.
        if not (0.35 * line_h <= hi - lo <= 1.8 * line_h):
            return False
        strip = crop.crop((0, lo, width, hi))
        bbox = strip.getbbox()
        if not bbox:
            return True                             # blank: nothing worth keeping
        return (bbox[2] - bbox[0]) >= 0.55 * width  # a caption or label is narrower

    writing = [is_writing(i) for i in range(len(bands))]
    drawn = [i for i, w in enumerate(writing) if not w]
    if not drawn:
        return None                                 # the box landed on prose
    # Anchor on the thickest band that is not writing: a three-line paragraph
    # above a small diagram is thicker than the diagram, and anchoring on the
    # thickest band of all would keep the paragraph and trim the diagram away.
    drawing = max(drawn, key=lambda i: bands[i][1] - bands[i][0])

    first, last = 0, len(bands) - 1
    while first < drawing and writing[first]:
        first += 1
    while last > drawing and writing[last]:
        last -= 1
    return x0, y0 + bands[first][0], x1, y0 + bands[last][1]


def snap_to_ink(dark: Image.Image, px: Tuple[int, int, int, int],
                line_h: int = 0) -> Optional[Tuple[int, int, int, int]]:
    """Move a model's estimate onto the drawing actually in the page image.

    ``dark`` is the page as a mask (ink = non-zero). Each edge is first pushed
    outward while the strip just outside it still has ink, so a box that cuts
    through a stroke or stops short of a label grows to contain the whole
    drawing; growth ends at the first clear strip, which is the whitespace
    around the figure, and is capped so it cannot swallow the text block. The
    box is then tightened onto the ink inside it. Returns None when the model
    pointed at blank paper, so that becomes a visible omission instead of a
    blank picture.
    """
    W, H = dark.size
    short = min(W, H)
    step = max(2, int(short * 0.004))
    gap = max(6, int(short * 0.011))         # a clear strip this deep ends the growth
    x0, y0, x1, y1 = px
    # How far an edge may travel. Growth is only meant to recover a stroke or label
    # the estimate clipped, so it stays well under one line of handwriting (~3.3% of
    # the page on these notes): where the lines are tightly spaced their gaps are
    # shallower than `gap`, and a bigger budget walked a box up through two whole
    # lines of text before any of them could be trimmed away.
    limit = max(8, int(short * 0.025))
    limit_x = limit_y = limit

    def has_ink(l: int, t: int, r: int, b: int) -> bool:
        l, t = max(0, l), max(0, t)
        r, b = min(W, r), min(H, b)
        return r > l and b > t and dark.crop((l, t, r, b)).getbbox() is not None

    travelled = 0
    while travelled < limit_y and y0 > 0 and has_ink(x0, y0 - gap, x1, y0):
        y0 = max(0, y0 - step)
        travelled += step
    travelled = 0
    while travelled < limit_y and y1 < H and has_ink(x0, y1, x1, y1 + gap):
        y1 = min(H, y1 + step)
        travelled += step
    travelled = 0
    while travelled < limit_x and x0 > 0 and has_ink(x0 - gap, y0, x0, y1):
        x0 = max(0, x0 - step)
        travelled += step
    travelled = 0
    while travelled < limit_x and x1 < W and has_ink(x1, y0, x1 + gap, y1):
        x1 = min(W, x1 + step)
        travelled += step

    def tighten(box: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        l, t, r, b = box
        inside = dark.crop(box).getbbox()
        return None if not inside else (l + inside[0], t + inside[1],
                                        l + inside[2], t + inside[3])

    box = tighten((x0, y0, x1, y1))
    if box is None:
        return None                                  # blank paper: nothing to show
    # Dropping a text line above a figure leaves the crop as wide as that line, so
    # tighten once more on what survived.
    trimmed = _trim_stray_bands(dark, box, line_h or line_height(dark))
    if trimmed is None:
        return None                                  # writing, not a drawing
    box = tighten(trimmed) or box
    x0, y0, x1, y1 = box
    pad = max(4, int(short * _PAD_FRAC))
    return (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))


def _crop(page: Image.Image, dark: Image.Image, line_h: int,
          box: Box, dest: Path) -> Optional[float]:
    """Crop ``box`` out of the page image into ``dest``; return the crop's width as a
    fraction of the page, or None if there is nothing usable there."""
    W, H = page.size
    px = (int(box.x0 / 1000 * W), int(box.y0 / 1000 * H),
          int(box.x1 / 1000 * W), int(box.y1 / 1000 * H))
    snapped = snap_to_ink(dark, px, line_h)
    if snapped is None:
        return None
    px = snapped
    if px[2] - px[0] < 8 or px[3] - px[1] < 8:
        return None
    crop = page.crop(px)
    if crop.width > _MAX_PX_WIDTH:
        scale = _MAX_PX_WIDTH / crop.width
        crop = crop.resize((_MAX_PX_WIDTH, max(1, int(crop.height * scale))), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dest, "PNG", optimize=True)
    return (px[2] - px[0]) / W


def render_figures(latex: str, image_path: Path, figures_dir: Path, page_index: int,
                   locator: Optional[Callable[[], Optional[Sequence[Box]]]] = None) -> str:
    """Replace every ``\\figbox{..}`` in ``latex`` with an ``\\includegraphics``.

    ``locator`` (optional) returns a list of accurately located boxes for the
    page; failures in it are non-fatal (the inline boxes are used as they are).
    """
    latex = unwrap_math(latex)
    matches = list(FIGBOX.finditer(latex))
    if not matches:
        return latex
    inline = [Box(*(int(g) for g in m.groups())).clamped() for m in matches]
    boxes = inline
    if locator is not None:
        try:
            located = locator()
        except Exception as exc:  # noqa: BLE001 - refinement is best-effort
            log.warning("figure locator failed on page %d: %s", page_index, exc)
            located = None
        if located:
            boxes = refine(inline, [(b if isinstance(b, Box) else Box(*b)).clamped()
                                    for b in located])

    # The page is opened, binarised and measured once, not once per figure.
    with Image.open(image_path) as opened:
        page = opened.convert("RGB")
    dark = page.convert("L").point(lambda v: 255 if v < _INK else 0)
    line_h = line_height(dark)

    out: List[str] = []
    last = 0
    for k, (m, box) in enumerate(zip(matches, boxes), 1):
        out.append(latex[last:m.start()])
        last = m.end()
        name = f"p{page_index:04d}_f{k}.png"
        frac = (_crop(page, dark, line_h, box, figures_dir / name)
                if box.usable() else None)
        if frac is None:
            out.append("\\textit{[figure omitted]}")
            continue
        width = min(_MAX_WIDTH_FRAC, max(_MIN_WIDTH_FRAC, frac / _TEXT_BLOCK * _SCALE))
        out.append("\\begin{center}\n"
                   f"\\includegraphics[width={width:.2f}\\linewidth]{{{name}}}\n"
                   "\\end{center}")
    out.append(latex[last:])
    return "".join(out)
