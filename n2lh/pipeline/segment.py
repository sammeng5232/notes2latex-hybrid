"""Classical line/equation segmentation (offline, no ML required).

Inspired by the segmentation stage of Math2LaTeX (their Mask R-CNN version was
listed as unimplemented; this projection-profile + connected-component baseline
runs anywhere with zero downloads and is what the heuristic engine uses).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class Region:
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1
    kind: str                        # "text" | "math" | "unknown"
    ink_ratio: float
    image: Optional[Image.Image] = None
    text: str = ""


def _to_mask(pil: Image.Image, threshold: int = 200) -> np.ndarray:
    """Binarize to a boolean ink mask (True = ink)."""
    arr = np.asarray(pil.convert("L"))
    return arr < threshold


def _row_bands(mask: np.ndarray, min_gap: int) -> List[Tuple[int, int]]:
    """Group ink rows into horizontal bands (y0, y1)."""
    rows = mask.any(axis=1)
    bands: List[Tuple[int, int]] = []
    start = None
    gap = 0
    for y, has in enumerate(rows):
        if has:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                bands.append((start, y - gap + 1))
                start, gap = None, 0
    if start is not None:
        bands.append((start, len(rows) - 1))
    return bands


def _column_span(mask_band: np.ndarray) -> Tuple[int, int]:
    cols = mask_band.any(axis=0)
    xs = np.nonzero(cols)[0]
    if xs.size == 0:
        return 0, 0
    return int(xs[0]), int(xs[-1]) + 1


def _component_heights(mask_band: np.ndarray) -> List[int]:
    """Heights of connected components via simple run-based labeling (4-conn rows)."""
    h, w = mask_band.shape
    if h == 0 or w == 0:
        return []
    # Two-pass union-find on a downsampled mask for speed.
    step = max(1, min(h, w) // 60)
    small = mask_band[::step, ::step]
    sh, sw = small.shape
    labels = np.zeros((sh, sw), dtype=np.int32)
    parent = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    next_label = 1
    for y in range(sh):
        for x in range(sw):
            if not small[y, x]:
                continue
            left = labels[y, x - 1] if x > 0 else 0
            up = labels[y - 1, x] if y > 0 else 0
            if left and up:
                labels[y, x] = min(left, up)
                union(left, up)
            elif left or up:
                labels[y, x] = left or up
            else:
                labels[y, x] = next_label
                parent.append(next_label)
                next_label += 1

    roots = {}
    for y in range(sh):
        for x in range(sw):
            if labels[y, x]:
                r = find(labels[y, x])
                roots.setdefault(r, [y, y])
                roots[r][0] = min(roots[r][0], y)
                roots[r][1] = max(roots[r][1], y)
    return [(b - a + 1) * step for a, b in roots.values()]


def _classify(band_mask: np.ndarray) -> str:
    """Guess text vs math from glyph geometry.

    Math lines contain tall glyphs (fractions, integrals, sum bounds, radicals)
    producing a large max/median component-height ratio.
    """
    heights = _component_heights(band_mask)
    if len(heights) < 2:
        return "unknown"
    med = float(np.median(heights))
    mx = float(np.max(heights))
    if med <= 0:
        return "unknown"
    if mx / med > 1.9:
        return "math"
    # Superscript/subscript pairs also signal math: many tiny components.
    tiny = sum(1 for hgt in heights if hgt < 0.55 * med)
    if tiny >= 3 and tiny / len(heights) > 0.3:
        return "math"
    return "text"


def segment_lines(pil: Image.Image, min_gap: Optional[int] = None,
                  pad: int = 4) -> List[Region]:
    """Split a page image into line regions with text/math guesses."""
    mask = _to_mask(pil)
    h = mask.shape[0]
    if min_gap is None:
        min_gap = max(4, h // 120)
    regions: List[Region] = []
    for y0, y1 in _row_bands(mask, min_gap):
        if y1 - y0 < max(6, h // 200):  # noise
            continue
        band = mask[y0:y1]
        x0, x1 = _column_span(band)
        if x1 <= x0:
            continue
        ink_ratio = float(band[:, x0:x1].mean())
        kind = _classify(band)
        crop = pil.crop((max(0, x0 - pad), max(0, y0 - pad),
                         min(pil.width, x1 + pad), min(pil.height, y1 + pad)))
        regions.append(Region(bbox=(x0, y0, x1, y1), kind=kind, ink_ratio=ink_ratio,
                              image=crop))
    return regions
