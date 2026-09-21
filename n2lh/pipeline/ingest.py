"""Input ingestion: PDFs and images -> normalized page PNGs."""

from __future__ import annotations

import shutil
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


class IngestError(RuntimeError):
    pass


# A pixel counts as colored ink when it is saturated AND not paper-bright in
# every channel; see color_ink_names for why both are needed.
_SAT_MIN = 40
_BRIGHT_MAX = 220


def _colored_mask(arr: np.ndarray) -> np.ndarray:
    sat = arr.max(axis=2) - arr.min(axis=2)
    return (sat > _SAT_MIN) & (arr.min(axis=2) < _BRIGHT_MAX)


def _classify(arr: np.ndarray, colored: np.ndarray) -> List[str]:
    """Ordered, mutually exclusive color-family names for a colored mask."""
    n = int(colored.sum())
    if n == 0:
        return []
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    families = (
        ("pink",   (r > g + 25) & (b > g + 25) & (b <= r + 25)),
        ("violet", (b > r + 25) & (r > g + 25)),
        ("orange", (r > b + 40) & (g > b + 25) & (r > g + 25)),
        ("yellow", (r > b + 40) & (g > b + 40) & (g > r - 40)),
        ("green",  (g > r + 25) & (g > b + 25)),
        ("cyan",   (g > r + 25) & (b > r + 25) & (np.abs(g - b) <= 60)),
        ("blue",   (b > r + 25) & (b > g + 25)),
        ("red",    (r > g + 25) & (r > b + 25)),
    )
    claimed = np.zeros(colored.shape, dtype=bool)
    names: List[str] = []
    for name, mask in families:
        m = colored & mask & ~claimed
        if int(m.sum()) >= 0.05 * n:
            names.append(name)
        claimed |= colored & mask
    return names


def color_ink_names(img: Image.Image) -> List[str]:
    """Names of the colored-ink families on the page, e.g. ["pink", "green"].

    Empty when there is no colored ink. Used twice: to decide whether to keep
    color at all, and to tell the recognizer WHAT to look for. A generic "mind
    the colors" line in a long system prompt was not acted on (a vocabulary
    table's English column, written in pink pen, came out black); a per-page
    directive that names the colors gives the model something concrete to find.

    "Colored" needs both channels of evidence: saturation (max-min RGB > 40)
    and ink-darkness (min RGB < 220). Saturation alone counts a tinted paper
    background or JPEG chroma noise; darkness alone is any black ink. A red or
    green pen stroke and a highlighter swipe satisfy both. The family names
    (pink covers magenta, violet covers purple) are anchors for the model, not
    exact xcolor names -- the prompt tells it to use a standard name.
    """
    small = img.convert("RGB")
    small.thumbnail((800, 800))
    arr = np.asarray(small).astype(int)
    colored = _colored_mask(arr)
    n = int(colored.sum())
    if n <= 2e-4 * arr.shape[0] * arr.shape[1]:
        return []
    return _classify(arr, colored)


def colored_ink_regions(img: Image.Image, max_regions: int = 3) -> List[Tuple[Tuple[int, int, int, int], List[str]]]:
    """Where the colored ink is: up to ``max_regions`` ``(box, names)`` pairs.

    Scattered saturated specks (JPEG noise, a stray mark) would stretch one
    bounding box over most of the page, so pixels are bucketed into a 50x50
    grid, only dense cells count, and connected dense cells become regions --
    the same density trick the table and figure finders use. On the real page
    that prompted this it separated a pink vocabulary column on the right from
    green margin marks on the left, which is exactly the split a focused
    re-read needs: one crop per colored thing, no black bulk around it.
    """
    arr = np.asarray(img.convert("RGB")).astype(int)
    h, w, _ = arr.shape
    colored = _colored_mask(arr)
    n = int(colored.sum())
    if n <= 2e-4 * h * w:
        return []
    ys, xs = np.where(colored)
    G = 50
    cells: dict = {}
    for y, x in zip(ys, xs):
        c = (int(y * G // h), int(x * G // w))
        cells[c] = cells.get(c, 0) + 1
    # A dense cell holds real strokes, not scattered noise.
    hot_threshold = max(20, n // 500)
    hot = {c for c, k in cells.items() if k >= hot_threshold}
    seen, comps = set(), []
    for c in hot:
        if c in seen:
            continue
        q, comp = deque([c]), []
        seen.add(c)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    m = (cur[0] + dy, cur[1] + dx)
                    if m in hot and m not in seen:
                        seen.add(m)
                        q.append(m)
        comps.append(comp)
    comps.sort(key=lambda comp: -sum(cells[c] for c in comp))
    keep_min = max(500, int(0.03 * n))
    regions = []
    for comp in comps:
        if sum(cells[c] for c in comp) < keep_min or len(regions) >= max_regions:
            break
        cys = [c[0] for c in comp]
        cxs = [c[1] for c in comp]
        pad_y, pad_x = max(2, int(h * 0.012)), max(2, int(w * 0.012))
        box = (max(0, int(min(cxs) * w / G) - pad_x),
               max(0, int(min(cys) * h / G) - pad_y),
               min(w, int((max(cxs) + 1) * w / G) + pad_x),
               min(h, int((max(cys) + 1) * h / G) + pad_y))
        sub = arr[box[1]:box[3], box[0]:box[2]]
        m = colored[box[1]:box[3], box[0]:box[2]]
        regions.append((box, _classify(sub, m)))
    return regions


def _has_color_ink(img: Image.Image) -> bool:
    """True when the page carries visibly colored ink (see color_ink_names).

    Handwritten notes are often two or three pens deep: black for the body, red
    or blue for corrections, emphasis and status marks. Flattening that to
    grayscale at ingestion deletes the information before the recognizer ever
    sees the page (on a real one-page summary the vocabulary table's English
    column was written in pink pen and came out black, and the model was blamed
    for "not transcribing the colors"). A page with no colored ink keeps the
    old, much smaller grayscale file.
    """
    return bool(color_ink_names(img))


def _save_page(img: Image.Image, outdir: Path, index: int) -> Path:
    rgb = img.convert("RGB")
    # Grayscale PNGs are far smaller; color is kept only when there is any.
    img = rgb if _has_color_ink(rgb) else rgb.convert("L")
    path = outdir / f"{index:04d}.png"
    img.save(path, "PNG")
    return path


def ingest_files(paths: List[Path], outdir: Path, dpi: int = 300) -> List[Path]:
    """Convert each input file to page PNG(s) in outdir (color kept when present).

    Returns pages sorted."""
    outdir.mkdir(parents=True, exist_ok=True)
    pages: List[Path] = []
    for src in paths:
        suffix = src.suffix.lower()
        if suffix == ".pdf":
            pages.extend(_pdf_to_pages(src, outdir, dpi, start=len(pages)))
        elif suffix in IMAGE_SUFFIXES:
            with Image.open(src) as img:
                pages.append(_save_page(img, outdir, len(pages) + 1))
        else:
            raise IngestError(f"Unsupported input type: {src.name}")
    if not pages:
        raise IngestError("No readable pages found in input files.")
    return sorted(pages)


def _pdf_to_pages(src: Path, outdir: Path, dpi: int, start: int) -> List[Path]:
    try:
        import pypdfium2 as pdfium  # optional dependency
    except ImportError as exc:
        raise IngestError(
            "PDF support requires pypdfium2: pip install 'notes2latex-hybrid[pdf]'"
        ) from exc
    pages: List[Path] = []
    scale = dpi / 72.0
    doc = pdfium.PdfDocument(str(src))
    try:
        for i, page in enumerate(doc):
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil()
            pages.append(_save_page(pil, outdir, start + i + 1))
    finally:
        doc.close()
    return pages
