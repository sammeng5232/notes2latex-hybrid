"""Input ingestion: PDFs and images -> normalized page PNGs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


class IngestError(RuntimeError):
    pass


def _save_page(img: Image.Image, outdir: Path, index: int) -> Path:
    img = img.convert("L")
    path = outdir / f"{index:04d}.png"
    img.save(path, "PNG")
    return path


def ingest_files(paths: List[Path], outdir: Path, dpi: int = 300) -> List[Path]:
    """Convert each input file to grayscale page PNG(s) in outdir. Returns pages sorted."""
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
