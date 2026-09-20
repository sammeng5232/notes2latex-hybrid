"""Offline heuristic engine (zero dependencies, zero downloads, fully private).

It cannot read handwriting -- that is what the TrOCR adapter or a VLM is for.
What it DOES provide, fully offline, is the document *structure*: it segments
each page into lines, classifies them as text/math by glyph geometry, and emits
a compilable skeleton with clearly-marked slots. This is the default primary in
hybrid mode (local structure pass, VLM only repairs/escalates failing pages)
and the engine used by tests and demos.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from n2lh.pipeline.segment import segment_lines
from n2lh.recognition.base import PageImage, Recognizer, TranscribeResult


class HeuristicRecognizer(Recognizer):
    name = "heuristic"

    def __init__(self, safe_mode_after_fix: bool = True) -> None:
        self._safe_mode_after_fix = safe_mode_after_fix

    def transcribe(self, page: PageImage, context_tail: str,
                   open_environments: List[str],
                   guidance: Optional[str] = None) -> TranscribeResult:
        with Image.open(page.path) as pil:
            regions = segment_lines(pil)
        safe = bool(guidance) and self._safe_mode_after_fix
        lines: List[str] = [f"% page {page.index}: {len(regions)} segmented regions"]
        for i, region in enumerate(regions, 1):
            if safe:
                # Fix pass: guarantee-compilable output so the pipeline can
                # always make progress offline.
                lines.append(f"\\text{{[region {i}: {region.kind}]}}\n")
            elif region.kind == "math":
                lines.append(f"\\begin{{equation*}} \\text{{[math region {i}]}} \\end{{equation*}}")
            else:
                lines.append(f"\\text{{[text region {i}]}}")
        latex = "\n".join(lines)
        return TranscribeResult(
            latex=latex,
            engine=self.name,
            confidence=None,
            notes=("safe-mode skeleton" if safe else f"{len(regions)} regions"),
        )
