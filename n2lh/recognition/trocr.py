"""TrOCR adapter for a fine-tuned image->LaTeX checkpoint.

This is the Math2LaTeX heritage slot: point it at a checkpoint produced by
fine-tuning a vision-encoder/decoder (e.g. TrOCR) on handwritten math, and the
document pipeline will use it as the fully-offline primary engine -- per-region
recognition driven by the classical segmentation stage, no API calls at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from n2lh.pipeline.segment import segment_lines
from n2lh.recognition.base import PageImage, Recognizer, TranscribeResult


class TrOCRError(RuntimeError):
    pass


class TrOCRRecognizer(Recognizer):
    name = "trocr"

    def __init__(self, model_dir: str, max_regions: int = 64) -> None:
        self.model_dir = Path(model_dir)
        self.max_regions = max_regions
        if not self.model_dir.exists():
            raise TrOCRError(f"model dir not found: {self.model_dir}")
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise TrOCRError(
                "TrOCR engine requires: pip install 'notes2latex-hybrid[trocr]'"
            ) from exc
        try:
            self._processor = TrOCRProcessor.from_pretrained(str(self.model_dir))
            self._model = VisionEncoderDecoderModel.from_pretrained(str(self.model_dir))
            self._model.eval()
        except Exception as exc:  # transformers raises many shapes of failure
            raise TrOCRError(f"failed to load checkpoint: {exc}") from exc

    def transcribe(self, page: PageImage, context_tail: str,
                   open_environments: List[str],
                   guidance: Optional[str] = None) -> TranscribeResult:
        import torch
        from PIL import Image

        regions = segment_lines(Image.open(page.path))[: self.max_regions]
        outputs: List[str] = []
        for i, region in enumerate(regions, 1):
            if region.image is None:
                continue
            pixel_values = self._processor(images=region.image.convert("RGB"),
                                           return_tensors="pt").pixel_values
            with torch.no_grad():
                generated = self._model.generate(pixel_values, max_new_tokens=384)
            text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
            text = text.strip()
            outputs.append(_wrap(text, region.kind, i))
        latex = "\n".join(outputs)
        if guidance:
            # A fine-tuned OCR model cannot read error logs; note it and let
            # the pipeline escalate to a VLM fixer if one is configured.
            latex += "\n% note: trocr engine cannot perform log-guided repairs"
        return TranscribeResult(latex=latex, engine=self.name,
                                notes=f"{len(outputs)} regions recognized")

    def close(self) -> None:
        self._model = None
        self._processor = None


def _wrap(text: str, kind: str, index: int) -> str:
    text = text.strip()
    if not text:
        return f"% region {index}: empty"
    if kind == "math":
        return f"\\begin{{equation*}}{text}\\end{{equation*}}"
    return text
