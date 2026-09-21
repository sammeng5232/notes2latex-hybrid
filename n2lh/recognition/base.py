"""Recognizer contract shared by all engines."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class PageImage:
    index: int                 # 1-based page number
    path: Path


@dataclass
class TranscribeResult:
    latex: str                 # body LaTeX for this page (no preamble)
    engine: str                # which engine produced it
    confidence: Optional[float] = None
    notes: Optional[str] = None


class Recognizer(abc.ABC):
    """A page -> LaTeX transcriber.

    ``guidance`` is non-empty on fix passes: it carries the compiler errors for
    the previous attempt so capable engines can repair instead of redo.
    Engines that cannot repair may just re-run their normal path.
    """

    name: str = "base"

    @abc.abstractmethod
    def transcribe(self, page: PageImage, context_tail: str,
                   open_environments: List[str],
                   guidance: Optional[str] = None) -> TranscribeResult:
        ...

    def transcribe_table(self, image_path) -> Optional[str]:
        """A LaTeX tabular read from a crop holding just one table, or None.

        Optional: an engine that cannot do a second, focused pass returns None
        and the page's own transcription of the table stands.
        """
        return None

    def transcribe_colors(self, image_path, names: Optional[List[str]] = None) -> Optional[str]:
        """LaTeX for the colored content of a crop, with \\textcolor runs, or
        None.

        Optional: an engine that cannot do a focused color pass returns None
        and the page keeps whatever colors its own transcription marked.
        """
        return None

    def locate_figures(self, page: PageImage) -> Optional[List[Tuple[int, int, int, int]]]:
        """Accurate figure boxes for ``page`` as ``(x0, y0, x1, y1)`` in 0..1000
        (relative to the image, origin top-left), or None if this engine cannot
        say. Used to refine the coarse ``\\figbox`` boxes in a transcription."""
        return None

    def close(self) -> None:  # release resources (http clients, models)
        pass
