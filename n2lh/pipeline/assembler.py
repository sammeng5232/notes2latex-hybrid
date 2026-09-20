"""Document assembly: preamble + body, compile-safe closing of environments."""

from __future__ import annotations

from n2lh.recognition.prompts import PREAMBLE_TEX


def build_document(body: str, preamble: str = PREAMBLE_TEX, closed_body: str = "") -> str:
    """Assemble a full compilable document.

    ``body`` is the raw accumulated LaTeX; ``closed_body`` optionally supplies a
    variant with dangling-environment closers appended (produced by
    EnvironmentTracker.close_for_compile) used for the compiled copy.
    """
    content = closed_body if closed_body else body
    return f"{preamble}\n\\begin{{document}}\n{content.strip()}\n\\end{{document}}\n"
