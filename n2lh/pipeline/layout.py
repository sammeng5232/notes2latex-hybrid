"""Make a transcribed page fit on the page, and fix the indicator function.

Two faults that compile cleanly and are wrong on paper, so only a deterministic
pass catches them:

* ``\\mathbb{1}`` for an indicator. amssymb's blackboard alphabet has no digits,
  so the glyph that comes out is ``\\nVdash`` -- a struck-through turnstile. One
  page of probability notes used it twenty times. The fix is ``\\mathbbm{1}``
  from bbm, which the preamble now loads.
* A table wider than the text block. LaTeX sets it at its natural width and lets
  it run off the paper: the summary table on that same page ended up 9.36in past
  a 6.27in text block, its last columns simply gone. Wrapping it in ``\\fitpage``
  scales it down, and only when it is too wide.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# \mathbb{1}, \mathbb 1 and \mathbb1 all mean the indicator.
_MATHBB_ONE = re.compile(r"\\mathbb\s*(?:\{\s*1\s*\}|1)")
# \mathds{1} is the dsfont spelling; the preamble loads bbm, not dsfont.
_MATHDS_ONE = re.compile(r"\\mathds\s*(?:\{\s*1\s*\}|1)")

_FIT = "\\fitpage"
# Environments laid out at their natural width, which may exceed the text block.
_WIDE_ENVS = ("tabular", "tabularx", "tabular*")


def fix_indicator(latex: str) -> Tuple[str, int]:
    """``\\mathbb{1}`` / ``\\mathds{1}`` -> ``\\mathbbm{1}``."""
    n = 0

    def sub(_m: "re.Match[str]") -> str:
        nonlocal n
        n += 1
        return r"\mathbbm{1}"

    latex, k = _MATHBB_ONE.subn(sub, latex)
    latex, _ = _MATHDS_ONE.subn(sub, latex)
    return latex, n


def _find_env(latex: str, name: str, start: int) -> Tuple[int, int]:
    """Span of the ``name`` environment beginning at ``start``, nesting-aware."""
    depth = 0
    pattern = re.compile(r"\\(begin|end)\{" + re.escape(name) + r"\}")
    for m in pattern.finditer(latex, start):
        depth += 1 if m.group(1) == "begin" else -1
        if depth == 0:
            return start, m.end()
    return start, -1


def fit_wide_tables(latex: str) -> Tuple[str, int]:
    """Wrap each outermost table in ``\\fitpage{...}`` so it cannot overflow.

    A table already inside a ``\\fitpage`` or ``\\resizebox`` is left alone, and
    nested tables are covered by the wrap around the outer one.
    """
    out: List[str] = []
    pos = 0
    n = 0
    begins = re.compile(r"\\begin\{(" + "|".join(re.escape(e) for e in _WIDE_ENVS) + r")\}")
    while True:
        m = begins.search(latex, pos)
        if not m:
            break
        start, end = _find_env(latex, m.group(1), m.start())
        if end < 0:                                  # unbalanced: leave it for autofix
            break
        before = latex[pos:start]
        # Already scaled by hand (or by an earlier run of this pass)?
        tail = before.rstrip()
        if tail.endswith(_FIT + "{") or tail.endswith("}{") and _FIT in tail[-40:]:
            out.append(latex[pos:end])
        else:
            out.append(before)
            out.append(_FIT + "{" + latex[start:end] + "}")
            n += 1
        pos = end
    out.append(latex[pos:])
    return "".join(out), n


# A margin section label as written by a strip read: \textbf{② 测度}, possibly
# colored. Between it and a Chinese word the space vanishes (xeCJK drops spaces
# next to CJK), so "② 测度 定理 2.1" printed as "测度定理 2.1".
_MARGIN_LABEL = re.compile(
    r"((?:\\textcolor\{\w+\}\{)?\\textbf\{[\u2460-\u2473\u24f5-\u24fe][^{}\n]{0,12}\}\}?)"
    r"[ \t]+(?=\S)")
# Text directly above a tabular inside a wraptable, with no break between: the
# heading was set BESIDE the table instead of above it.
_WRAP_HEADING = re.compile(
    r"(\\begin\{wraptable\}\{[^}]*\}\{[^}]*\}[ \t]*\n)([^\n]*\S)[ \t]*\n"
    r"(?=[ \t]*(?:\\fitpage\{)?\\begin\{tabular\})")


def space_margin_labels(latex: str) -> Tuple[str, int]:
    """Put a gap between a margin section label and the text after it."""
    return _MARGIN_LABEL.subn(lambda m: m.group(1) + "\\quad ", latex)


def heading_above_corner_table(latex: str) -> Tuple[str, int]:
    """End a wraptable's heading line before its tabular starts."""
    def fix(m: "re.Match[str]") -> str:
        heading = m.group(2)
        if heading.rstrip().endswith(("\\\\", "\\par")):
            return m.group(0)
        return m.group(1) + heading + "\\par\n"
    out, n = _WRAP_HEADING.subn(fix, latex)
    return out, (n if out != latex else 0)


def tidy_layout(latex: str) -> Tuple[str, List[str]]:
    """Both passes. Returns ``(latex, changes)``; ``changes`` is empty when clean."""
    changes: List[str] = []
    latex, n_ind = fix_indicator(latex)
    if n_ind:
        changes.append(f"indicator: {n_ind} \\mathbb{{1}} -> \\mathbbm{{1}}")
    latex, n_fit = fit_wide_tables(latex)
    if n_fit:
        changes.append(f"fitted {n_fit} table(s) to the text width")
    latex, n_gap = space_margin_labels(latex)
    if n_gap:
        changes.append(f"spaced {n_gap} margin label(s)")
    latex, n_head = heading_above_corner_table(latex)
    if n_head:
        changes.append("put the corner table's heading above it")
    return latex, changes
