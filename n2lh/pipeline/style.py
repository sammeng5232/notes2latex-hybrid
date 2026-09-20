"""Typographic conventions of the notes that the transcription should keep.

Lecture notes label what they are doing -- ``Def.``, ``Thm (Poincare duality).``,
``Pf.``, ``Eg.`` -- and a lecture starts with its number, date and week. In the
handwriting those labels carry the structure of the page; in a transcription
they are indistinguishable from running text unless they are set in bold.

Asking the model for the bold is not enough on its own: it obliges on some
pages and forgets on others, the same page comes back different on a repair
pass, and nothing in the compile output reveals the omission. This pass is
deterministic, so the rule is applied to every page in the same way, and it runs
after the model on top of whatever the prompt achieved -- a label the model
already emboldened is left alone.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# The labels these notes use. A trailing `.` or `:` is required, so "Defined"
# and "Corollaries" (and an "Eg" inside a word) are never touched.
_LABELS = (
    "Def", "Defn", "Definition", "Thm", "Theorem", "Cor", "Corollary", "Lem", "Lemma",
    "Prop", "Proposition", "Claim", "Conj", "Conjecture", "Pf", "Prf", "Proof",
    "Eg", "E\\.g", "Ex", "Example", "Rmk", "Rem", "Remark", "Note", "Obs", "Observation",
    "Fact", "Notation", "Sps", "Q", "Ans", "Soln", "Step",
)
# `Thm (Hopf degree thm).` -- the parenthetical names the result and belongs to the label.
_LABEL_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<label>(?:" + "|".join(_LABELS) + r")"
    r"(?:\s*\([^()\n]{0,60}\))?"
    r"\s*[.:])")

# `Lecture 9 2025/11/9 Week 12`, `Lecture 3 20250917 Week 3`, `Lecture 1, 3 Sep, Week 1`.
_LECTURE_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<head>Lecture\s*\d+[^\n\\{}$]*?(?:Week\s*\d+|\d{4}[-/]?\d{1,2}[-/]?\d{1,2}))"
    r"(?P<rest>[ \t]*\.?)\s*$",
    re.IGNORECASE)

_MATH_OPEN = re.compile(r"\\\[|\$\$")
_MATH_CLOSE = re.compile(r"\\\]|\$\$")
_MATH_ENVS = {
    "equation", "equation*", "align", "align*", "alignat", "alignat*", "gather",
    "gather*", "multline", "multline*", "flalign", "flalign*", "eqnarray", "eqnarray*",
    "displaymath", "math", "array", "aligned", "cases", "tikzpicture", "tikzcd",
    "verbatim", "lstlisting", "tabular",
}
_BEGIN = re.compile(r"\\begin\{([A-Za-z*]+)\}")
_END = re.compile(r"\\end\{([A-Za-z*]+)\}")


def _display_math_depth(line: str, depth: int) -> int:
    """Track `\\[ ... \\]` / `$$ ... $$` spanning several lines."""
    for m in re.finditer(r"\\\[|\\\]|\$\$", line):
        token = m.group(0)
        if token == "$$":
            depth = 0 if depth else 1
        elif token == "\\[":
            depth += 1
        else:
            depth = max(0, depth - 1)
    return depth


def bold_labels(latex: str) -> Tuple[str, int]:
    """Bold the note labels that start a line. Returns ``(latex, n_bolded)``."""
    out: List[str] = []
    n = 0
    depth = 0
    envs: List[str] = []
    for line in latex.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        newline = line[len(body):]
        skip = depth > 0 or any(e in _MATH_ENVS for e in envs) or body.lstrip().startswith("%")

        if not skip:
            m = _LECTURE_RE.match(body)
            if m:
                body = f"{m.group('indent')}\\textbf{{{m.group('head').rstrip()}}}{m.group('rest')}"
                n += 1
            else:
                m = _LABEL_RE.match(body)
                if m and not body.lstrip().startswith("\\textbf"):
                    label = m.group("label").rstrip()
                    body = f"{m.group('indent')}\\textbf{{{label}}}{body[m.end():]}"
                    n += 1

        out.append(body + newline)
        depth = _display_math_depth(body, depth)
        for env in _BEGIN.findall(body):
            envs.append(env)
        for env in _END.findall(body):
            if env in envs:
                envs.remove(env)
    return "".join(out), n
