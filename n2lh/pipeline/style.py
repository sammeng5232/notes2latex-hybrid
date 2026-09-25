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

# Chinese notes number their statements: `定理 1.16 (Bolzano-Weierstrass).`,
# `推论 2.2.`, `定义 3.6.`. The English pattern above never matched these -- it
# has no Chinese words, and it wants the punctuation straight after the label,
# so even `Thm 3.2.` slipped past. A number is required here, which is what
# makes a label safe to bold where it stands mid-line: a dense summary sheet
# writes `...次可加。 推论 2.2. 对可数点集...`, two statements on one line.
_ZH_LABELS = ("定理", "定义", "推论", "引理", "命题", "性质", "公理", "例题", "例",
              "注记", "注", "证明")
_ZH_LABEL_RE = re.compile(
    # Not after a backslash or an ASCII letter/digit. A CJK character before it
    # is fine: the required number already keeps `延拓定理` (no number) out, and
    # a margin mark such as `预` often sits right against the label.
    r"(?<![\\A-Za-z0-9])(?P<label>(?:" + "|".join(_ZH_LABELS) + r")"
    r"\s*\d+(?:\.\d+)*"                       # the number is required
    r"(?:\s*[(（][^()（）\n$]{0,40}[)）])?"   # (Bolzano-Weierstrass), （逐项积分）
    r"\s*[.．:：]?)")
# Marks from the left-margin column that a transcription keeps at the start of
# the line they stand beside: a circled section number, or one character of a
# vertical section label (预 / 备, 测 / 度 ...). The label after them is bolded,
# the mark itself is left as it is.
_MARGIN_PREFIX = re.compile(r"^(?P<margin>[ \t]*(?:[①-⑳][ \t]*)?(?:[一-鿿][ \t]+)?)")

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


_INLINE_MATH = re.compile(r"(\$[^$]*\$)")


def _bold_zh(line: str) -> Tuple[str, int]:
    """Bold every numbered Chinese label in the text parts of one line."""
    n = 0

    def sub(m: "re.Match[str]") -> str:
        nonlocal n
        before = m.string[:m.start()].rstrip()
        if before.endswith("\\textbf{"):
            return m.group(0)                      # the model already bolded it
        n += 1
        label = m.group("label").rstrip()
        return f"\\textbf{{{label}}}" + m.group(0)[len(m.group("label").rstrip()):]

    parts = _INLINE_MATH.split(line)
    for i in range(0, len(parts), 2):             # even parts are text, odd are $...$
        parts[i] = _ZH_LABEL_RE.sub(sub, parts[i])
    return "".join(parts), n


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
                else:
                    body, k = _bold_zh(body)
                    n += k

        out.append(body + newline)
        depth = _display_math_depth(body, depth)
        for env in _BEGIN.findall(body):
            envs.append(env)
        for env in _END.findall(body):
            if env in envs:
                envs.remove(env)
    return "".join(out), n
