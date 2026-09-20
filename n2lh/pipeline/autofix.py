"""Deterministic repairs for the mechanical LaTeX faults models keep making.

Only ever applied to a page that has ALREADY failed to compile, so a page that
builds is never touched. Three faults, all seen repeatedly on real notes (a bare
``&`` was 16-18 of the 22-25 first-attempt failures in each 43-page run, and one
page bounced between ``&``, ``aligned`` outside math mode and an unclosed
``proof`` through every model repair):

* a bare ``&`` outside any alignment environment  -> ``\\&``
* ``\\begin{X}`` never closed / a stray or crossed ``\\end{X}``  -> balanced
* ``aligned`` / ``cases`` / matrices / ``array`` used in text mode  -> wrapped in ``\\[ ... \\]``
* an ``array`` / ``tabular`` whose column spec is narrower than its widest row
  ("Extra alignment tab has been changed to \\cr")  -> spec widened
* a bare ``#``  -> ``\\#``
* raw Unicode math characters (``∤ ∀ ∈ ℝ α ...``), which pdflatex rejects
  ("Unicode character ... not set up for use with LaTeX")  -> the macro

The scan is a single left-to-right pass over LaTeX tokens with an environment
stack and a small math-mode state. It is deliberately conservative: it only
rewrites constructs that are illegal as written.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Unicode characters pdflatex cannot typeset -> their macro. Names follow
# unicode-math (so U+03C6 is \varphi and U+03D5 is \phi).
_UNICODE: Dict[str, str] = {
    "∀": r"\forall", "∃": r"\exists", "∄": r"\nexists", "∈": r"\in", "∉": r"\notin",
    "∋": r"\ni", "⊂": r"\subset", "⊆": r"\subseteq", "⊃": r"\supset", "⊇": r"\supseteq",
    "⊄": r"\not\subset", "⊈": r"\nsubseteq", "⊊": r"\subsetneq", "∪": r"\cup",
    "∩": r"\cap", "⋃": r"\bigcup", "⋂": r"\bigcap", "∅": r"\emptyset", "∖": r"\setminus",
    "≤": r"\le", "≥": r"\ge", "≠": r"\ne", "≈": r"\approx", "≅": r"\cong",
    "≡": r"\equiv", "∼": r"\sim", "≃": r"\simeq", "≪": r"\ll", "≫": r"\gg",
    "∝": r"\propto", "∣": r"\mid", "∤": r"\nmid", "∥": r"\parallel", "⊥": r"\perp",
    "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow", "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow", "⟹": r"\Longrightarrow",
    "⟸": r"\Longleftarrow", "⟺": r"\Longleftrightarrow", "⟶": r"\longrightarrow",
    "↦": r"\mapsto", "↪": r"\hookrightarrow", "↠": r"\twoheadrightarrow",
    "↑": r"\uparrow", "↓": r"\downarrow", "∞": r"\infty", "∂": r"\partial",
    "∇": r"\nabla", "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∮": r"\oint",
    "∘": r"\circ", "∙": r"\bullet", "⋅": r"\cdot", "⋯": r"\cdots", "∓": r"\mp",
    "⊕": r"\oplus", "⊗": r"\otimes", "⊙": r"\odot", "∧": r"\wedge", "∨": r"\vee",
    "¬": r"\neg", "⟨": r"\langle", "⟩": r"\rangle", "⌊": r"\lfloor", "⌋": r"\rfloor",
    "⌈": r"\lceil", "⌉": r"\rceil", "‖": r"\|", "⊢": r"\vdash", "⊨": r"\models",
    "∠": r"\angle", "△": r"\triangle", "□": r"\square", "∎": r"\blacksquare",
    "∴": r"\therefore", "∵": r"\because", "ℓ": r"\ell", "ℏ": r"\hbar",
    "ℝ": r"\mathbb{R}", "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}", "ℕ": r"\mathbb{N}",
    "ℂ": r"\mathbb{C}", "ℙ": r"\mathbb{P}", "ℍ": r"\mathbb{H}",
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\varepsilon",
    "ϵ": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta", "ϑ": r"\vartheta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu",
    "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau",
    "υ": r"\upsilon", "φ": r"\varphi", "ϕ": r"\phi", "χ": r"\chi", "ψ": r"\psi",
    "ω": r"\omega", "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon", "Φ": r"\Phi",
    "Ψ": r"\Psi", "Ω": r"\Omega",
}
# Latin-1 symbols pdflatex accepts in running text but not inside math.
_LATIN1_MATH: Dict[str, str] = {"×": r"\times", "·": r"\cdot", "±": r"\pm", "−": "-"}
_UNI_RE = re.compile("[" + re.escape("".join(_UNICODE) + "".join(_LATIN1_MATH)) + "]")


def _translate(segment: str, in_math: bool) -> Tuple[str, int]:
    """Replace unsupported Unicode characters in a stretch of plain text."""
    count = 0

    def sub(m: "re.Match[str]") -> str:
        nonlocal count
        ch = m.group(0)
        if ch in _LATIN1_MATH:
            if not in_math:
                return ch                       # fine in text mode
            count += 1
            macro = _LATIN1_MATH[ch]
            return macro if macro == "-" else macro + " "
        count += 1
        macro = _UNICODE[ch]
        return macro + " " if in_math else f"${macro}$"

    return _UNI_RE.sub(sub, segment), count

# Environments in which a bare & is a legal alignment tab (tikzpicture: \matrix).
_ALIGN = {
    "array", "tabular", "tabular*", "tabularx", "longtable", "align", "align*",
    "alignat", "alignat*", "aligned", "alignedat", "flalign", "flalign*", "xalignat",
    "xxalignat", "split", "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix",
    "Vmatrix", "smallmatrix", "cases", "dcases", "eqnarray", "eqnarray*", "tikzcd",
    "IEEEeqnarray", "tikzpicture",
}
# Environments that are only legal inside math mode.
_NEEDS_MATH = {
    "aligned", "alignedat", "gathered", "split", "cases", "dcases", "array", "matrix",
    "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix",
}
# Environments that ARE math mode.
_MATH_ENVS = {
    "equation", "equation*", "align", "align*", "alignat", "alignat*", "gather",
    "gather*", "multline", "multline*", "flalign", "flalign*", "eqnarray", "eqnarray*",
    "displaymath", "math", "IEEEeqnarray",
}

_TOKEN = re.compile(
    r"(?P<comment>%[^\n]*)"
    r"|(?P<dbs>\\\\)"                               # \\ (so that \\[2pt] is not \[ )
    r"|(?P<esc>\\[&%$#_{}^~])"                      # escaped specials
    r"|(?P<begin>\\begin\{(?P<bname>[A-Za-z*]+)\})"
    r"|(?P<end>\\end\{(?P<ename>[A-Za-z*]+)\})"
    r"|(?P<dopen>\\\[)|(?P<dclose>\\\])"
    r"|(?P<popen>\\\()|(?P<pclose>\\\))"
    r"|(?P<dd>\$\$)|(?P<d>\$)"
    r"|(?P<amp>&)"
    r"|(?P<hash>#)"
    r"|(?P<sup>[\^_])"
)
# A sub/superscript in text mode ("L^p completeness" inside \textbf) is
# "Missing $ inserted". The operand on either side is a letter, a digit, a
# braced group or a control sequence, so the whole thing can be put into math.
_OPERAND_BEFORE = re.compile(r"(?:\\[A-Za-z]+|\{[^{}]*\}|[A-Za-z0-9])$")
_OPERAND_AFTER = re.compile(r"(?:\\[A-Za-z]+|\{[^{}]*\}|[A-Za-z0-9])")
# `#` is only legal in macro definitions, which a page transcription never has.
_DEFINES_MACROS = re.compile(r"\\(?:re)?newcommand|\\providecommand|\\def\b|\\newenvironment|\\renewenvironment")

# --- column specs -----------------------------------------------------------
_TABLE_BEGIN = re.compile(r"\\begin\{(?P<name>array|tabular)\}\s*(?:\[[^\]\n]*\])?\s*\{")
_CELL_TOKEN = re.compile(
    r"(?P<comment>%[^\n]*)|(?P<esc>\\[&%$#_{}])|(?P<dbs>\\\\)"
    r"|\\begin\{(?P<b>[A-Za-z*]+)\}|\\end\{(?P<e>[A-Za-z*]+)\}"
    r"|(?P<amp>&)|(?P<multi>\\multi(?:column|row)\b)")


def _skip_group(s: str, i: int) -> Optional[int]:
    """``s[i]`` is ``{``: return the index just after its matching ``}``."""
    if i >= len(s) or s[i] != "{":
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return None


def _spec_columns(spec: str) -> Optional[int]:
    """Number of columns a tabular/array spec declares; None if it is too exotic to count."""
    i = n = 0
    while i < len(spec):
        ch = spec[i]
        if ch in "lcrX":
            n += 1
            i += 1
        elif ch in "pmb":
            j = _skip_group(spec, i + 1)
            if j is None:
                return None
            n += 1
            i = j
        elif ch in "@!><":
            j = _skip_group(spec, i + 1)
            if j is None:
                return None
            i = j
        elif ch == "|" or ch.isspace():
            i += 1
        else:                                   # *{n}{..}, d{..}, S, ...
            return None
    return n


def _max_cells(body: str) -> Optional[int]:
    """Widest row of an alignment body, ignoring nested alignment environments."""
    depth = cur = best = 0
    for m in _CELL_TOKEN.finditer(body):
        if m.group("multi"):
            return None                         # spans change the count: leave alone
        if m.group("b") is not None:
            depth += m.group("b") in _ALIGN
        elif m.group("e") is not None:
            depth = max(0, depth - (m.group("e") in _ALIGN))
        elif depth == 0 and m.group("amp"):
            cur += 1
        elif depth == 0 and m.group("dbs"):
            best = max(best, cur + 1)
            cur = 0
    return max(best, cur + 1)


def _widen_columns(latex: str) -> Tuple[str, int]:
    """Pad the column spec of every array/tabular that has a row wider than its spec."""
    edits: List[Tuple[int, int, str]] = []       # (start, end, replacement) of the spec group
    for m in _TABLE_BEGIN.finditer(latex):
        name = m.group("name")
        spec_start = m.end() - 1
        spec_end = _skip_group(latex, spec_start)
        if spec_end is None:
            continue
        depth = 1
        body_end = None
        for t in re.finditer(r"\\(begin|end)\{" + name + r"\}", latex[spec_end:]):
            depth += 1 if t.group(1) == "begin" else -1
            if depth == 0:
                body_end = spec_end + t.start()
                break
        if body_end is None:
            continue
        cols = _spec_columns(latex[spec_start + 1:spec_end - 1])
        need = _max_cells(latex[spec_end:body_end])
        if cols is None or need is None or need <= cols:
            continue
        edits.append((spec_start + 1, spec_end - 1,
                      latex[spec_start + 1:spec_end - 1] + "c" * (need - cols)))
    for start, end, repl in reversed(edits):
        latex = latex[:start] + repl + latex[end:]
    return latex, len(edits)


class _State:
    def __init__(self) -> None:
        self.stack: List[Tuple[str, bool]] = []      # (env name, we wrapped it in \[ \])
        self.dollar = False
        self.dd = False
        self.dopen = False
        self.popen = False

    def in_math(self) -> bool:
        return (self.dollar or self.dd or self.dopen or self.popen
                or any(name in _MATH_ENVS or wrapped for name, wrapped in self.stack))

    def in_align(self) -> bool:
        return any(name in _ALIGN for name, _ in self.stack)


def autofix_latex(latex: str) -> Tuple[str, List[str]]:
    """Return ``(fixed_latex, changes)``; ``changes`` is empty when nothing was wrong."""
    original = latex
    latex, n_cols = _widen_columns(latex)
    escape_hash = not _DEFINES_MACROS.search(latex)
    st = _State()
    out: List[str] = []
    pos = 0
    n_amp = n_close = n_stray = n_wrap = n_hash = n_uni = n_sup = 0

    def closer(name: str, wrapped: bool) -> str:
        return f"\\end{{{name}}}" + ("\n\\]" if wrapped else "")

    def plain(segment: str) -> str:
        nonlocal n_uni
        translated, n = _translate(segment, st.in_math())
        n_uni += n
        return translated

    for m in _TOKEN.finditer(latex):
        out.append(plain(latex[pos:m.start()]))
        pos = m.end()
        token = m.group(0)
        kind = m.lastgroup
        # lastgroup is the innermost named group that matched; map sub-groups back.
        if m.group("begin"):
            kind = "begin"
        elif m.group("end"):
            kind = "end"

        if kind == "begin":
            name = m.group("bname")
            if name in _NEEDS_MATH and not st.in_math():
                out.append("\\[\n")
                st.stack.append((name, True))
                n_wrap += 1
            else:
                st.stack.append((name, False))
            out.append(token)
        elif kind == "end":
            name = m.group("ename")
            idx = next((i for i in range(len(st.stack) - 1, -1, -1)
                        if st.stack[i][0] == name), None)
            if idx is None:
                n_stray += 1                     # \end with no matching \begin: drop it
                continue
            for cname, cwrapped in reversed(st.stack[idx + 1:]):
                out.append(closer(cname, cwrapped) + "\n")
                n_close += 1
            _, wrapped = st.stack[idx]
            del st.stack[idx:]
            out.append(closer(name, wrapped))
        elif kind == "amp":
            if st.in_align():
                out.append(token)
            else:
                out.append("\\&")
                n_amp += 1
        elif kind == "hash":
            if escape_hash:
                out.append("\\#")
                n_hash += 1
            else:
                out.append(token)
        elif kind == "sup":
            if st.in_math():
                out.append(token)
            else:
                # Pull the operand that is already written back out, take the one
                # that follows, and set the three of them as maths.
                tail = "".join(out)
                before = _OPERAND_BEFORE.search(tail)
                head = before.group(0) if before else ""
                if before:
                    out = [tail[:before.start()]]
                after = _OPERAND_AFTER.match(latex, pos)
                foot = ""
                if after:
                    foot = after.group(0)
                    pos = after.end()
                out.append(f"${head}{token}{foot}$")
                n_sup += 1
        elif kind == "d":
            if not st.dd:
                st.dollar = not st.dollar
            out.append(token)
        elif kind == "dd":
            if not st.dollar:
                st.dd = not st.dd
            out.append(token)
        elif kind == "dopen":
            st.dopen = True
            out.append(token)
        elif kind == "dclose":
            st.dopen = False
            out.append(token)
        elif kind == "popen":
            st.popen = True
            out.append(token)
        elif kind == "pclose":
            st.popen = False
            out.append(token)
        else:                                    # comment, \\, escaped special
            out.append(token)

    out.append(plain(latex[pos:]))
    for cname, cwrapped in reversed(st.stack):   # close whatever was left open
        out.append("\n" + closer(cname, cwrapped))
        n_close += 1

    changes: List[str] = []
    if n_amp:
        changes.append(f"escaped {n_amp} bare '&'")
    if n_hash:
        changes.append(f"escaped {n_hash} bare '#'")
    if n_sup:
        changes.append(f"put {n_sup} text-mode sub/superscript(s) into math")
    if n_close:
        changes.append(f"closed {n_close} unclosed environment(s)")
    if n_stray:
        changes.append(f"removed {n_stray} stray \\end")
    if n_wrap:
        changes.append(f"wrapped {n_wrap} math-only block(s) in \\[ \\]")
    if n_cols:
        changes.append(f"widened {n_cols} array/tabular column spec(s)")
    if n_uni:
        changes.append(f"replaced {n_uni} unsupported Unicode symbol(s)")
    return ("".join(out) if changes else original), changes
