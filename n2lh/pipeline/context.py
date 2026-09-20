"""Cross-page context: rolling LaTeX window + open-environment tracking.

Mirrors the sequential-page design of notes2latex: each page sees the last N
lines of generated LaTeX plus any unclosed environments (e.g. a dangling
align*) so notation and structure stay consistent across the document.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Deque, List, Optional, Tuple

_BEGIN = re.compile(r"\\begin\{([A-Za-z*]+)\}")
_END = re.compile(r"\\end\{([A-Za-z*]+)\}")
# Environments that legitimately pair non-symmetrically or nest implicitly.
_NEVER_TRACK = {"document", "verbatim", "verbatim*", "comment"}


class EnvironmentTracker:
    """Maintains a stack of currently-open environments from generated LaTeX."""

    def __init__(self) -> None:
        self._stack: List[str] = []

    def update(self, latex: str) -> None:
        tokens: List[Tuple[str, str]] = []
        for m in _BEGIN.finditer(latex):
            tokens.append((m.start(), "b", m.group(1)))
        for m in _END.finditer(latex):
            tokens.append((m.start(), "e", m.group(1)))
        for _, kind, env in sorted(tokens):
            if env in _NEVER_TRACK:
                continue
            if kind == "b":
                self._stack.append(env)
            else:
                # Close the most recent matching env (ignore unbalanced \end).
                for i in range(len(self._stack) - 1, -1, -1):
                    if self._stack[i] == env:
                        del self._stack[i:]
                        break

    @property
    def open_environments(self) -> List[str]:
        return list(self._stack)

    def close_for_compile(self, body: str) -> Tuple[str, int]:
        """Return (body + missing \\end{} closers, number appended).

        The accumulated body is kept raw; closers are only appended to the
        throw-away copy handed to the compiler so a page split mid-environment
        still compiles.
        """
        closers = [f"\\end{{{env}}}" for env in reversed(self._stack)]
        if not closers:
            return body, 0
        return body + "\n" + "\n".join(closers), len(closers)


class ContextWindow:
    """Rolling window of the last N lines of generated LaTeX."""

    def __init__(self, max_lines: int = 40) -> None:
        self.max_lines = max_lines
        self._lines: Deque[str] = deque(maxlen=max_lines)

    def push(self, latex: str) -> None:
        for line in latex.splitlines():
            line = line.rstrip()
            if line:
                self._lines.append(line)

    def tail(self, n: Optional[int] = None) -> str:
        if n is None:
            return "\n".join(self._lines)
        take = list(self._lines)[-n:]
        return "\n".join(take)

    def __len__(self) -> int:
        return len(self._lines)
