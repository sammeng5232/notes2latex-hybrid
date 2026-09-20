from __future__ import annotations

from n2lh.pipeline.context import ContextWindow, EnvironmentTracker


def test_environment_tracker_balanced():
    t = EnvironmentTracker()
    t.update("\\begin{align*} x &= 1 \\\\ \\end{align*}\nhello")
    assert t.open_environments == []


def test_environment_tracker_dangling():
    t = EnvironmentTracker()
    t.update("\\begin{align*}\nx &= 1 \\\\")
    assert t.open_environments == ["align*"]
    closed, n = t.close_for_compile("\\begin{align*}\nx &= 1 \\\\")
    assert n == 1
    assert closed.endswith("\\end{align*}")


def test_environment_tracker_nested_and_document_ignored():
    t = EnvironmentTracker()
    t.update("\\begin{document}\n\\begin{itemize}\n\\item a\n\\begin{align*} 1+1")
    # document is never tracked
    assert t.open_environments == ["itemize", "align*"]
    t.update("\\end{align*}")
    assert t.open_environments == ["itemize"]


def test_environment_tracker_interleaved_close():
    t = EnvironmentTracker()
    t.update("\\begin{itemize}\n\\begin{center}\n\\end{itemize}\nbody")
    # \end{itemize} closes itemize (and implicitly the nested center)
    assert t.open_environments == []


def test_context_window_tail():
    w = ContextWindow(max_lines=3)
    w.push("line-a\nline-b")
    w.push("line-c\nline-d")
    assert w.tail().splitlines() == ["line-b", "line-c", "line-d"]
    assert len(w) == 3
