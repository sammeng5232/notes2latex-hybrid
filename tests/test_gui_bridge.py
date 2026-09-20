"""The JS bridge the desktop window exposes to the page.

pywebview builds the bridge by walking ``dir()`` of the js_api object and
recursing into every public non-callable attribute (webview/util.py:
get_functions). A public attribute holding the window led it into the WinForms
form and on into .NET COM objects, where it followed
``native.AccessibilityObject.Bounds.Empty.Empty...`` until Python's recursion
limit, logged a megabyte-long error and left the window not responding.
"""

from __future__ import annotations

import inspect

from n2lh.gui import _Bridge


def walk_like_pywebview(obj, base="", found=None, seen=None, depth=0):
    """A faithful copy of pywebview's traversal, used to prove ours terminates."""
    found = {} if found is None else found
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 12:
        return found
    seen.add(id(obj))
    for name in dir(obj):
        if name.startswith("_"):
            continue
        attr = getattr(obj, name)
        full = f"{base}.{name}" if base else name
        if inspect.ismethod(attr) or inspect.isfunction(attr):
            found[full] = "callable"
        elif inspect.isclass(attr) or (
            isinstance(attr, object) and not callable(attr) and hasattr(attr, "__module__")
        ):
            walk_like_pywebview(attr, full, found, seen, depth + 1)
    return found


def test_the_bridge_exposes_exactly_the_two_calls_the_page_makes():
    assert set(walk_like_pywebview(_Bridge())) == {"pick_folder", "reveal"}


def test_the_window_is_private_so_the_bridge_walk_cannot_reach_it():
    """The regression: a public `window` attribute is what pywebview followed."""
    bridge = _Bridge()

    class FakeWindow:                      # stands in for the WinForms window
        def __init__(self):
            self.native = self             # the self-reference that ran away
            self.some_widget = self

    bridge._window = FakeWindow()
    public = [n for n in vars(bridge) if not n.startswith("_")]
    assert public == [], f"public attributes reach the JS bridge: {public}"
    assert set(walk_like_pywebview(bridge)) == {"pick_folder", "reveal"}


def test_pick_folder_returns_empty_when_the_dialog_is_cancelled():
    bridge = _Bridge()

    class Cancelled:
        def create_file_dialog(self, *a, **k):
            return None

    bridge._window = Cancelled()
    assert bridge.pick_folder() == ""


def test_pick_folder_returns_the_first_path_chosen():
    bridge = _Bridge()

    class Chose:
        def create_file_dialog(self, *a, **k):
            return (r"C:\Users\me\Notes",)

    bridge._window = Chose()
    assert bridge.pick_folder() == r"C:\Users\me\Notes"


def test_a_failing_dialog_does_not_propagate_into_the_page():
    bridge = _Bridge()

    class Broken:
        def create_file_dialog(self, *a, **k):
            raise RuntimeError("no window handle")

    bridge._window = Broken()
    assert bridge.pick_folder() == ""


def test_reveal_refuses_a_path_that_does_not_exist(tmp_path):
    assert _Bridge().reveal(str(tmp_path / "nope")) is False
