"""Desktop GUI shell: local server + native window (or tray/browser fallback).

Used by the frozen exe. The window hosts the existing web UI; closing the
window quits the app. Fallbacks (no WebView2 / missing deps): browser tab +
system-tray icon with Quit.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

log = logging.getLogger("n2lh.gui")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def frozen_paths() -> "tuple[Path, Path]":
    """(data_dir, logs_dir). Frozen apps live under %LOCALAPPDATA%."""
    if getattr(sys, "frozen", False):
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "notes2latex-hybrid"
    else:
        base = Path(".n2lh-dev")
    data = base / "data"
    logs = base / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return data, logs


def run() -> None:
    data_dir, logs_dir = frozen_paths()
    logging.basicConfig(
        filename=str(logs_dir / "app.log"), level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    port = int(os.environ.get("N2LH_PORT") or _free_port())
    url = f"http://127.0.0.1:{port}"

    from n2lh.config import load_settings
    from n2lh.main import create_app

    settings = load_settings(data_dir)
    app = create_app(settings=settings, data_dir=data_dir)

    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=port,
                            log_level="info", log_config=None)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="n2lh-server")
    thread.start()
    if not _wait_up(url, timeout=30):
        log.error("server did not come up on %s", url)
        os._exit(1)
    log.info("server ready at %s (data=%s)", url, data_dir)

    if os.environ.get("N2LH_GUI_NO_WINDOW"):
        # Headless mode for automation/testing: print URL, serve forever.
        sys.stdout.write(url + "\n")
        sys.stdout.flush()
        thread.join()
        return

    if _open_webview(url):
        log.info("window closed; exiting")
        os._exit(0)

    log.warning("webview unavailable; falling back to browser + tray")
    webbrowser.open(url)
    _run_tray(url)  # blocks until Quit; final fallback blocks on the server


def _wait_up(url: str, timeout: float) -> bool:
    import http.client
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=1)
            conn.request("GET", "/api/settings")
            if conn.getresponse().status == 200:
                return True
        except OSError:
            pass
        time.sleep(0.2)
    return False


class _Bridge:
    """Exposed to the page as ``window.pywebview.api``.

    The window is a browser without a browser's chrome: there is no Downloads
    bar and no "Save as" unless the app provides one, so the page asks for a
    real folder picker through here.
    """

    def __init__(self) -> None:
        self.window = None

    def pick_folder(self, start: str = "") -> str:
        """Native folder chooser; returns "" when the user cancels."""
        import webview

        try:
            chosen = self.window.create_file_dialog(
                webview.FileDialog.FOLDER, directory=start or "")
        except Exception:
            log.exception("folder dialog failed")
            return ""
        if not chosen:
            return ""
        return str(chosen[0]) if isinstance(chosen, (list, tuple)) else str(chosen)

    def reveal(self, path: str) -> bool:
        """Open a saved folder in the system file manager."""
        target = Path(path)
        if not target.exists():
            return False
        try:
            if sys.platform == "win32":
                os.startfile(str(target))  # noqa: S606 - a folder the user just chose
            else:
                import subprocess
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open",
                                  str(target)])
            return True
        except Exception:
            log.exception("could not open %s", target)
            return False


def _open_webview(url: str) -> bool:
    try:
        import webview
    except Exception:
        log.exception("pywebview import failed")
        return False
    try:
        # Without this every download the page starts is cancelled silently, so
        # the Download .tex / .pdf buttons do nothing at all in the window.
        webview.settings["ALLOW_DOWNLOADS"] = True
        bridge = _Bridge()
        bridge.window = webview.create_window(
            "notes2latex-hybrid", url, width=1280, height=860, min_size=(900, 600),
            js_api=bridge)
        webview.start()
        return True
    except Exception:
        log.exception("webview window failed")
        return False


def _run_tray(url: str) -> None:
    try:
        import pystray
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (64, 64), (16, 20, 24, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([4, 4, 59, 59], outline=(76, 166, 255, 255), width=4)
        d.text((12, 22), "TeX", fill=(230, 237, 243, 255))

        menu = pystray.Menu(
            pystray.MenuItem("Open notes2latex-hybrid",
                             lambda *_: webbrowser.open(url), default=True),
            pystray.MenuItem("Quit", lambda *_: os._exit(0)),
        )
        pystray.Icon("n2lh", img, "notes2latex-hybrid", menu).run()
    except Exception:
        log.exception("tray unavailable; idling until killed")
        threading.Event().wait()
