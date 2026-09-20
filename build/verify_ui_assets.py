"""Verify the frozen exe serves every static UI asset and the settings UI state.

This is a focused check of the static-asset path and of the "is my API key saved?"
indicator, independent of any external model.

Usage: .venv\\Scripts\\python build\\verify_ui_assets.py [path-to-exe]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

PORT = int(os.environ.get("N2LH_UI_PORT", "8781"))
BASE = f"http://127.0.0.1:{PORT}"

ASSETS = {
    "/": ("text/html", "notes2latex-hybrid"),
    "/app.js": ("javascript", "key-status"),
    "/style.css": ("text/css", "engine-summary"),
    "/favicon.ico": ("image/", None),
}


def main() -> int:
    exe = Path(sys.argv[1] if len(sys.argv) > 1 else "dist/notes2latex-hybrid.exe")
    if not exe.exists():
        raise SystemExit(f"exe not found: {exe}")

    env = dict(os.environ, N2LH_GUI_NO_WINDOW="1", N2LH_PORT=str(PORT),
               N2LH_ENGINE="heuristic")
    proc = subprocess.Popen([str(exe)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ok = False
    try:
        with httpx.Client(timeout=20) as client:
            deadline = time.time() + 60
            while time.time() < deadline:
                try:
                    if client.get(BASE + "/api/preflight", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.3)
            else:
                raise SystemExit("FAIL: server did not come up")

            for path, (ctype, needle) in ASSETS.items():
                r = client.get(BASE + path)
                body = r.content
                print(f"{path:14} {r.status_code} {len(body):>7} bytes "
                      f"{r.headers.get('content-type','')}")
                assert r.status_code == 200, f"{path} -> {r.status_code}"
                assert body, f"{path} served empty body"
                if needle:
                    assert needle in body.decode("utf-8", "replace"), \
                        f"{path} missing marker {needle!r}"

            s = client.get(BASE + "/api/settings").json()
            print("settings: engine=%s model=%s key_set=%s"
                  % (s.get("engine"), s.get("vlm_model"), s.get("vlm_api_key_set")))
            assert s.get("vlm_api_key_set") is True, "saved API key not reported as set"
            assert "vlm_api_key" not in s or s["vlm_api_key"] in ("", None) \
                or "..." in s["vlm_api_key"], "full key leaked to client"
            print("settings API: key present + masked")

        ok = True
    finally:
        proc.kill()
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())