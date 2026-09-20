"""Headless verification of the frozen exe: boot, upload, convert, download, cleanup.

Usage: .venv\\Scripts\\python verify_exe.py [path-to-exe]
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

PORT = int(os.environ.get("N2LH_VERIFY_PORT", "8757"))
BASE = f"http://127.0.0.1:{PORT}"


def sample_png() -> bytes:
    img = Image.new("L", (400, 500), 255)
    d = ImageDraw.Draw(img)
    for i in range(5):
        d.rectangle([30, 30 + i * 90, 350, 52 + i * 90], fill=0)
    d.rectangle([180, 180, 188, 330], fill=0)  # tall glyph -> math band
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def wait_up(client: httpx.Client, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = client.get(BASE + "/api/preflight", timeout=2)
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    raise SystemExit("FAIL: server did not come up")


def wait_job(client: httpx.Client, job_id: str, timeout: float = 420.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"{BASE}/api/jobs/{job_id}", timeout=5).json()
        if job.get("status") in ("done", "error"):
            return job
        time.sleep(0.5)
    raise SystemExit("FAIL: job did not finish")


def main() -> int:
    exe = Path(sys.argv[1] if len(sys.argv) > 1 else "dist/notes2latex-hybrid.exe")
    if not exe.exists():
        raise SystemExit(f"exe not found: {exe}")

    # Acceptance test of the FROZEN APP, not of any external model: force the
    # offline heuristic engine via env override (settings.json is untouched).
    # A separate one-off VLM smoke test covers the live model path.
    env = dict(os.environ, N2LH_GUI_NO_WINDOW="1", N2LH_PORT=str(PORT),
               N2LH_ENGINE="heuristic")
    proc = subprocess.Popen([str(exe)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ok = False
    try:
        with httpx.Client(timeout=30) as client:
            pre = wait_up(client)
            print(f"server up: version={pre['version']} latex_toolchain={pre['latex_toolchain']}")
            assert pre["latex_toolchain"], "LaTeX toolchain not detected"

            html = client.get(BASE + "/").text
            assert "notes2latex-hybrid" in html, "web UI not bundled"
            print("web UI bundled: yes")

            files = [("files", ("note.png", sample_png(), "image/png")),
                     ("files", ("note2.png", sample_png(), "image/png"))]
            r = client.post(BASE + "/api/jobs", files=files)
            assert r.status_code == 200, r.text
            job_id = r.json()["id"]
            print(f"job created: {job_id}")

            job = wait_job(client, job_id)
            print(f"job status: {job['status']} ok={job['result']['ok']} "
                  f"pages={job['result']['pages_ok']}/{job['result']['pages']}")
            assert job["status"] == "done" and job["result"]["ok"] is True
            assert job["result"]["pages"] == 2 and job["result"]["pages_ok"] == 2
            assert all(p["status"] == "ok" for p in job["pages"])

            pdf = client.get(f"{BASE}/api/jobs/{job_id}/download/pdf").content
            assert pdf[:4] == b"%PDF", "downloaded file is not a PDF"
            print(f"pdf download: {len(pdf)} bytes, valid header")

            tex = client.get(f"{BASE}/api/jobs/{job_id}/download/tex").text
            assert "\\documentclass" in tex
            print("tex download: valid document")

            events = client.get(f"{BASE}/api/jobs/{job_id}/events").text
            for marker in ("job_start", "ingested", "page_done", "job_done"):
                assert marker in events, f"missing event {marker}"
            print("event stream: complete lifecycle present")

            img = client.get(f"{BASE}/api/jobs/{job_id}/pages/1/image").content
            assert img[:8] == b"\x89PNG\r\n\x1a\n"
            print("page images: served")

        # Strict aux-cleanup audit under the frozen data dir (not logs/ —
        # the GUI's own app.log there is intentional diagnostics).
        data_root = Path(os.environ["LOCALAPPDATA"]) / "notes2latex-hybrid" / "data"
        leftovers = [str(p) for p in data_root.rglob("*")
                     if p.is_file() and p.suffix in (".aux", ".log", ".out",
                                                     ".fls", ".fdb_latexmk", ".synctex.gz")]
        assert not leftovers, f"aux leftovers: {leftovers}"
        print(f"aux cleanup under {data_root}: clean")
        ok = True
    finally:
        proc.kill()
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
