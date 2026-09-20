"""Verify the failed-job path: status/ok/pages_ok must not masquerade as success.

Runs against the in-process app (no frozen exe needed) with a stub recognizer
that always yields LaTeX the compiler rejects, so every page ends "failed".

Usage: .venv\\Scripts\\python build\\verify_job_failure.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from n2lh.main import create_app

api = "/api"


def sample_png() -> bytes:
    img = Image.new("L", (300, 200), 255)
    ImageDraw.Draw(img).rectangle([20, 20, 280, 40], fill=0)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="n2lh-failjob-"))
    os.environ["N2LH_DATA_DIR"] = str(tmp)
    os.environ["N2LH_ENGINE"] = "heuristic"

    app = create_app()

    # Patch the pipeline so every page produces non-compiling LaTeX.
    from n2lh import jobs as jobs_mod
    from n2lh.pipeline.graph import DocumentPipeline

    real_build = jobs_mod.build_engines

    class BadRecognizer:
        name = "bad"

        def transcribe(self, page, tail=None, open_envs=None, guidance=None):
            from n2lh.recognition.base import TranscribeResult
            return TranscribeResult(latex="\\begin{align*} 1 + ", engine="bad")

    def patched(settings):
        if settings.engine != "heuristic":
            return real_build(settings)
        return BadRecognizer(), BadRecognizer()

    jobs_mod.build_engines = patched

    with TestClient(app) as client:
        r = client.post(api + "/jobs",
                        files=[("files", ("n.png", sample_png(), "image/png"))])
        assert r.status_code == 200, r.text
        job_id = r.json()["id"]
        print("job created:", job_id)

        deadline = time.time() + 120
        job = {}
        while time.time() < deadline:
            job = client.get(f"{api}/jobs/{job_id}").json()
            if job.get("status") in ("done", "error"):
                break
            time.sleep(0.3)

        print("job status:", job.get("status"))
        print("job result:", job.get("result"))
        assert job.get("status") == "done", "job did not finish"
        res = job["result"]
        assert res["ok"] is False, "FAILED job reported ok=true"
        assert res["pages_ok"] == 0, f"pages_ok={res['pages_ok']} for an all-failed job"

        pages = job["pages"]
        assert pages and all(p["status"] == "failed" for p in pages), \
            f"expected all pages failed, got {[p['status'] for p in pages]}"
        print(f"pages: {len(pages)}, all status=failed")
        print("unverified pages kept visible (not silently dropped):",
              all(p["latex"].strip() for p in pages))

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())