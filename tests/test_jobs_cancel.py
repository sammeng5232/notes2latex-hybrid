"""JobManager.cancel: a running job stops, keeps its partial output, and reports it."""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path

from PIL import Image

from n2lh import jobs as jobs_mod
from n2lh.config import Settings
from n2lh.jobs import JobManager
from n2lh.recognition.base import Recognizer, TranscribeResult
from n2lh.store import JobStore


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("L", (80, 80), 255).save(buf, "PNG")
    return buf.getvalue()


class SlowEngine(Recognizer):
    """Answers page 1 at once, then blocks until cancelled (like a hung request
    that the VLM watchdog aborts on cancel)."""
    name = "slow"

    def __init__(self):
        self.cancel_event = None
        self.closed = False

    def transcribe(self, page, context_tail, open_environments, guidance=None):
        if page.index >= 2:
            assert self.cancel_event.wait(30), "cancel never reached the engine"
            raise RuntimeError("cancelled")
        return TranscribeResult(latex="\\text{page one}", engine=self.name)

    def close(self):
        self.closed = True


def test_cancel_stops_a_running_job_and_keeps_partial_output(tmp_path: Path, monkeypatch):
    engine = SlowEngine()
    monkeypatch.setattr(jobs_mod, "build_engines",
                        lambda settings, doc_hint="": (engine, None))
    store = JobStore(tmp_path / "data")
    mgr = JobManager(store, Settings(engine="heuristic", data_dir=str(tmp_path / "data")))

    src = tmp_path / "in.png"
    src.write_bytes(png_bytes())
    # Two identical page images -> pages 1 and 2.
    src2 = tmp_path / "in2.png"
    src2.write_bytes(png_bytes())
    job = mgr.start_job([src, src2], ["a.png", "b.png"])
    job_id = job["id"]

    deadline = time.time() + 30
    while time.time() < deadline and not any(
            e["type"] == "page_done" for e in store.read_events(job_id)):
        time.sleep(0.1)
    assert mgr.cancel(job_id) is True

    deadline = time.time() + 60
    while time.time() < deadline and store.get(job_id)["status"] in ("pending", "running"):
        time.sleep(0.2)
    final = store.get(job_id)
    assert final["status"] == "cancelled"
    assert final["result"]["aborted"] == "cancelled by user"
    assert final["result"]["pdf"]                       # partial document still built
    kinds = [e["type"] for e in store.read_events(job_id)]
    assert "cancel_requested" in kinds and "job_done" in kinds
    assert engine.closed                                # client released at job end
    assert mgr.cancel(job_id) is False                  # already finished