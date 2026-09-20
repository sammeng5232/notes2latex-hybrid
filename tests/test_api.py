from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from n2lh.config import Settings
from n2lh.main import create_app


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    settings = Settings(engine="heuristic", dpi=150, data_dir=str(tmp_path / "data"))
    app = create_app(settings=settings, data_dir=tmp_path / "data")
    return TestClient(app)


def tiny_note_png() -> bytes:
    img = Image.new("L", (200, 300), 255)
    d = ImageDraw.Draw(img)
    for i in range(6):
        d.rectangle([20, 30 + i * 40, 160, 42 + i * 40], fill=0)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ settings
def test_settings_roundtrip(client: TestClient):
    get = client.get("/api/settings")
    assert get.status_code == 200
    assert get.json()["engine"] == "heuristic"

    put = client.put("/api/settings", json={"dpi": 200, "engine": "hybrid",
                                            "vlm_base_url": "http://localhost:11434/v1",
                                            "vlm_model": "qwen/qwen3-vl"})
    assert put.status_code == 200
    body = put.json()
    assert body["dpi"] == 200
    assert body["engine"] == "hybrid"

    # persisted
    again = client.get("/api/settings")
    assert again.json()["dpi"] == 200


def test_settings_reject_bad_values(client: TestClient):
    res = client.put("/api/settings", json={"engine": "nonsense"})
    assert res.status_code == 400


# ---------------------------------------------------------------------- jobs
def test_upload_rejects_bad_type(client: TestClient):
    res = client.post("/api/jobs", files={"files": ("notes.exe", b"MZ", "application/octet-stream")})
    assert res.status_code == 400


def test_heuristic_job_end_to_end(client: TestClient, tmp_path: Path):
    res = client.post("/api/jobs", files=[
        ("files", ("note1.png", tiny_note_png(), "image/png")),
        ("files", ("note2.png", tiny_note_png(), "image/png")),
    ])
    assert res.status_code == 200
    job_id = res.json()["id"]

    # Wait for the background job (real heuristic engine + real LaTeX compile).
    deadline = time.time() + 120
    job = None
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert job is not None and job["status"] == "done", job
    assert job["result"]["ok"] is True
    assert job["result"]["pages"] == 2
    assert job["result"]["pages_ok"] == 2
    assert all(p["status"] == "ok" for p in job["pages"])
    assert all("region" in p["latex"] for p in job["pages"])

    # Downloads
    tex = client.get(f"/api/jobs/{job_id}/download/tex")
    assert tex.status_code == 200 and "\\documentclass" in tex.text
    pdf = client.get(f"/api/jobs/{job_id}/download/pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"

    # Page image endpoint
    img = client.get(f"/api/jobs/{job_id}/pages/1/image")
    assert img.status_code == 200 and img.content[:8] == b"\x89PNG\r\n\x1a\n"

    # Event log replay contains the full lifecycle
    events = client.get(f"/api/jobs/{job_id}/events").text
    for marker in ("job_start", "ingested", "page_done", "job_done"):
        assert marker in events

    # Strict cleanup inside job workdirs: no aux/log files anywhere under data/jobs
    data_jobs = tmp_path / "data" / "jobs"
    leftovers = [p.name for p in data_jobs.rglob("*")
                 if p.is_file() and p.suffix in (".aux", ".log", ".out", ".synctex.gz")]
    assert leftovers == []


def test_jobs_listing(client: TestClient):
    res = client.post("/api/jobs", files=[
        ("files", ("note.png", tiny_note_png(), "image/png"))])
    assert res.status_code == 200
    listing = client.get("/api/jobs").json()
    assert len(listing) >= 1
    assert listing[0]["id"] == res.json()["id"]


# ------------------------------------------------------- cancel / interruption
def test_cancel_unknown_and_finished_jobs(client: TestClient):
    assert client.post("/api/jobs/does-not-exist/cancel").status_code == 404

    res = client.post("/api/jobs", files=[("files", ("n.png", tiny_note_png(), "image/png"))])
    job_id = res.json()["id"]
    deadline = time.time() + 120
    while time.time() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    # Nothing left to cancel once it has finished.
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409


def test_orphaned_running_jobs_are_marked_interrupted_on_startup(tmp_path: Path):
    """A job left at 'running' by an app that was closed used to stay 'running'
    forever in the Jobs list. The next start must flip it to error."""
    import json
    import os
    from n2lh.store import JobStore

    data = tmp_path / "data"
    store = JobStore(data)
    stale = store.create(["old.pdf"])
    store.update(stale["id"], status="running")
    store.append_event(stale["id"], {"type": "job_start"})
    events = data / "jobs" / stale["id"] / "events.jsonl"
    old = time.time() - 3600
    os.utime(events, (old, old))                       # last activity an hour ago

    fresh = store.create(["live.pdf"])                 # e.g. another instance, just active
    store.update(fresh["id"], status="running")
    store.append_event(fresh["id"], {"type": "job_start"})

    app = create_app(settings=Settings(engine="heuristic", data_dir=str(data)), data_dir=data)
    c = TestClient(app)

    stale_job = c.get(f"/api/jobs/{stale['id']}").json()
    assert stale_job["status"] == "error" and "interrupted" in stale_job["error"]
    assert "job_error" in c.get(f"/api/jobs/{stale['id']}/events").text
    assert c.get(f"/api/jobs/{fresh['id']}").json()["status"] == "running"   # left alone