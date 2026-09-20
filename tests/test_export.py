"""Getting the finished document out of the app: downloads and Save to folder."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from n2lh.config import Settings
from n2lh.export import safe_stem, save_outputs, unique
from n2lh.main import create_app


def finished_job(tmp_path: Path, client: TestClient, filename="Topology Notes.pdf") -> str:
    """A job record whose output folder holds a document and one figure."""
    store = client.app.state.store
    job = store.create([filename])
    out = store.out_dir(job["id"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "document.tex").write_text("\\documentclass{article}\\begin{document}x\\end{document}",
                                      encoding="utf-8")
    (out / "document.pdf").write_bytes(b"%PDF-1.4 fake")
    (out / "figures").mkdir(exist_ok=True)
    (out / "figures" / "p0001_f1.png").write_bytes(b"\x89PNG fake")
    store.update(job["id"], status="done")
    return job["id"]


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    settings = Settings(engine="heuristic", data_dir=str(tmp_path / "data"))
    return TestClient(create_app(settings=settings, data_dir=tmp_path / "data"))


# ------------------------------------------------------------------ the copy
def test_save_outputs_copies_the_document_and_its_figures(tmp_path):
    out = tmp_path / "out"
    (out / "figures").mkdir(parents=True)
    (out / "document.tex").write_text("x", encoding="utf-8")
    (out / "document.pdf").write_bytes(b"%PDF")
    for n in (1, 2):
        (out / "figures" / f"p0001_f{n}.png").write_bytes(b"png")

    written = save_outputs(out, tmp_path / "target", "My Notes")

    assert (tmp_path / "target" / "My Notes.tex").exists()
    assert (tmp_path / "target" / "My Notes.pdf").read_bytes() == b"%PDF"
    # The .tex says \includegraphics{figures/...}, so that folder has to come along.
    assert (tmp_path / "target" / "figures" / "p0001_f2.png").exists()
    assert written == ["My Notes.tex", "My Notes.pdf", "figures/ (2 images)"]


def test_saving_twice_keeps_the_earlier_copy(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "document.pdf").write_bytes(b"first")
    save_outputs(out, tmp_path / "t", "notes")
    (out / "document.pdf").write_bytes(b"second")
    save_outputs(out, tmp_path / "t", "notes")
    assert (tmp_path / "t" / "notes.pdf").read_bytes() == b"first"
    assert (tmp_path / "t" / "notes (2).pdf").read_bytes() == b"second"


def test_a_missing_target_folder_is_created(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "document.tex").write_text("x", encoding="utf-8")
    save_outputs(out, tmp_path / "new" / "nested", "notes")
    assert (tmp_path / "new" / "nested" / "notes.tex").exists()


def test_a_partial_job_saves_what_exists(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "document.tex").write_text("x", encoding="utf-8")       # compile failed: no pdf
    assert save_outputs(out, tmp_path / "t", "notes") == ["notes.tex"]


@pytest.mark.parametrize("raw,expected", [
    ("Topology Notes.pdf", "Topology Notes"),
    ("week 3/4: charts.pdf", "week 34 charts"),
    ("....pdf", "fallback"),
    ("", "fallback"),
    ("../../secret.pdf", "secret"),
])
def test_the_saved_name_comes_from_the_uploaded_file(raw, expected):
    assert safe_stem(raw, "fallback") == expected


def test_unique_gives_up_rather_than_looping_forever(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert unique(tmp_path / "a.txt").name == "a (2).txt"
    assert unique(tmp_path / "b.txt").name == "b.txt"


# --------------------------------------------------------------- downloads
def test_download_serves_the_document_with_a_useful_name(client, tmp_path):
    job_id = finished_job(tmp_path, client)
    res = client.get(f"/api/jobs/{job_id}/download/pdf")
    assert res.status_code == 200
    assert res.content == b"%PDF-1.4 fake"
    # A .tex served as text/html opens in the window instead of downloading.
    assert res.headers["content-type"] == "application/pdf"
    disposition = res.headers["content-disposition"]
    assert disposition.startswith("attachment") and "Topology%20Notes.pdf" in disposition
    tex = client.get(f"/api/jobs/{job_id}/download/tex")
    assert tex.headers["content-type"] == "application/x-tex"


def test_download_of_an_unfinished_job_is_a_clean_404(client, tmp_path):
    job = client.app.state.store.create(["x.pdf"])
    assert client.get(f"/api/jobs/{job['id']}/download/pdf").status_code == 404
    assert client.get("/api/jobs/nope/download/pdf").status_code == 404
    assert client.get(f"/api/jobs/{job['id']}/download/docx").status_code == 400


# ------------------------------------------------------------ save endpoint
def test_save_endpoint_writes_where_the_user_asked(client, tmp_path):
    job_id = finished_job(tmp_path, client)
    target = tmp_path / "Documents" / "Notes"
    res = client.post(f"/api/jobs/{job_id}/save", json={"dir": str(target)})
    assert res.status_code == 200
    assert (target / "Topology Notes.pdf").exists()
    assert (target / "figures" / "p0001_f1.png").exists()
    assert res.json()["dir"] == str(target)


@pytest.mark.parametrize("payload,code", [
    ({"dir": ""}, 400),
    ({}, 400),
    ({"dir": "relative/path"}, 400),
])
def test_save_endpoint_rejects_a_folder_it_cannot_use(client, tmp_path, payload, code):
    job_id = finished_job(tmp_path, client)
    assert client.post(f"/api/jobs/{job_id}/save", json=payload).status_code == code


def test_save_endpoint_404s_for_an_unknown_or_empty_job(client, tmp_path):
    assert client.post("/api/jobs/nope/save", json={"dir": str(tmp_path)}).status_code == 404
    job = client.app.state.store.create(["x.pdf"])
    res = client.post(f"/api/jobs/{job['id']}/save", json={"dir": str(tmp_path / "t")})
    assert res.status_code == 404


# -------------------------------------------------------- the output_dir setting
def test_output_folder_must_be_a_full_path(client):
    assert client.put("/api/settings", json={"output_dir": "notes"}).status_code == 400


def test_output_folder_is_saved_and_reported(client, tmp_path):
    target = str(tmp_path / "Notes")
    assert client.put("/api/settings", json={"output_dir": target}).status_code == 200
    assert client.get("/api/settings").json()["output_dir"] == target


def test_a_finished_job_is_copied_to_the_output_folder(tmp_path):
    """The whole point of the setting: the PDF lands in the user's own folder."""
    from n2lh.jobs import JobManager
    from n2lh.store import JobStore

    target = tmp_path / "Notes"
    store = JobStore(tmp_path / "data")
    settings = Settings(engine="heuristic", data_dir=str(tmp_path / "data"),
                        output_dir=str(target))
    manager = JobManager(store, settings)
    job_id = finished_job(tmp_path, _FakeClient(store))

    assert manager._auto_save(job_id) == str(target)
    assert (target / "Topology Notes.pdf").exists()
    assert [e["type"] for e in store.read_events(job_id)] == ["saved"]


def test_a_broken_output_folder_is_reported_not_raised(tmp_path):
    from n2lh.jobs import JobManager
    from n2lh.store import JobStore

    store = JobStore(tmp_path / "data")
    blocker = tmp_path / "not-a-folder.txt"
    blocker.write_text("x", encoding="utf-8")
    settings = Settings(engine="heuristic", data_dir=str(tmp_path / "data"),
                        output_dir=str(blocker))
    manager = JobManager(store, settings)
    job_id = finished_job(tmp_path, _FakeClient(store))

    assert manager._auto_save(job_id) is None          # the job still succeeds
    assert [e["type"] for e in store.read_events(job_id)] == ["save_failed"]


def test_no_output_folder_means_no_copying(tmp_path):
    from n2lh.jobs import JobManager
    from n2lh.store import JobStore

    store = JobStore(tmp_path / "data")
    manager = JobManager(store, Settings(engine="heuristic", data_dir=str(tmp_path / "data")))
    job_id = finished_job(tmp_path, _FakeClient(store))
    assert manager._auto_save(job_id) is None
    assert store.read_events(job_id) == []


class _FakeClient:
    """Just enough of the TestClient shape for finished_job()."""

    def __init__(self, store):
        self.app = type("app", (), {"state": type("state", (), {"store": store})})
