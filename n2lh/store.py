"""File-backed job store: data/jobs/<id>/{job.json, events.jsonl, pages/, out/}.

No server DB required; everything (pages, outputs, progress log) lives under
the data directory so results survive restarts and downloads are trivial.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional


class JobStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir)
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ jobs
    def create(self, filenames: List[str]) -> dict:
        job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        job_dir = self.jobs_dir / job_id
        (job_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (job_dir / "pages").mkdir(exist_ok=True)
        (job_dir / "out").mkdir(exist_ok=True)
        job = {
            "id": job_id,
            "status": "pending",       # pending | running | done | error
            "created_at": time.time(),
            "updated_at": time.time(),
            "filenames": filenames,
            "pages": [],               # [{index,image,latex,status,attempts,engine}]
            "result": None,            # {ok, pages_ok, pages, pdf, tex}
            "error": None,
        }
        self._write_job(job)
        return job

    def uploads_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "uploads"

    def get(self, job_id: str) -> Optional[dict]:
        path = self.jobs_dir / job_id / "job.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return None

    def list_jobs(self, limit: int = 50) -> List[dict]:
        jobs = []
        for job_dir in sorted(self.jobs_dir.iterdir(), reverse=True):
            if not job_dir.is_dir():
                continue
            job = self.get(job_dir.name)
            if job:
                jobs.append(job)
            if len(jobs) >= limit:
                break
        return jobs

    def update(self, job_id: str, **values) -> None:
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return
            job.update(values)
            job["updated_at"] = time.time()
            self._write_job(job)

    def set_pages(self, job_id: str, pages: List[dict]) -> None:
        self.update(job_id, pages=pages)

    def mark_interrupted(self, grace_seconds: float = 120.0) -> int:
        """Flip jobs stuck at pending/running to ``error``.

        Called once at startup: this process has no job threads yet, so a job
        still marked running was orphaned by an app that was closed or crashed
        (they used to sit at "running" in the Jobs list forever). A job whose
        event log was written within ``grace_seconds`` is left alone in case
        another app instance sharing this data dir is genuinely running it.
        Returns how many jobs were changed.
        """
        changed = 0
        now = time.time()
        for job_dir in self.jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue
            job = self.get(job_dir.name)
            if not job or job.get("status") not in ("pending", "running"):
                continue
            events = job_dir / "events.jsonl"
            last_activity = events.stat().st_mtime if events.exists() else job.get("updated_at", 0)
            if now - last_activity < grace_seconds:
                continue
            message = "interrupted: the app was closed while this job was running"
            self.update(job["id"], status="error", error=message)
            self.append_event(job["id"], {"type": "job_error", "error": message, "ts": now})
            changed += 1
        return changed

    # ---------------------------------------------------------------- events
    def append_event(self, job_id: str, event: dict) -> None:
        with self._lock:
            path = self.jobs_dir / job_id / "events.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")

    def read_events(self, job_id: str, after: int = 0) -> List[dict]:
        path = self.jobs_dir / job_id / "events.jsonl"
        if not path.exists():
            return []
        events = []
        with path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i < after:
                    continue
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
        return events

    # ----------------------------------------------------------------- paths
    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def out_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "out"

    def _write_job(self, job: dict) -> None:
        path = self.jobs_dir / job["id"] / "job.json"
        path.write_text(json.dumps(job, indent=2), "utf-8")
