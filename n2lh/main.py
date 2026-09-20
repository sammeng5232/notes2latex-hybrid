"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from n2lh import __version__
from n2lh.api import router as api_router
from n2lh.config import Settings, load_settings
from n2lh.jobs import JobManager
from n2lh.store import JobStore

STATIC_DIR = Path(__file__).parent / "web" / "static"


def create_app(settings: Settings | None = None, data_dir: Path | None = None) -> FastAPI:
    data_dir = data_dir or Path(settings.data_dir if settings else "data")
    if settings is None:
        settings = load_settings(data_dir)

    app = FastAPI(title="notes2latex-hybrid", version=__version__)
    app.state.settings = settings
    app.state.store = JobStore(data_dir)
    app.state.manager = JobManager(app.state.store, settings)

    app.include_router(api_router)
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app
