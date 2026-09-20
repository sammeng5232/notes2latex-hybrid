"""Configure notes2latex-hybrid from the opencode API configuration.

- provider/model defaults come from ~/.config/opencode/opencode.jsonc
  (cu hkecon4 / SCRP proxy; qwen3.6-35b-1 = Vision + Text + Reasoning)
- the API key is copied from opencode's auth.json directly into the app's
  settings.json (source data dir + frozen %LOCALAPPDATA% dir); never printed.

Usage: .venv\\Scripts\\python scripts\\configure_from_opencode.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTH = Path.home() / ".local" / "share" / "opencode" / "auth.json"

PROVIDER = "cuhkecon4"
BASE_URL = "https://scrp-chat.econ.cuhk.edu.hk/api"
MODEL = "qwen3.6-35b-1"          # Vision + Text + Reasoning (probed & verified)
FROZEN_DATA = Path.home() / "AppData" / "Local" / "notes2latex-hybrid" / "data"


def get_key() -> str:
    data = json.loads(AUTH.read_text("utf-8"))
    info = data.get(PROVIDER) or {}
    key = info.get("apiKey") or info.get("token") or info.get("key") or ""
    if not key:
        raise SystemExit(f"no API key found for provider '{PROVIDER}' in {AUTH}")
    return key


def write_settings(data_dir: Path, key: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "settings.json"
    settings: dict = {}
    if path.exists():
        try:
            settings = json.loads(path.read_text("utf-8"))
        except ValueError:
            settings = {}
    settings.update({
        "engine": "vlm",
        "vlm_base_url": BASE_URL,
        "vlm_model": MODEL,
        "vlm_api_key": key,
        "vlm_timeout": 180,
        "dpi": 300,
        "latex_engine": "pdflatex",
        "compile_timeout": 60,
        "max_retries": 3,
        "context_lines": 40,
        "escalate_to_vlm": True,
    })
    path.write_text(json.dumps(settings, indent=2), "utf-8")
    print(f"wrote {path} (engine=vlm model={MODEL}, key: {len(key)} chars, not shown)")


def main() -> None:
    key = get_key()
    write_settings(ROOT / "data", key)
    write_settings(FROZEN_DATA, key)


if __name__ == "__main__":
    main()
