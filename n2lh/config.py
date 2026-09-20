from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

from n2lh.recognition.prompts import FONT_SIZES, PAPER_SIZES, make_preamble

ENV_PREFIX = "N2LH_"

VALID_ENGINES = ("heuristic", "vlm", "hybrid")


@dataclass
class Settings:
    """Central configuration. Persisted in <data_dir>/settings.json; env vars win."""

    # Default: the opencode-configured provider on this machine (SCRP proxy),
    # whose qwen3.6-35b models support Vision + Text + Reasoning.
    engine: str = "vlm"
    # VLM (OpenAI-compatible). Empty base_url disables VLM.
    vlm_base_url: str = "https://scrp-chat.econ.cuhk.edu.hk/api"
    vlm_api_key: str = ""
    vlm_model: str = "qwen3.6-35b-1"
    # Hard cap on ONE VLM request, wall-clock seconds (not an idle timeout).
    vlm_timeout: int = 600
    # A request is aborted (socket shut down) and retried if no model output
    # event arrives for this long; keep-alive pings don't count. Healthy pages
    # start streaming within seconds, but raise this for a provider that hides
    # its reasoning and sends nothing until it has finished thinking.
    vlm_stall_timeout: int = 120
    # Extra attempts (fresh connection each) after a stalled/looping/failed
    # request, before the error is reported against the page. Cheap now: a
    # runaway loop is caught in ~5s, and on the CUHK endpoint a retry succeeds
    # about half the time on the worst pages (they loop on 35-50% of requests).
    vlm_retries: int = 4
    # Reasoning phase of thinking models (Qwen3 & co.): "off" asks the model not
    # to think, "auto" keeps the provider default but retries a looping/stalled
    # attempt with it off, "on" leaves it to the provider. Measured on the SCRP
    # endpoint: same or better transcripts, ~4-10x faster, and none of the
    # endless reasoning loops that made some pages run for 10+ minutes.
    vlm_thinking: str = "off"
    # Route VLM calls through the system HTTPS_PROXY. Off by default: a global
    # proxy such as Clash Verge can break TLS to campus endpoints.
    vlm_use_proxy: bool = False
    # TrOCR adapter (Math2LaTeX-style fine-tuned checkpoint directory).
    trocr_model_dir: str = ""
    # Pipeline knobs.
    dpi: int = 300
    latex_engine: str = "pdflatex"
    compile_timeout: int = 60
    max_retries: int = 3
    context_lines: int = 40
    escalate_to_vlm: bool = True
    # Fetch every page's first-attempt VLM transcription concurrently instead
    # of one page at a time. The VLM round-trip (seconds to minutes per page)
    # is by far the slowest step, so this is the main lever for wall-clock
    # time on multi-page documents. 1 = fully sequential (each page sees the
    # real rolling context from prior pages, matching the original
    # notes2latex design exactly). >1 trades some cross-page notation/
    # environment-continuity awareness on the first attempt for parallel
    # throughput; failed pages still repair sequentially with real context.
    vlm_parallel_workers: int = 4
    # Page setup of the generated document. Paper and margins matter for notes:
    # a3paper fits a dense scanned page without shrinking the figures.
    doc_font_pt: int = 11
    doc_paper: str = "a4paper"
    doc_margin_in: float = 1.0
    doc_landscape: bool = False
    doc_two_column: bool = False
    # Where finished documents are copied, named after the uploaded file. Empty =
    # keep them only in the job folder (they stay downloadable either way).
    output_dir: str = ""
    data_dir: str = "data"

    # ---------------------------------------------------------------- helpers
    def validate(self) -> list[str]:
        problems = []
        if self.engine not in VALID_ENGINES:
            problems.append(f"engine must be one of {VALID_ENGINES}")
        if self.engine == "vlm" and not self.vlm_ready:
            problems.append("engine=vlm requires vlm_base_url and vlm_model")
        # engine=hybrid without a VLM endpoint is valid: it degrades to a
        # fully-offline pipeline (heuristic/trocr primary, no escalation).
        if self.dpi < 50 or self.dpi > 1200:
            problems.append("dpi should be between 50 and 1200")
        if self.max_retries < 0:
            problems.append("max_retries must be >= 0")
        if self.vlm_parallel_workers < 1 or self.vlm_parallel_workers > 16:
            problems.append("vlm_parallel_workers should be between 1 and 16")
        if self.vlm_stall_timeout < 10:
            problems.append("vlm_stall_timeout should be at least 10 seconds")
        if self.vlm_timeout < self.vlm_stall_timeout:
            problems.append("vlm_timeout must be >= vlm_stall_timeout")
        if self.vlm_retries < 0 or self.vlm_retries > 10:
            problems.append("vlm_retries should be between 0 and 10")
        if self.vlm_thinking not in ("off", "auto", "on"):
            problems.append("vlm_thinking must be one of off, auto, on")
        if self.doc_font_pt not in FONT_SIZES:
            problems.append(f"doc_font_pt must be one of {FONT_SIZES}")
        if self.doc_paper not in PAPER_SIZES:
            problems.append(f"doc_paper must be one of {PAPER_SIZES}")
        if not 0.25 <= float(self.doc_margin_in) <= 4:
            problems.append("doc_margin_in should be between 0.25 and 4 inches")
        if self.output_dir.strip():
            target = Path(self.output_dir).expanduser()
            if not target.is_absolute():
                problems.append("output folder must be a full path")
            elif target.exists() and not target.is_dir():
                problems.append("output folder is a file, not a folder")
        return problems

    def preamble(self) -> str:
        """The LaTeX preamble for the configured page setup (one place, two callers)."""
        return make_preamble(self.doc_font_pt, self.doc_paper, self.doc_margin_in,
                             self.doc_landscape, self.doc_two_column)

    @property
    def vlm_ready(self) -> bool:
        return bool(self.vlm_base_url.strip() and self.vlm_model.strip())

    @property
    def trocr_ready(self) -> bool:
        return bool(self.trocr_model_dir.strip() and Path(self.trocr_model_dir).exists())

    def to_dict(self) -> dict:
        d = asdict(self)
        # Never echo the full API key back to browsers.
        if d.get("vlm_api_key"):
            d["vlm_api_key_set"] = True
        return d

    def masked(self) -> dict:
        d = self.to_dict()
        if d.get("vlm_api_key"):
            d["vlm_api_key"] = d["vlm_api_key"][:4] + "..." if len(d["vlm_api_key"]) > 4 else "***"
        return d

    # ------------------------------------------------------------ persistence
    def apply_dict(self, values: dict) -> None:
        names = {f.name for f in fields(self)}
        for k, v in (values or {}).items():
            if k in names and v is not None:
                current = getattr(self, k)
                if isinstance(current, bool):
                    v = str(v).strip().lower() in ("1", "true", "yes", "on")
                elif isinstance(current, int):
                    try:
                        v = int(v)
                    except (TypeError, ValueError):
                        continue
                elif isinstance(current, float):
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        continue
                # Allow clearing the API key only by sending a non-empty value.
                if k == "vlm_api_key" and v == "__keep__":
                    continue
                setattr(self, k, v)

    def apply_env(self, environ: Optional[dict] = None) -> None:
        env = environ if environ is not None else os.environ
        values = {}
        for f in fields(self):
            raw = env.get(ENV_PREFIX + f.name.upper())
            if raw is not None and raw != "":
                values[f.name] = raw
        if values:
            self.apply_dict(values)

    def save(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "settings.json"
        payload = asdict(self)
        if not payload["vlm_api_key"]:
            # preserve an existing stored key if the in-memory copy was masked
            try:
                old = json.loads(path.read_text("utf-8-sig"))  # tolerate a BOM
                if old.get("vlm_api_key"):
                    payload["vlm_api_key"] = old["vlm_api_key"]
            except (OSError, ValueError):
                pass
        path.write_text(json.dumps(payload, indent=2), "utf-8")


def load_settings(data_dir: Path, environ: Optional[dict] = None) -> Settings:
    """Load settings.json -> apply env overrides -> return."""
    s = Settings()
    path = data_dir / "settings.json"
    if path.exists():
        try:
            # utf-8-sig transparently strips a leading BOM (e.g. written by
            # PowerShell's `Set-Content -Encoding utf8`, which is BOM'd unlike
            # every other common utf-8 writer); plain "utf-8" would otherwise
            # fail json.loads() and this except would SILENTLY fall back to
            # blank defaults -- including an empty API key -- which is much
            # worse than a visible error, so we avoid the trap entirely.
            s.apply_dict(json.loads(path.read_text("utf-8-sig")))
        except (OSError, ValueError):
            pass
    s.apply_env(environ)
    s.data_dir = str(data_dir)
    return s
