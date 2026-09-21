"""CLI: `n2lh convert` (no server) and `n2lh serve` (web app)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="n2lh",
        description="notes2latex-hybrid: handwritten notes to compiled LaTeX, "
                    "offline-first with optional VLM escalation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert", help="convert files to LaTeX/PDF directly")
    p_convert.add_argument("inputs", nargs="+", help="PDF or image files")
    p_convert.add_argument("-o", "--out", default="out", help="output directory")
    p_convert.add_argument("--engine", default=None,
                           choices=["heuristic", "vlm", "hybrid"],
                           help="override engine (default: settings/env)")
    p_convert.add_argument("--dpi", type=int, default=None, help="PDF rasterization DPI")
    p_convert.add_argument("--vlm-base-url", default=None,
                           help="OpenAI-compatible VLM endpoint (cloud or local)")
    p_convert.add_argument("--vlm-model", default=None)
    p_convert.add_argument("--vlm-api-key", default=None)
    p_convert.add_argument("--trocr-model-dir", default=None,
                           help="fine-tuned image->LaTeX checkpoint dir")

    p_serve = sub.add_parser("serve", help="run the web app")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8710)
    p_serve.add_argument("--data-dir", default="data")

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn
        from n2lh.main import create_app

        app = create_app(data_dir=Path(args.data_dir))
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return 0

    return _convert(args)


def _convert(args: argparse.Namespace) -> int:
    from n2lh.compiler.latex import LatexCompiler
    from n2lh.config import load_settings
    from n2lh.jobs import build_engines
    from n2lh.pipeline.graph import DocumentPipeline
    from n2lh.pipeline.ingest import ingest_files
    from n2lh.recognition.base import PageImage
    from n2lh.recognition.prompts import doc_hint_block
    
    settings = load_settings(Path("data"))
    overrides = {
        "engine": args.engine,
        "dpi": args.dpi,
        "vlm_base_url": args.vlm_base_url,
        "vlm_model": args.vlm_model,
        "vlm_api_key": args.vlm_api_key,
        "trocr_model_dir": args.trocr_model_dir,
    }
    settings.apply_dict({k: v for k, v in overrides.items() if v is not None})

    try:
        primary, fixer = build_engines(
            settings, doc_hint=doc_hint_block([Path(p).name for p in args.inputs]))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    pages_dir = outdir / ".pages"

    print(f"[n2lh] engine={settings.engine} primary={primary.name} "
          f"fixer={fixer.name if fixer else None}")
    try:
        page_paths = ingest_files([Path(p) for p in args.inputs], pages_dir,
                                  dpi=settings.dpi)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"[n2lh] {len(page_paths)} page(s) ingested")

    pipeline = DocumentPipeline(
        primary, LatexCompiler(settings.latex_engine, settings.compile_timeout),
        fixer, settings.max_retries, settings.context_lines,
        settings.vlm_parallel_workers,
        preamble=settings.preamble(),
    )

    def on_event(ev: dict) -> None:
        kind = ev["type"]
        if kind in ("page_done", "job_done"):
            print(f"[n2lh] {kind}: {ev}")

    result = pipeline.run([PageImage(i + 1, p) for i, p in enumerate(page_paths)],
                          outdir, on_event=on_event)

    print(f"[n2lh] document: {outdir / 'document.tex'}")
    if result.pdf_path:
        print(f"[n2lh] pdf:     {result.pdf_path}")
    print(f"[n2lh] pages ok: {result.n_ok}/{len(result.pages)} "
          f"(verified compile: {'yes' if result.ok else 'NO'})")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
