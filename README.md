# notes2latex-hybrid

Handwritten notes in, **compiler-verified** LaTeX/PDF out — **offline-first**, with optional VLM escalation.

This project combines the complementary strengths of two open-source projects:

| Inherited from [`advaypakhale/notes2latex`](https://github.com/advaypakhale/notes2latex) | Inherited from [`wz-ml/Math2LaTeX`](https://github.com/wz-ml/Math2LaTeX) |
|---|---|
| Agentic generate → compile → fix loop (every page verified with `latexmk`) | Offline-first recognition: no cloud dependency required |
| Sequential pages with carried context + open-environment tracking | Classical line/equation segmentation stage (projection profiles + glyph geometry) |
| BYOK, model-agnostic VLM via OpenAI-compatible endpoints | Adapter slot for a fine-tuned TrOCR image→LaTeX checkpoint |
| Web UI + REST/SSE + CLI | Fully free/private inference path |

**The hybrid idea:** pages that the *local* engine can handle never touch an API. A VLM is
invoked only as a *repair/escalation* engine when a page fails to compile — so API tokens
are spent on the hard pages, not the whole notebook. With a local VLM endpoint
(Ollama / vLLM / LM Studio) the entire pipeline runs with zero cloud exposure.

## Architecture

```
input (PDF/images)            engines (pluggable)
      |                              |
  ingest ── rasterize pages          |-- heuristic   (offline structure skeleton, zero deps)
      |                              |-- trocr       (your fine-tuned image->LaTeX checkpoint)
  segment (text/math regions)        |-- vlm         (OpenAI-compatible: OpenRouter/Ollama/vLLM/...)
      |                              |
      +──> DocumentPipeline:  transcribe page ── compile (latexmk) ── ok? next page
                                    ^                          |
                                    └──── error log fix (max N retries, VLM escalation)
      |
  context window (last N lines) + open-environment tracker carried across pages
      |
  output: document.tex + document.pdf   (aux files auto-deleted after every compile)
```

## Quick start

```powershell
# 1. install (Python 3.10+)
uv venv .venv
uv pip install -e ".[dev,pdf]"

# 2. run the web app
.venv\Scripts\python -m n2lh.cli serve          # http://127.0.0.1:8710

# ...or convert directly from the CLI (no server)
.venv\Scripts\python -m n2lh.cli convert notes.pdf -o out --engine heuristic
```

Open the UI, optionally configure Settings (engine, VLM endpoint, TrOCR checkpoint),
upload a PDF or photos, and watch pages stream through transcribe → compile → fix.
Each page is shown side-by-side with its LaTeX for review; download the `.tex`/`.pdf`
when done.

## Engines

| engine | primary | repair/escalation | needs network | notes |
|---|---|---|---|---|
| `heuristic` | classical segmentation → labeled skeleton | itself (safe-mode) | no | structure-only demo engine; output compiles but does not transcribe content |
| `vlm` | VLM | VLM | yes (unless local endpoint) | closest to upstream notes2latex behavior |
| `hybrid` *(default)* | TrOCR checkpoint if configured, else heuristic | VLM | only on failing pages | the combined-advantages mode |

### Plugging in real recognition

- **VLM (easiest quality):** Settings -> VLM base URL + model. This install defaults to the
  opencode-configured provider (`https://scrp-chat.econ.cuhk.edu.hk/api`) with
  **`qwen3.6-35b-1`** -- probed and verified as Vision + Text + Reasoning
  (`qwen3.6-35b-2` is the alternate; `glm-5.3`/`glm-5.2` there are text-only).
  API credentials were copied from opencode's `auth.json` via
  `scripts/configure_from_opencode.py` into the app's settings.json
  (source `data\` and frozen `%LOCALAPPDATA%\notes2latex-hybrid\data\`).
  Also works with OpenRouter (`https://openrouter.ai/api/v1`) or fully local
  endpoints: Ollama (`http://localhost:11434/v1`), LM Studio
  (`http://localhost:1234/v1`), vLLM.

  The provider's display names (`Vision, Text and Reasoning`, `Reasoning`, `Text`)
  are **chat-app presets, not callable model ids** -- they return
  `400 Model not found` on the OpenAI-compatible API. Use a concrete id (or the
  `vision` alias, which routes to `qwen3.6-35b-2`).

  **System proxies:** the client connects **directly** and ignores
  `HTTPS_PROXY`/`HTTP_PROXY` by default, because a global proxy (e.g. Clash Verge)
  drops the TLS handshake to this endpoint with `UNEXPECTED_EOF_WHILE_READING`.
  Set `N2LH_VLM_USE_PROXY=true` (or the Settings toggle) to route through it
  deliberately. `GET /api/preflight` reports live reachability as
  `vlm_ok`/`vlm_error`, so an unreachable or misconfigured model is visible
  before a job is queued.
- **TrOCR (fully offline, Math2LaTeX heritage):** fine-tune an image→LaTeX checkpoint
  (e.g. following [Math2LaTeX](https://github.com/wz-ml/Math2LaTeX) training), then set
  `N2LH_TROCR_MODEL_DIR` to the saved model dir. The adapter recognizes page regions
  one-by-one using the segmentation stage. Requires the `trocr` extra:
  `uv pip install -e ".[trocr]"`.

## Configuration

Web UI Settings page, `.env` (see `.env.example`, prefix `N2LH_`), or CLI flags.
Key knobs: `engine`, `vlm_base_url`, `vlm_model`, `vlm_api_key`, `trocr_model_dir`,
`dpi`, `max_retries` (repairs per page), `context_lines`, `latex_engine`,
`compile_timeout`, `escalate_to_vlm`, `vlm_timeout`, `vlm_stall_timeout`,
`vlm_retries`, `vlm_thinking`, `vlm_parallel_workers`, `doc_font_pt`, `output_dir`.

**Getting the result out.** The job page has **Download .tex** / **Download .pdf**
and **Save to folder...**, and Settings has an **Output folder**.

- *Save to folder* / *Output folder* copy `document.tex`, `document.pdf` **and the
  `figures/` folder** into a folder of your choosing, named after the file you
  uploaded (`Topology Notes.pdf`), never overwriting an earlier save
  (`Topology Notes (2).pdf`). In the desktop window the folder is chosen with the
  native Windows folder picker (`window.pywebview.api`); in a browser tab, which has
  no such dialog, you are asked for the path. Set **Output folder** and every
  finished job is copied there automatically; if that copy fails (folder gone, disk
  full) the job still succeeds and the failure is reported in the progress log.
- The two Download buttons stream the single file. They did nothing at all in the
  desktop window before: pywebview cancels every download unless
  `ALLOW_DOWNLOADS` is set, and the window has no download bar to show it. They now
  work and open the system Save-as dialog. A `.tex` is served as
  `application/x-tex` so it downloads instead of opening inside the window.

**`doc_font_pt` (default 11; 10, 11 or 12):** the base text size of the generated
document (`\documentclass[NNpt]{article}`). Sizes *within* a page -- a larger
title, a heading -- come from the handwriting itself, see "Formatting" below.

**Figures.** Hand-drawn diagrams are embedded as **images cropped from the page**,
not redrawn in TikZ (the model's TikZ for hand-drawn diagrams often did not
compile, and the page then ended up unverified or with the diagram omitted).
Three things decide a crop, each doing what it is actually good at:

1. **The transcription says how many figures there are and where they belong.**
   While transcribing, the model writes `\figbox{x0}{y0}{x1}{y1}` (integers
   0-1000, origin top-left) in reading order.
2. **A second, dedicated request says where they are.** A model writing out a page
   cannot also measure it: on a real page its four inline boxes were
   `(200,200,400,350)`, `(200,450,400,600)`, `(200,650,400,800)`,
   `(200,850,400,1000)` -- evenly spaced round numbers sitting to the left of the
   actual drawings. Asked *only* to locate figures, the same model returns boxes
   that land on them. So the located boxes supply the coordinates: equal counts are
   paired in reading order (both lists describe the same drawings), otherwise each
   marker takes the nearest located box by height, and a marker with no match keeps
   its own box. (An earlier version matched them by overlap and required IoU >= 0.3;
   the hallucinated boxes score ~0.12, so *every* crop silently used the wrong
   coordinates. Do not gate a good estimate on agreeing with a bad one.)
3. **The page image says where the ink is.** The box is then snapped onto the
   drawing. Each edge grows while a stroke still crosses it, stopping at the first
   clear strip and travelling at most 2.5% of the page (where lines of writing are
   tightly spaced, a bigger budget walked a box up through two whole lines of
   text). The box is tightened onto the ink, any band of ink that is *writing* is
   dropped from the top and bottom, and it is tightened again.

   A band is writing when it is between 0.35 and 1.8 times **this page's own line
   height** and runs across at least 55% of the crop. The line height is measured
   from the page, not assumed, so it follows the hand and the scan: ink runs that
   cross half the page are collected and the low quartile is taken, because in a
   close hand the descenders of one line touch the ascenders of the next and a run
   is usually a whole paragraph -- only the single-line ones measure a line. On
   these notes that separates a stroke spanning the crop (0.19-0.26 of a line),
   a line of writing even where the crop clips it (0.42-1.52) and a drawing (3.9).
   Captions and axis labels are narrower than 55% and stay.

   The figure is anchored on the thickest band that is *not* writing, so a remark
   above a small diagram cannot become the figure, and bands between kept ones
   stay, so a multi-part diagram is whole. **Only rows are trimmed**: writing runs
   horizontally, so a stray line is always a row, whereas a column is one of the
   figure's own parts -- trimming columns by the same rule sheared the `V` and
   `R^n` labels off the right of a chart diagram. If *every* band is writing the
   locator pointed at prose (it did once, with the drawing just below its box), and
   the figure is reported as `\textit{[figure omitted]}` rather than embedding a
   picture of a sentence. A box on blank paper is omitted for the same reason.

The crop (at most 1400 px wide) goes to `figures/pNNNN_fK.png` next to
`document.tex`, and the marker becomes a centered `\includegraphics` sized from the
drawing's real width on the page, capped at 0.78\linewidth. The preamble's
`\graphicspath` finds `figures/` from the per-page compile directories and from the
final compile, so figure pages are verified like any other. A page with figures
costs one extra VLM request. The PDF has the images embedded; a `.tex` used on its
own needs the `figures/` folder beside it (**Save to folder** copies both).

**Note labels.** `Def.`, `Thm (Poincare duality).`, `Cor.`, `Eg.`, `Pf.`, `Rmk.`
and a `Lecture 9 20251109 Week 12` heading are set in **bold** -- in the
handwriting those labels carry the structure of the page. The prompt asks for it
and `n2lh/pipeline/style.py` then applies it deterministically to every page,
because the model obliges on some pages, forgets on others, and nothing in the
compile output reveals the omission. A label the model already emboldened is left
alone, and `Defined`, `Corollaries`, a label mid-sentence, one inside math and one
in a comment are all untouched.

**Formatting.** The prompts ask the model to reproduce what is visible, not only
the words: text centered on the page (titles, cover lines) in
`\begin{center}`, underlined text with `\underline{}`, visibly larger writing
with `\LARGE` / `\Large` / `\large`, and headings as `\subsection*{}`. The
notation glossary tells it that a handwritten `∀` (an upside-down A) is
`\forall` and not a `v`/`V` -- "Vp ∈ M" is `\forall p \in M`. (An underscore in
the handwriting is an underline; a literal `_` in the notes is escaped as
`\_`.)

**Mechanical auto-repair.** When a page fails to compile, a deterministic pass
(`n2lh/pipeline/autofix.py`) runs *before* asking the model to fix it: it
escapes a bare `&` outside alignment environments, balances unclosed / stray /
crossed `\begin`/`\end`, wraps math-only environments (`aligned`, `cases`,
matrices, `array`) used in text mode in `\[ ... \]`, widens an `array`/`tabular`
column spec that is narrower than its widest row, escapes a bare `#`, and
replaces raw Unicode math symbols (`∤ ∈ ℝ α ...`, which pdflatex rejects) with
their macros. In real runs a bare `&` (handwritten "&" for "and") was 16-18 of the
22-25 first-attempt failures, one page bounced between three faults through
four model repairs, and another failed four times on `array{ccccccc}` holding
rows of nine cells. The autofix result is kept only if it compiles or changes
the error; it costs no model call. The repair prompt also carries hints for
these errors (including "a picture must not sit inside `\[ \]`").

A figure marker the model wrapped in display math (`\[ \figbox{..} \]`) is
unwrapped before it is replaced by the picture; without that, the picture ended
up in math mode and the page failed `Missing $ inserted` on every repair.

**VLM request limits.** Every VLM request streams, and a watchdog thread watches
for actual model output (`data:` events -- keep-alive `: ping` comments do not
count):

- `vlm_stall_timeout` (default 120s): no output for this long => the socket is
  shut down and the request is retried. Healthy pages start streaming within
  seconds; raise it only for a provider that hides its reasoning and sends
  nothing until it has finished thinking.
- `vlm_timeout` (default 600s): hard wall-clock cap on ONE request, however
  much it is streaming.
- `vlm_retries` (default 4): extra attempts, each on a fresh connection, before
  the failure is reported against the page. Cheap, because a stalled or looping
  attempt is now cut off in seconds (see below), not after the 10-minute cap.

**Runaway loops.** On the CUHK endpoint the model sometimes finishes part of a
page and then emits `\quad \quad \quad ...` forever (20-25k characters in 50s;
35-50% of identical requests on the worst pages; same at temperature 0.1/0.3/0.6,
thinking on or off). A guard compresses the last 2,000 characters of the stream:
real transcripts score 0.33-0.42, loops ~0.03, and below 0.10 the request is
aborted (~1,000 characters into the loop, ~5s), the connection dropped, and the
request retried with the temperature nudged up. The partial text is deliberately
**not** salvaged: on real samples the loop began ~65% of the way through the page,
so keeping the head would silently drop the rest. A 24,000-character ceiling
backstops output that is degenerate but not compressible. (A frequency penalty
also stops the loop, but on the test page it fabricated a proof and dropped half
the page, so it is not used.)
- `vlm_thinking` (default `off`): `off` sends
  `chat_template_kwargs.enable_thinking=false` (vLLM/Qwen) so the model answers
  directly; `auto` keeps the provider default but retries a stalled/looping
  attempt with it off; `on` never sends the switch. A provider that rejects the
  field gets the request resent without it. Measured on the CUHK endpoint
  (qwen3.6-35b-2): same-or-better transcripts (on a dense page the direct answer
  read the handwritten phi/psi correctly where the reasoning run wrote psi/chi),
  4-10x faster (9s vs 42s; 15s vs 600s+), and none of the endless reasoning
  loops -- page 9 streamed 8,346 events for 10 minutes with thinking on and never
  finished.

Why a watchdog and not just httpx's read timeout: the shared campus endpoint
this was built against answers `200 OK` headers immediately and *sometimes never
sends a body* (and may send keep-alive pings), which resets a read timeout
forever. The old code then waited out a 15-minute deadline per attempt without
ever closing the connection. Aborting the socket also lets the server see the
disconnect and drop the request. `GET /api/preflight` only checks reachability
(15s cap), not per-page latency under load.

**`vlm_parallel_workers` (default 4):** the VLM round-trip (seconds to
minutes per page), not compiling, dominates wall-clock time on a multi-page
document, and by default pages are processed one at a time. Setting this
above 1 prefetches that many pages' first-attempt transcriptions
*concurrently*, then still runs the compile+repair loop sequentially in page
order using the prefetched results. Trade-off: prefetched pages are
transcribed without seeing the rolling context / open-environment state from
earlier pages (they're all dispatched at once, before any page has actually
finished), so cross-page notation consistency and environments left open
across a page break lean more heavily on the repair pass to catch and fix --
which it still does, with the real context and the real compiler error. Set
to `1` for fully sequential processing (matches the original notes2latex
behavior exactly, maximum cross-page consistency, slowest). A prefetch
failure for one page (e.g. a transient network error) is not fatal -- that
page transparently falls back to a live sequential call in the normal loop.

**Failure isolation.** A page that cannot be made to compile is kept in the
`.tex` only as a *commented-out* block under a gray "unverified" note, never as
live LaTeX, and it does not feed the rolling context or open-environment
tracking. (Before this, one bad page stayed in the accumulating document and
made every later page fail to compile with the same error -- and the final PDF
too.) Model output is also sanitized before compiling: a stray
`\documentclass` / `\usepackage` / `\begin{document}` wrapper is stripped, since
the prompts ask for body-only LaTeX but models do not always comply, especially
on repair passes.

**Stopping early and cancelling.** If the recognizer itself fails on 3 pages in
a row (`DocumentPipeline(max_consecutive_engine_failures=3)`) the job stops instead of grinding
through the rest against a dead endpoint; pages that were already prefetched are
still compiled and kept, the remaining ones are marked "not attempted", and the
partial document is still built. The job page has a **Cancel job** button
(`POST /api/jobs/{id}/cancel`) that aborts the in-flight request the same way.
Jobs left "running" by an app that was closed are marked *interrupted* on the
next start.

**Last-resort deadline.** On top of the client's own limits, every recognizer
call runs under a wall-clock deadline (15 minutes, or the recognizer's declared
`hard_budget` if larger) so a custom engine that cannot abort itself still can't
hang the job; a call that exceeds it is abandoned and the page fails cleanly.

## Output hygiene

After **every** compile (per-page and final), auxiliary files
(`.aux .log .out .toc .fls .fdb_latexmk .synctex.gz ...` and `_minted-*`) are deleted;
only `.tex` and `.pdf` are ever left behind (plus the `figures/` image folder,
which is an input of the document, not a compile byproduct). This is enforced in code
(`n2lh/compiler/latex.py: clean_aux`), not just convention.

## Desktop exe (Windows)

Build a single-file GUI executable:

```powershell
powershell -File build\build_exe.ps1        # runs tests, then PyInstaller
# result: dist\notes2latex-hybrid.exe  (~40 MB, windowed, no install needed)
```

Using it:

| invocation | behavior |
|---|---|
| `notes2latex-hybrid.exe` | native desktop window (WebView2) hosting the app; close the window to quit. Falls back to browser tab + system-tray icon (Open / Quit) if WebView2 is unavailable. |
| `notes2latex-hybrid.exe convert FILES -o out` | scriptable CLI conversion, same engines |
| `notes2latex-hybrid.exe serve --port 8710` | plain web server |

Frozen-app details:

- Data, settings, job history, and logs live under
  `%LOCALAPPDATA%\notes2latex-hybrid\` (`data\`, `logs\app.log`).
- `N2LH_PORT` forces the GUI port; `N2LH_GUI_NO_WINDOW=1` runs headless
  (automation/testing — used by `build\verify_exe.py`).
- Requires a LaTeX toolchain on PATH (`latexmk`/`pdflatex`, e.g. MiKTeX);
  the UI shows a warning banner if it is missing.
- The TrOCR engine (torch/transformers) is not bundled in the exe — use the
  heuristic/VLM engines, or install from source with the `trocr` extra.
- `build\verify_exe.py` is a full headless acceptance test of the frozen exe
  (boot, upload, 2-page job, real compile, downloads, event stream, aux-cleanup audit).
- `build\verify_ui_assets.py` checks every static asset served by the frozen exe
  (`/`, `/app.js`, `/style.css`, `/favicon.ico`) and that `/api/settings` reports the
  saved API key as present while masking it.
- `build\verify_job_failure.py` proves a job whose pages all fail to compile is
  reported as `ok=false` (`pages_ok=0`, every page `failed`) instead of a false success.

## Development

```powershell
uv pip install -e ".[dev,pdf]"
.venv\Scripts\python -m pytest -q
```

Tests cover segmentation, context/environment tracking, log parsing, the
generate-compile-fix loop (with fake engines), and an end-to-end API job that runs a
real LaTeX compile. Tests are skipped automatically when no TeX toolchain is present.

## Honest limitations

- The built-in heuristic engine produces a *compilable structure skeleton*, not a
  transcription — real content recognition requires the TrOCR checkpoint or a VLM.
- Compile-verification guarantees the document *compiles*; it cannot prove the math
  matches your handwriting. Use the side-by-side review.
- PDF ingestion needs the `pdf` extra (`pypdfium2`); TrOCR needs heavyweight ML deps.

## Credits & licenses

- Design ideas from [advaypakhale/notes2latex](https://github.com/advaypakhale/notes2latex) (MIT)
  and [wz-ml/Math2LaTeX](https://github.com/wz-ml/Math2LaTeX) (MIT).
- This project: MIT (see LICENSE).
