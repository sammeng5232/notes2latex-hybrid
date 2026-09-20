"""LaTeX compilation wrapper: run, parse errors, always clean aux files.

Implements the strict cleanup policy used on this machine: after every compile,
auxiliary files (.aux/.log/.out/.toc/... and _minted-* dirs) are deleted; only
.tex and .pdf artifacts are kept.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

AUX_EXTENSIONS = (
    ".aux", ".log", ".out", ".toc", ".lof", ".lot", ".fls", ".fdb_latexmk",
    ".synctex.gz", ".bcf", ".run.xml", ".idx", ".ind", ".ilg", ".glo", ".gls",
    ".acn", ".acr", ".ist", ".nav", ".snm", ".vrb", ".xdv", ".dvi", ".blg", ".bbl",
)


@dataclass
class LatexError:
    message: str
    line: Optional[int] = None

    def __str__(self) -> str:  # pragma: no cover - convenience
        loc = f" (line {self.line})" if self.line else ""
        return f"{self.message}{loc}"


@dataclass
class CompileResult:
    ok: bool
    pdf_path: Optional[Path] = None
    errors: List[LatexError] = field(default_factory=list)
    log_excerpt: str = ""


class LatexCompiler:
    """Compile a .tex string in an isolated work directory.

    Prefers latexmk; falls back to a double pdflatex/xelatex pass. Aux files are
    removed in a finally block regardless of outcome.
    """

    def __init__(self, engine: str = "pdflatex", timeout: int = 60, cleanup_aux: bool = True) -> None:
        self.engine = engine
        self.timeout = timeout
        self.cleanup_aux = cleanup_aux

    # ------------------------------------------------------------------ public
    def compile(self, tex: str, workdir: Path, name: str = "document") -> CompileResult:
        workdir.mkdir(parents=True, exist_ok=True)
        tex_path = workdir / f"{name}.tex"
        tex_path.write_text(tex, encoding="utf-8")
        try:
            return self._compile(tex_path, workdir, name)
        finally:
            if self.cleanup_aux:
                clean_aux(workdir)

    # ----------------------------------------------------------------- private
    def _compile(self, tex_path: Path, workdir: Path, name: str) -> CompileResult:
        latexmk = shutil.which("latexmk")
        plain = shutil.which(self.engine)
        if latexmk:
            cmd = [
                latexmk, "-interaction=nonstopmode", "-halt-on-error",
                f"-{self.engine}", tex_path.name,
            ]
            return self._run(cmd, workdir, name)
        if plain:
            # Two passes resolve references; we only need success/failure + log.
            cmd = [plain, "-interaction=nonstopmode", "-file-line-error", tex_path.name]
            res = self._run(cmd, workdir, name)
            if res.ok:
                self._run(cmd, workdir, name)
            return res
        return CompileResult(
            ok=False,
            errors=[LatexError("No LaTeX toolchain found (latexmk or pdflatex). Install MiKTeX/TeX Live.")],
        )

    def _run(self, cmd: List[str], workdir: Path, name: str) -> CompileResult:
        try:
            proc = subprocess.run(
                cmd, cwd=str(workdir), timeout=self.timeout,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace",
            )
        except subprocess.TimeoutExpired:
            return CompileResult(ok=False, errors=[LatexError(f"Compilation timed out after {self.timeout}s")])
        except OSError as exc:
            return CompileResult(ok=False, errors=[LatexError(f"Failed to run compiler: {exc}")])

        pdf_path = workdir / f"{name}.pdf"
        log_text = _read_log(workdir, name) or proc.stdout
        errors = parse_log(log_text)
        ok = proc.returncode == 0 and pdf_path.exists()
        if not ok and not errors:
            errors = [LatexError(f"Compiler exited with code {proc.returncode}")]
        return CompileResult(ok=ok, pdf_path=pdf_path if pdf_path.exists() else None,
                             errors=errors[:5], log_excerpt=log_text[:4000])


# ---------------------------------------------------------------------- module
def parse_log(log_text: str, limit: int = 10) -> List[LatexError]:
    """Extract TeX errors from a .log / stdout blob."""
    errors: List[LatexError] = []
    lines = log_text.splitlines()
    i = 0
    while i < len(lines) and len(errors) < limit:
        line = lines[i]
        m = re.match(r"^(?:!\s+|(?:.+?\.tex:\d+):\s)(.+)$", line)
        if m and not _ignorable(m.group(1)):
            msg = m.group(1).strip()
            line_no = None
            # TeX prints "l.<n> ..." shortly after the error message.
            for j in range(i + 1, min(i + 6, len(lines))):
                lm = re.match(r"^l\.(\d+)", lines[j])
                if lm:
                    line_no = int(lm.group(1))
                    break
            errors.append(LatexError(message=msg, line=line_no))
        i += 1
    return errors


def _ignorable(msg: str) -> bool:
    return msg.startswith("Emergency stop") or msg.startswith("See the LaTeX manual")


def clean_aux(workdir: Path) -> None:
    """Strict aux cleanup: delete everything TeX-generated except .tex/.pdf."""
    if not workdir.exists():
        return
    for item in list(workdir.rglob("*")):
        if item.is_dir():
            if item.name.startswith("_minted-"):
                shutil.rmtree(item, ignore_errors=True)
            continue
        ext = "".join(item.suffixes)
        if item.suffix in AUX_EXTENSIONS or ext.endswith(AUX_EXTENSIONS):
            try:
                item.unlink()
            except OSError:
                pass


def _read_log(workdir: Path, name: str) -> str:
    log = workdir / f"{name}.log"
    if log.exists():
        try:
            return log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return ""
