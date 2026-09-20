"""Copy a finished job's output to a folder the user keeps.

Job output lives under the app's data directory, which is fine for the app and
useless for the person: they want the PDF in their notes folder, under the name
of the file they uploaded. The `.tex` references its pictures as
``figures/<name>.png``, so the folder travels with it -- copying the `.tex`
alone would give a document that no longer compiles.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List


def safe_stem(name: str, fallback: str) -> str:
    """A filename from the uploaded document's name, safe on every platform.

    The separators go first, so an upload called ``week 3/4: charts.pdf`` keeps
    its name instead of being cut down to its last path segment, and a name like
    ``../../secret.pdf`` cannot point anywhere.
    """
    flat = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "", name)
    return (Path(flat).stem.strip(" .") or fallback)[:80]


def unique(path: Path) -> Path:
    """``notes.pdf`` -> ``notes (2).pdf``, so a second save never overwrites the first."""
    if not path.exists():
        return path
    for n in range(2, 1000):
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not candidate.exists():
            return candidate
    return path


def save_outputs(out_dir: Path, target: Path, stem: str) -> List[str]:
    """Copy ``document.tex``/``document.pdf`` and ``figures/`` into ``target``.

    Returns what was written. Raises OSError if the folder cannot be written.
    """
    target.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for kind in ("tex", "pdf"):
        src = out_dir / f"document.{kind}"
        if src.exists():
            dest = unique(target / f"{stem}.{kind}")
            shutil.copy2(src, dest)
            written.append(dest.name)
    figures = sorted((out_dir / "figures").glob("*.png"))
    if figures:
        dest_dir = target / "figures"
        dest_dir.mkdir(exist_ok=True)
        for png in figures:
            shutil.copy2(png, dest_dir / png.name)
        written.append(f"figures/ ({len(figures)} images)")
    return written
