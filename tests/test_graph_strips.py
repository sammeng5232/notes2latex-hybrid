"""A dense page is read as full-resolution strips (see n2lh/pipeline/tiles.py)."""

from __future__ import annotations

import threading
from typing import List, Optional

from PIL import Image, ImageDraw

from n2lh.pipeline.graph import DocumentPipeline
from n2lh.pipeline.tiles import STRIP_EDGE, count_lines, planning_mask
from n2lh.recognition.base import PageImage, Recognizer, TranscribeResult

from tests.test_graph import FakeCompiler, ScriptedRecognizer


def page_image(path, pitch: int, width: int = 1200, height: int = 1700) -> PageImage:
    """A page of 'handwriting' lines at the given pitch."""
    im = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(im)
    for y in range(pitch, height - pitch, pitch):
        for x in range(60, width - 60, 75):
            d.rectangle([x, y, x + 30, y + pitch // 2], fill="black")
    im.save(path, "PNG")
    return PageImage(index=int(path.stem[1:]), path=path)


def dense_pages(tmp_path, n: int = 1) -> List[PageImage]:
    # 30px lines on a 1700px page: 28px per line at the whole-page size.
    return [page_image(tmp_path / f"p{i:04d}.png", pitch=30) for i in range(1, n + 1)]


class StripRecognizer(Recognizer):
    """Reads strips; records every call. It writes one line per line of ink it
    sees. ``fail`` maps a strip index to how many of its reads raise before one
    succeeds; ``short`` to how many of its reads come back with lines skipped."""

    reads_strips = True

    def __init__(self, name: str = "strips", first: str = "",
                 fail: Optional[dict] = None, repaired: str = "\\text{repaired}",
                 short: Optional[dict] = None) -> None:
        self.name = name
        self.first = first
        self.fail = dict(fail or {})
        self.short = dict(short or {})
        self.repaired = repaired
        self.lock = threading.Lock()
        self.strip_calls: List[dict] = []
        self.page_calls: List[dict] = []
        self.repairs: List[str] = []

    def transcribe(self, page, context_tail, open_environments, guidance=None):
        self.page_calls.append({"page": page.index, "guidance": guidance})
        return TranscribeResult(latex="\\text{whole page}", engine=self.name)

    def transcribe_strip(self, image_path, index, total, *, max_edge, context_tail="",
                         open_environments=(), page_colors=None, on_retry=None, variant=0):
        with self.lock:
            self.strip_calls.append({"path": image_path, "index": index, "total": total,
                                     "max_edge": max_edge, "tail": context_tail,
                                     "variant": variant})
            failing = self.fail.get(index, 0)
            if failing:
                self.fail[index] = failing - 1
            skipping = self.short.get(index, 0)
            if skipping:
                self.short[index] = skipping - 1
        if failing:
            raise RuntimeError(f"strip {index} failed")
        page = int(str(image_path.name).split("-p")[1][:4])
        with Image.open(image_path) as im:
            n = max(1, count_lines(planning_mask(im.convert("RGB"))))
        lines = [f"page {page} strip {index} of {total}, line {j}: $x_{j}$" for j in range(n)]
        if skipping:
            lines = lines[:1]
        text = "\n".join(lines)
        return (self.first + "\n" + text) if index == 1 and self.first else text

    def repair_text(self, guidance):
        self.repairs.append(guidance)
        return TranscribeResult(latex=self.repaired, engine=self.name)


def strip_indices(rec) -> List[int]:
    return [c["index"] for c in rec.strip_calls]


def test_a_dense_page_is_read_as_full_resolution_strips_in_order(tmp_path):
    rec = StripRecognizer()
    events = []
    result = DocumentPipeline(rec, FakeCompiler(), fixer=None).run(
        dense_pages(tmp_path), tmp_path / "out", events.append)

    assert result.ok and rec.page_calls == []
    total = rec.strip_calls[0]["total"]
    assert total >= 2 and sorted(strip_indices(rec)) == list(range(1, total + 1))
    assert all(c["max_edge"] == STRIP_EDGE for c in rec.strip_calls)
    body = result.tex
    order = [body.index(f"page 1 strip {k} of {total}") for k in range(1, total + 1)]
    assert order == sorted(order)
    planned = [e for e in events if e["type"] == "strips_planned"]
    assert planned and planned[0]["strips"] == total
    assert len([e for e in events if e["type"] == "strip_done"]) == total


def test_the_strips_are_cut_from_the_page_itself(tmp_path):
    rec = StripRecognizer()
    pages = dense_pages(tmp_path)
    DocumentPipeline(rec, FakeCompiler(), fixer=None).run(pages, tmp_path / "out")
    heights = []
    for c in sorted((c for c in rec.strip_calls if c["variant"] == 0),
                    key=lambda c: c["index"]):
        with Image.open(c["path"]) as im:
            assert im.width == 1200
            heights.append(im.height)
    assert sum(heights) == 1700


def test_a_sparse_page_is_read_whole(tmp_path):
    rec = StripRecognizer()
    page = page_image(tmp_path / "p0001.png", pitch=90)     # 85px per line at 1600
    result = DocumentPipeline(rec, FakeCompiler(), fixer=None).run([page], tmp_path / "out")
    assert result.ok and rec.strip_calls == [] and len(rec.page_calls) == 1


def test_an_engine_that_cannot_read_strips_reads_the_dense_page_whole(tmp_path):
    rec = ScriptedRecognizer(outputs=["\\text{whole}"])
    result = DocumentPipeline(rec, FakeCompiler(), fixer=None).run(
        dense_pages(tmp_path), tmp_path / "out")
    assert result.ok and len(rec.calls) == 1 and "\\text{whole}" in result.tex


def test_a_failed_strip_is_read_again_and_the_others_are_not(tmp_path):
    rec = StripRecognizer(fail={3: 1})
    events = []
    result = DocumentPipeline(rec, FakeCompiler(), fixer=None, max_retries=3).run(
        dense_pages(tmp_path), tmp_path / "out", events.append)
    assert result.ok
    total = rec.strip_calls[0]["total"]
    counts = {k: strip_indices(rec).count(k) for k in range(1, total + 1)}
    assert counts[3] == 2
    assert all(n == 1 for k, n in counts.items() if k != 3)
    assert f"strip 3 of {total}" in result.tex


def test_a_tiled_page_is_repaired_from_its_latex_alone(tmp_path):
    """The only image a repair could send is the illegible whole page."""
    rec = StripRecognizer(first="\\BROKEN")
    result = DocumentPipeline(rec, FakeCompiler(), fixer=None, max_retries=3).run(
        dense_pages(tmp_path), tmp_path / "out")
    assert result.ok and result.pages[0].status == "fixed"
    assert len(rec.repairs) == 1 and rec.page_calls == []
    assert "No image is attached" in rec.repairs[0]
    # the whole transcription goes into the repair, not only its tail
    assert "strip 1 of" in rec.repairs[0]
    assert "\\text{repaired}" in result.tex


def test_prefetch_reads_each_pages_strips(tmp_path):
    rec = StripRecognizer()
    events = []
    result = DocumentPipeline(rec, FakeCompiler(), fixer=None, parallel_workers=2).run(
        dense_pages(tmp_path, 2), tmp_path / "out", events.append)
    assert result.ok and rec.page_calls == []
    for page in (1, 2):
        assert f"page {page} strip 1 of" in result.tex
    assert [e["ok"] for e in events if e["type"] == "prefetched"] == [True, True]


def test_only_the_first_strip_gets_the_previous_pages_context(tmp_path):
    rec = StripRecognizer()
    DocumentPipeline(rec, FakeCompiler(), fixer=None).run(
        dense_pages(tmp_path, 2), tmp_path / "out")
    later = [c for c in rec.strip_calls if "p0002" in c["path"].name]
    assert [c["tail"] != "" for c in sorted(later, key=lambda c: c["index"])][1:] == \
        [False] * (len(later) - 1)
    assert next(c for c in later if c["index"] == 1)["tail"] != ""


def test_a_read_with_skipped_lines_is_read_again_from_a_changed_image(tmp_path):
    """The endpoint answers an identical request identically, skipped lines and
    all, so the re-read goes out as a changed image."""
    rec = StripRecognizer(short={2: 1})
    events = []
    result = DocumentPipeline(rec, FakeCompiler(), fixer=None).run(
        dense_pages(tmp_path), tmp_path / "out", events.append)
    assert result.ok
    second = [c for c in rec.strip_calls if c["index"] == 2]
    assert [c["variant"] for c in second] == [0, 1]
    with Image.open(second[0]["path"]) as plain, Image.open(second[1]["path"]) as changed:
        assert changed.size != plain.size
    assert "strip 2 of" in result.tex and ", line 2:" in result.tex.split("strip 2 of")[-1]
    assert [e["reason"] for e in events if e["type"] == "strip_reread"][0].endswith(
        "lines skipped")


def test_when_every_read_skips_lines_the_best_one_is_kept(tmp_path):
    rec = StripRecognizer(short={2: 99})
    events = []
    result = DocumentPipeline(rec, FakeCompiler(), fixer=None).run(
        dense_pages(tmp_path), tmp_path / "out", events.append)
    assert result.ok and "strip 2 of" in result.tex
    assert [e for e in events if e["type"] == "strip_doubtful" and e["strip"] == 2]


def test_a_strip_that_never_reads_leaves_a_visible_gap_not_a_lost_page(tmp_path):
    rec = StripRecognizer(fail={3: 99})
    events = []
    result = DocumentPipeline(rec, FakeCompiler(), fixer=None, max_retries=3).run(
        dense_pages(tmp_path), tmp_path / "out", events.append)
    assert result.ok
    assert "could not be read" in result.tex
    assert "strip 2 of" in result.tex and "strip 4 of" in result.tex
    assert [e for e in events if e["type"] == "strip_unreadable" and e["strip"] == 3]
