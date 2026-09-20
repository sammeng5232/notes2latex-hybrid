"""Re-reading a ruled table from a crop of the page."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw

from n2lh.pipeline.graph import DocumentPipeline
from n2lh.pipeline.tables import crop_band, find_table_band, replace_tabular
from n2lh.recognition.base import PageImage, Recognizer, TranscribeResult

PAGE = (1000, 1400)


def ruled_page(tmp_path: Path, rules=(200, 260, 320, 380, 440), width=0.9,
               extra=(), name="page.png") -> Path:
    """A page with horizontal rules, like a table's row separators."""
    im = Image.new("RGB", PAGE, "white")
    d = ImageDraw.Draw(im)
    x0, x1 = 0, int(PAGE[0] * width)
    for y in rules:
        d.rectangle([x0, y, x1, y + 3], fill="black")
    for rect in extra:
        d.rectangle(rect, fill="black")
    path = tmp_path / name
    im.save(path)
    return path


def dark_of(path: Path) -> Image.Image:
    return Image.open(path).convert("L").point(lambda v: 255 if v < 145 else 0)


# -------------------------------------------------------------- finding it
def test_the_rules_of_a_table_give_away_where_it_is(tmp_path):
    band = find_table_band(dark_of(ruled_page(tmp_path)))
    assert band is not None
    top, bottom = band
    assert top <= 200 and bottom >= 443


def test_a_page_with_no_table_is_left_alone(tmp_path):
    page = ruled_page(tmp_path, rules=(), extra=[(100, 100, 400, 140)])
    assert find_table_band(dark_of(page)) is None


def test_one_underlined_heading_is_not_a_table(tmp_path):
    assert find_table_band(dark_of(ruled_page(tmp_path, rules=(300,)))) is None


def test_a_rule_far_from_the_table_does_not_stretch_the_band(tmp_path):
    """An underlined heading at the top must not drag the crop up to it."""
    page = ruled_page(tmp_path, rules=(60, 700, 760, 820, 880, 940))
    top, bottom = find_table_band(dark_of(page))
    assert top > 300, "the lone rule at y=60 was swept into the band"
    assert bottom >= 943


def test_short_rules_are_not_counted(tmp_path):
    """A rule spanning a third of the page is a fraction bar, not a table."""
    assert find_table_band(dark_of(ruled_page(tmp_path, width=0.3))) is None


# ------------------------------------------------------------- cropping it
def test_the_crop_is_enlarged_so_it_survives_the_downscale(tmp_path):
    """The client sends at most 1600px; a band left at page width would arrive
    no bigger than it was inside the whole page, which is the whole problem."""
    page = ruled_page(tmp_path)
    image = Image.open(page).convert("RGB")
    out = crop_band(image, (200, 450), tmp_path / "band.png")
    crop = Image.open(out)
    assert crop.width >= 4000
    assert crop.height / crop.width == round(250 / PAGE[0], 6) or crop.height > 900


# -------------------------------------------------------------- swapping it
TABLE = ("\\begin{tabular}{|l|l|}\n\\hline\na & b \\\\\nc & d \\\\\n\\end{tabular}")
BETTER = ("\\begin{tabular}{|l|l|}\n\\hline\nhead1 & head2 \\\\\na & b, and more \\\\\n"
          "c & d \\\\\n\\end{tabular}")


def test_the_re_read_replaces_the_pages_own_table():
    page = "Before.\n\n" + TABLE + "\n\nAfter."
    out, swapped = replace_tabular(page, BETTER)
    assert swapped
    assert "and more" in out
    assert out.startswith("Before.") and out.rstrip().endswith("After.")


def test_a_re_read_that_lost_rows_is_discarded():
    page = "x\n" + BETTER
    short = "\\begin{tabular}{|l|l|}\na & b \\\\\n\\end{tabular}"
    assert replace_tabular(page, short) == (page, False)


def test_a_missing_header_row_is_tolerated():
    """The crop often omits the header; that is formatting, not lost content."""
    page = "x\n" + BETTER                       # 3 rows
    no_header = TABLE                            # 2 rows, one fewer
    out, swapped = replace_tabular(page, no_header)
    assert swapped and "head1" not in out


def test_a_page_with_two_tables_is_left_alone():
    page = TABLE + "\n\n" + TABLE
    assert replace_tabular(page, BETTER) == (page, False)


def test_nothing_happens_without_a_table_on_either_side():
    assert replace_tabular("no table here", BETTER)[1] is False
    assert replace_tabular("x\n" + TABLE, "the model returned prose")[1] is False


# ------------------------------------------------------ through the pipeline
class TableRecognizer(Recognizer):
    """Transcribes the page badly and the cropped table well, like the real one."""

    name = "tbl"

    def __init__(self, table: Optional[str] = BETTER) -> None:
        self._table = table
        self.table_calls: List[str] = []

    def transcribe(self, page, context_tail, open_environments, guidance=None):
        return TranscribeResult(latex="Def. a summary.\n\n" + TABLE, engine=self.name)

    def transcribe_table(self, image_path):
        self.table_calls.append(str(image_path))
        return self._table


class OkCompiler:
    def compile(self, tex, workdir):
        from n2lh.compiler.latex import CompileResult
        Path(workdir).mkdir(parents=True, exist_ok=True)
        (Path(workdir) / "document.pdf").write_bytes(b"%pdf")
        return CompileResult(ok=True, pdf_path=Path(workdir) / "document.pdf")


def run_page(tmp_path, rec, page_path):
    pipe = DocumentPipeline(rec, OkCompiler(), fixer=None, max_retries=0)
    return pipe.run([PageImage(1, page_path)], tmp_path / "out")


def test_a_page_with_a_table_is_re_read_once_and_the_better_table_wins(tmp_path):
    rec = TableRecognizer()
    result = run_page(tmp_path, rec, ruled_page(tmp_path))
    assert "and more" in result.tex
    assert len(rec.table_calls) == 1, "the table must not be re-read per attempt"


def test_a_page_without_a_table_costs_no_extra_call(tmp_path):
    rec = TableRecognizer()
    page = ruled_page(tmp_path, rules=(), extra=[(100, 100, 400, 140)])
    run_page(tmp_path, rec, page)
    assert rec.table_calls == []


def test_a_recognizer_that_cannot_re_read_leaves_the_page_as_it_is(tmp_path):
    rec = TableRecognizer(table=None)
    result = run_page(tmp_path, rec, ruled_page(tmp_path))
    assert "\\begin{tabular}" in result.tex


def test_a_failing_re_read_is_not_fatal(tmp_path):
    class Boom(TableRecognizer):
        def transcribe_table(self, image_path):
            raise RuntimeError("endpoint down")

    result = run_page(tmp_path, Boom(), ruled_page(tmp_path))
    assert "\\begin{tabular}" in result.tex     # the page's own table still stands
