from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Optional

import pytest
from PIL import Image, ImageDraw

from n2lh.compiler.latex import LatexCompiler
from n2lh.pipeline.figures import (Box, FIGBOX, iou, line_height, parse_boxes, refine,
                                   render_figures)
from n2lh.pipeline.graph import DocumentPipeline
from n2lh.recognition.base import PageImage, Recognizer, TranscribeResult
from n2lh.recognition.prompts import PREAMBLE_TEX, make_preamble

HAS_TEX = shutil.which("latexmk") is not None or shutil.which("pdflatex") is not None


def page_with_figure(tmp_path: Path, size=(1000, 2000), rect=(200, 800, 700, 1200)) -> Path:
    """A white page with one black filled rectangle standing in for a drawing."""
    im = Image.new("RGB", size, "white")
    ImageDraw.Draw(im).rectangle(rect, fill="black")
    path = tmp_path / "page.png"
    im.save(path)
    return path


# ------------------------------------------------------------------ box parsing
def test_parse_boxes_accepts_fenced_json_and_alternate_key():
    text = '```json\n[{"bbox": [10, 20, 300, 400]}, {"bbox_2d": [1, 2, 3, 4], "label": "graph"}]\n```'
    assert parse_boxes(text) == [Box(10, 20, 300, 400), Box(1, 2, 3, 4)]


def test_parse_boxes_clamps_orders_and_survives_garbage():
    assert parse_boxes('[{"bbox": [900, 500, -20, 1500]}]') == [Box(0, 500, 900, 1000)]
    for junk in ("", "no figures", "[]", "[{'bbox': broken}]", '[{"bbox": [1,2]}]'):
        assert parse_boxes(junk) == []


def test_iou_and_refine_match_by_overlap_and_keep_unmatched_inline_boxes():
    coarse = [Box(0, 500, 1000, 800), Box(300, 900, 700, 950)]
    located = [Box(26, 498, 956, 829)]
    assert iou(coarse[0], located[0]) > 0.8
    refined = refine(coarse, located)
    assert refined[0] == located[0]            # coarse box replaced by the tight one
    assert refined[1] == coarse[1]             # nothing overlapped it: keeps its own


def test_each_located_box_is_used_once():
    inline = [Box(100, 100, 500, 500), Box(110, 110, 510, 510)]
    located = [Box(105, 105, 505, 505)]
    refined = refine(inline, located)
    assert refined.count(located[0]) == 1


# ------------------------------------------------------------------- cropping
def test_render_crops_the_figure_and_writes_an_includegraphics(tmp_path):
    page = page_with_figure(tmp_path)
    latex = "Before.\n\n\\figbox{200}{400}{700}{600}\n\nAfter."
    out = render_figures(latex, page, tmp_path / "figures", 7)
    assert "\\figbox" not in out
    assert "\\includegraphics[width=" in out and "{p0007_f1.png}" in out
    assert out.startswith("Before.") and out.rstrip().endswith("After.")
    crop = Image.open(tmp_path / "figures" / "p0007_f1.png")
    # The drawing is 500x400 px; the crop is snapped onto it plus a small margin.
    assert 500 <= crop.width <= 540 and 400 <= crop.height <= 440


def test_locator_refines_a_coarse_box(tmp_path):
    page = page_with_figure(tmp_path)
    coarse = "\\figbox{100}{350}{800}{650}"                 # too big, but the same drawing
    tight = [(200, 400, 700, 600)]
    out = render_figures(coarse, page, tmp_path / "f", 1, locator=lambda: tight)
    assert "{p0001_f1.png}" in out
    crop = Image.open(tmp_path / "f" / "p0001_f1.png")
    assert crop.width < 600                               # tight box (~530px), not the coarse one (~730px)


def test_located_boxes_win_even_when_the_inline_ones_are_nowhere_near():
    """What the model writes while transcribing is not a measurement: on a real page
    its four boxes were evenly spaced round numbers to the left of the drawings, all
    of them below any sane overlap threshold. Equal counts => positional pairing."""
    inline = [Box(200, 200, 400, 350), Box(200, 450, 400, 600), Box(200, 650, 400, 800)]
    located = [Box(345, 213, 510, 300), Box(375, 470, 520, 600), Box(375, 650, 520, 780)]
    assert max(iou(a, b) for a, b in zip(inline, located)) < 0.3
    assert refine(inline, located) == located


def test_located_boxes_are_put_back_into_reading_order():
    inline = [Box(100, 100, 200, 200), Box(100, 500, 200, 600)]
    assert refine(inline, [Box(0, 500, 90, 600), Box(0, 100, 90, 200)]) == [
        Box(0, 100, 90, 200), Box(0, 500, 90, 600)]


def test_when_the_counts_differ_each_marker_takes_the_box_at_its_height():
    inline = [Box(200, 100, 400, 200), Box(200, 600, 400, 700)]
    # Only the second figure was located; the first keeps its own coordinates.
    assert refine(inline, [Box(300, 620, 500, 680)]) == [inline[0], Box(300, 620, 500, 680)]


def test_a_located_box_from_a_different_part_of_the_page_is_not_used():
    """Only when the counts disagree: with equal counts the two lists describe the
    same figures, however far apart the model's own guess was."""
    inline = [Box(200, 100, 400, 200), Box(200, 300, 400, 400)]
    far = [Box(200, 800, 400, 900)]                  # matches neither marker's height
    assert refine(inline, far, max_gap=150) == inline


def test_locator_failure_or_nothing_falls_back_to_the_inline_box(tmp_path):
    page = page_with_figure(tmp_path)
    latex = "\\figbox{200}{400}{700}{600}"

    def boom():
        raise RuntimeError("endpoint down")

    for locator in (boom, lambda: None, lambda: []):
        out = render_figures(latex, page, tmp_path / "f", 1, locator=locator)
        assert "\\includegraphics" in out


def test_unusable_box_becomes_a_visible_omission_not_a_crash(tmp_path):
    page = page_with_figure(tmp_path)
    out = render_figures("\\figbox{500}{500}{503}{503}", page, tmp_path / "f", 1)
    assert out == "\\textit{[figure omitted]}"


def test_no_figbox_means_no_work(tmp_path):
    assert render_figures("plain text", tmp_path / "missing.png", tmp_path / "f", 1) == "plain text"


def test_several_figures_get_distinct_files_in_order(tmp_path):
    page = page_with(tmp_path, [(150, 250, 450, 550), (200, 800, 700, 1200)])
    latex = "\\figbox{100}{100}{500}{300}\ntext\n\\figbox{200}{400}{700}{600}"
    out = render_figures(latex, page, tmp_path / "f", 3)
    assert out.index("p0003_f1.png") < out.index("text") < out.index("p0003_f2.png")
    assert (tmp_path / "f" / "p0003_f1.png").exists() and (tmp_path / "f" / "p0003_f2.png").exists()


def test_figbox_pattern_tolerates_spaces():
    assert FIGBOX.search("\\figbox{ 1 }{2}{ 3}{4 }")


# ------------------------------------------------------------------- preamble
def test_the_preamble_can_carry_the_cropped_figures():
    """Page size and font size are covered in test_page_setup.py; what matters
    here is that a page full of \\includegraphics compiles."""
    assert PREAMBLE_TEX == make_preamble()
    for needed in ("graphicx", "\\graphicspath", "\\figbox", "\\parskip"):
        assert needed in PREAMBLE_TEX


# ----------------------------------------------------- through the pipeline
class FigureRecognizer(Recognizer):
    name = "fig"

    def __init__(self, locate: Optional[List[tuple]] = None):
        self._locate = locate
        self.locate_calls = 0

    def transcribe(self, page, context_tail, open_environments, guidance=None):
        return TranscribeResult(
            latex="Eg. see the picture.\n\n\\figbox{0}{300}{1000}{700}\n\nDone.", engine=self.name)

    def locate_figures(self, page):
        self.locate_calls += 1
        return self._locate


class OkCompiler:
    def compile(self, tex, workdir):
        from n2lh.compiler.latex import CompileResult
        Path(workdir).mkdir(parents=True, exist_ok=True)
        (Path(workdir) / "document.pdf").write_bytes(b"%pdf")
        return CompileResult(ok=True, pdf_path=Path(workdir) / "document.pdf")


def test_pipeline_turns_figboxes_into_files_beside_the_output(tmp_path):
    page_png = page_with_figure(tmp_path)
    rec = FigureRecognizer(locate=[(200, 400, 700, 600)])
    pipe = DocumentPipeline(rec, OkCompiler(), fixer=None, max_retries=0)
    result = pipe.run([PageImage(1, page_png)], tmp_path / "out")
    assert (tmp_path / "out" / "figures" / "p0001_f1.png").exists()
    assert "\\includegraphics" in result.tex and "\\figbox{0}" not in result.tex
    assert rec.locate_calls == 1                       # one dedicated request for that page


def test_pipeline_prefetch_also_crops_figures(tmp_path):
    pngs = []
    for i in (1, 2):
        d = tmp_path / f"p{i}"
        d.mkdir()
        pngs.append(page_with_figure(d))
    pipe = DocumentPipeline(FigureRecognizer(locate=[(200, 400, 700, 600)]), OkCompiler(),
                            fixer=None, max_retries=0, parallel_workers=2)
    result = pipe.run([PageImage(1, pngs[0]), PageImage(2, pngs[1])], tmp_path / "out")
    assert (tmp_path / "out" / "figures" / "p0001_f1.png").exists()
    assert (tmp_path / "out" / "figures" / "p0002_f1.png").exists()
    assert result.tex.count("\\includegraphics") == 2


def test_pipeline_without_a_locator_uses_the_inline_box(tmp_path):
    page_png = page_with_figure(tmp_path)

    class NoLocate(FigureRecognizer):
        locate_figures = Recognizer.locate_figures      # base behaviour: None

    pipe = DocumentPipeline(NoLocate(), OkCompiler(), fixer=None, max_retries=0)
    pipe.run([PageImage(1, page_png)], tmp_path / "out")
    assert (tmp_path / "out" / "figures" / "p0001_f1.png").exists()


@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
def test_figures_resolve_in_the_real_per_page_and_final_compiles(tmp_path):
    """The per-page compile runs in <pages>/.compile-pNNNN and the final one in out/:
    the preamble's \\graphicspath has to find figures/ from both."""
    job = tmp_path / "job"
    pages = job / "pages"
    pages.mkdir(parents=True)
    page_png = page_with_figure(pages)
    pipe = DocumentPipeline(FigureRecognizer(locate=[(200, 400, 700, 600)]),
                            LatexCompiler(timeout=120), fixer=None, max_retries=0)
    result = pipe.run([PageImage(1, page_png)], job / "out")
    assert result.pages[0].compile_ok, result.pages[0].errors
    assert result.ok and result.pdf_path is not None and result.pdf_path.exists()

# ------------------------------------------ figure marker inside display math
def test_a_marker_the_model_put_inside_display_math_is_taken_out_of_it():
    from n2lh.pipeline.figures import unwrap_math
    assert unwrap_math("a\n\\[\n\\figbox{1}{2}{3}{4}\n\\]\nb") == "a\n\\figbox{1}{2}{3}{4}\nb"
    assert unwrap_math("$$ \\figbox{1}{2}{3}{4} $$") == "\\figbox{1}{2}{3}{4}"
    assert unwrap_math("\\begin{equation*}\\figbox{1}{2}{3}{4}\\end{equation*}") == "\\figbox{1}{2}{3}{4}"
    two = unwrap_math("\\[ \\figbox{1}{2}{3}{4} \\figbox{5}{6}{7}{8} \\]")
    assert two == "\\figbox{1}{2}{3}{4}\n\n\\figbox{5}{6}{7}{8}"


def test_real_math_next_to_a_marker_is_never_unwrapped():
    from n2lh.pipeline.figures import unwrap_math
    src = "\\[ x = 1 \\]\n\\figbox{1}{2}{3}{4}\n\\[ y = 2 \\]"
    assert unwrap_math(src) == src
    mixed = "\\[ x = \\figbox{1}{2}{3}{4} \\]"
    assert unwrap_math(mixed) == mixed


def test_render_never_puts_a_picture_inside_math_mode(tmp_path):
    page = page_with_figure(tmp_path)
    out = render_figures("Def.\n\\[\n\\figbox{200}{400}{700}{600}\n\\]\nDone.", page, tmp_path / "f", 1)
    assert "\\[" not in out and "\\]" not in out
    assert out.count("\\includegraphics") == 1


@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX toolchain installed")
def test_display_wrapped_figure_compiles_after_rendering(tmp_path):
    """Page 34 of the real run: `\\[ \\figbox \\]` became `\\[ \\begin{center}..` and failed
    with `Missing $ inserted` on every repair."""
    from n2lh.pipeline.assembler import build_document
    job = tmp_path / "job"
    job.mkdir()
    page = page_with_figure(job)
    body = render_figures("Def. a map.\n\n\\[\n\\figbox{200}{400}{700}{600}\n\\]\n\nEg. done.",
                          page, job / "out" / "figures", 34)        # where the pipeline puts them
    result = LatexCompiler(timeout=120).compile(build_document(body), job / "out")
    assert result.ok, result.errors

# --------------------------------------------------------------- ink snapping
def page_with(tmp_path: Path, shapes, size=(1000, 2000), name="page.png") -> Path:
    """A white page carrying the given (x0, y0, x1, y1) filled rectangles."""
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    for rect in shapes:
        d.rectangle(rect, fill="black")
    path = tmp_path / name
    im.save(path)
    return path


def crop_box(tmp_path, latex, page, located=None, index=1):
    """Render one figure and return the crop's pixel size."""
    render_figures(latex, page, tmp_path / "f", index,
                   locator=(lambda: located) if located else None)
    return Image.open(tmp_path / "f" / ("p%04d_f1.png" % index)).size


def test_a_box_that_cuts_through_the_drawing_grows_to_contain_it(tmp_path):
    page = page_with(tmp_path, [(300, 900, 700, 1100)])
    # The model's box clips ~20px off each side of the drawing.
    w, h = crop_box(tmp_path, "\\figbox{320}{460}{680}{540}", page)
    assert 400 <= w <= 440 and 200 <= h <= 240       # the whole 400x200 drawing, plus margin


def test_a_box_far_larger_than_the_drawing_is_tightened_onto_it(tmp_path):
    page = page_with(tmp_path, [(400, 950, 600, 1050)])
    w, h = crop_box(tmp_path, "\\figbox{50}{300}{950}{700}", page)
    assert 200 <= w <= 240 and 100 <= h <= 140


def test_a_line_of_text_above_the_drawing_is_left_out(tmp_path):
    """The real failure: crops arrived with the tail of the previous sentence in them."""
    text_line = (150, 700, 850, 740)                 # thin and wide, like handwriting
    drawing = (300, 900, 700, 1400)
    page = page_with(tmp_path, [text_line, drawing])
    w, h = crop_box(tmp_path, "\\figbox{140}{340}{860}{720}", page)
    assert 400 <= w <= 440 and 500 <= h <= 540       # the drawing only


def test_growth_stops_before_the_next_line_of_handwriting(tmp_path):
    page = page_with(tmp_path, [(150, 600, 850, 640), (300, 900, 700, 1100)])
    w, h = crop_box(tmp_path, "\\figbox{300}{450}{700}{550}", page)
    assert h <= 260                                   # did not travel up to the text line


def test_a_figure_in_several_parts_keeps_all_of_them(tmp_path):
    """Two boxes and the arrow between them are one diagram, not a stray band."""
    page = page_with(tmp_path, [(200, 900, 400, 1100), (600, 900, 800, 1100),
                                (400, 990, 600, 1010)])
    w, h = crop_box(tmp_path, "\\figbox{190}{440}{810}{560}", page)
    assert 600 <= w <= 640 and 200 <= h <= 240


def test_labels_beside_a_diagram_are_kept(tmp_path):
    """A chart diagram carries `V` and `R^n` to the right of the box. Trimming
    columns as well as rows cut them off: that column is thin and, holding two
    labels at different heights, spans most of the crop -- indistinguishable from
    a line of writing by the same rule."""
    box = (300, 900, 700, 1300)
    label_top = (780, 910, 850, 960)
    label_bottom = (780, 1240, 860, 1290)
    page = page_with(tmp_path, [box, label_top, label_bottom])
    w, h = crop_box(tmp_path, "\\figbox{290}{440}{870}{660}", page)
    assert 550 <= w <= 590                            # 300..860 plus margin: labels kept


def test_a_caption_under_a_drawing_is_kept(tmp_path):
    """Short, so it is not the full-width band that marks a line of prose."""
    page = page_with(tmp_path, [(300, 900, 700, 1300), (420, 1360, 580, 1400)])
    w, h = crop_box(tmp_path, "\\figbox{290}{440}{710}{710}", page)
    assert 500 <= h <= 540                            # 900..1400 plus margin


def test_the_picture_is_sized_from_the_drawing_not_the_model_box(tmp_path):
    page = page_with(tmp_path, [(400, 950, 600, 1050)])
    out = render_figures("\\figbox{50}{300}{950}{700}", page, tmp_path / "f", 1)
    width = float(re.search(r"width=([0-9.]+)\\linewidth", out).group(1))
    assert 0.20 <= width <= 0.30                      # ~20% of the page, not the 90% box


def test_pictures_stay_under_the_text_width(tmp_path):
    page = page_with(tmp_path, [(20, 900, 980, 1200)])
    out = render_figures("\\figbox{0}{400}{1000}{650}", page, tmp_path / "f", 1)
    width = float(re.search(r"width=([0-9.]+)\\linewidth", out).group(1))
    assert width <= 0.78


def test_a_blank_area_is_reported_as_an_omitted_figure(tmp_path):
    page = page_with(tmp_path, [(100, 100, 200, 200)])
    out = render_figures("\\figbox{400}{700}{600}{800}", page, tmp_path / "f", 1)
    assert out == "\\textit{[figure omitted]}"

# ------------------------------------------------- telling writing from drawing
def writing_row(y, lines=1, x0=120, x1=880, h=110, pitch=170):
    """Marks that look like a line of handwriting: many short strokes across the page."""
    out = []
    for n in range(lines):
        top = y + n * pitch
        for x in range(x0, x1, 46):
            out.append((x, top, x + 30, top + h))
    return out


def test_the_page_tells_us_how_tall_a_line_of_its_handwriting_is(tmp_path):
    page = page_with(tmp_path, sum((writing_row(200 + i * 170) for i in range(8)), []),
                     size=(1000, 2000))
    dark = Image.open(page).convert("L").point(lambda v: 255 if v < 145 else 0)
    assert 100 <= line_height(dark) <= 130


def test_a_line_of_writing_above_the_drawing_is_dropped(tmp_path):
    shapes = sum((writing_row(100 + i * 170) for i in range(6)), [])     # a page of prose
    shapes += writing_row(1200)                                          # the stray line
    shapes.append((300, 1400, 700, 1800))                                # the drawing
    page = page_with(tmp_path, shapes, size=(1000, 2400))
    w, h = crop_box(tmp_path, "\\figbox{290}{490}{710}{760}", page)
    assert 400 <= w <= 440 and 400 <= h <= 440                           # the drawing alone


def test_writing_thicker_than_the_figure_does_not_become_the_figure(tmp_path):
    """The anchor is the thickest band that is not writing. Anchoring on the
    thickest band of any kind keeps the line of text -- taller than the strokes
    it sits above -- and trims the figure away instead."""
    shapes = sum((writing_row(100 + i * 170) for i in range(6)), [])
    shapes += writing_row(1200)                                          # a line of text
    shapes += [(150, 1400, 850, 1420), (150, 1500, 850, 1520)]           # the figure
    page = page_with(tmp_path, shapes, size=(1000, 2000))
    w, h = crop_box(tmp_path, "\\figbox{140}{590}{860}{780}", page)
    assert 600 <= w <= 740 and 120 <= h <= 160                           # strokes only


def test_long_strokes_are_not_mistaken_for_lines_of_writing(tmp_path):
    """A figure of two long horizontal rules spans the crop like text does, but a
    stroke is a fraction of a line's height."""
    shapes = sum((writing_row(100 + i * 170) for i in range(6)), [])
    shapes += [(150, 1300, 850, 1320), (150, 1450, 850, 1470)]
    page = page_with(tmp_path, shapes, size=(1000, 2000))
    w, h = crop_box(tmp_path, "\\figbox{140}{630}{860}{750}", page)
    assert 600 <= w <= 740 and 170 <= h <= 210                           # both rules kept


def test_a_box_that_lands_on_prose_is_reported_as_an_omission(tmp_path):
    """Seen in a real run: the locator put a box over two lines of text with the
    drawing just below it. A picture of a sentence is worse than saying so."""
    shapes = sum((writing_row(100 + i * 170) for i in range(8)), [])
    page = page_with(tmp_path, shapes, size=(1000, 2000))
    out = render_figures("\\figbox{100}{200}{900}{400}", page, tmp_path / "f", 1)
    assert out == "\\textit{[figure omitted]}"
    assert not (tmp_path / "f" / "p0001_f1.png").exists()