"""Cutting a dense page into full-resolution strips."""

from __future__ import annotations

from PIL import Image, ImageDraw

from n2lh.pipeline.figures import FIGBOX
from n2lh.pipeline.tiles import (STRIP_VARIANTS, Strip, count_lines, join_strips, judge_strip,
                                 line_pitch, lines_as_paragraphs, strip_variant, unpad_figbox,
                                 needs_strips, plan_strips, planning_mask, protected_bands,
                                 remap_figbox, tabularize_wraptables, tidy_strip,
                                 wraptable_after_title)


def lined_page(width=3508, height=4961, pitch=50, glyph=30, left=0.05, right=0.95,
               gaps=()):
    """A page of 'handwriting': rows of short strokes at a fixed line pitch."""
    im = Image.new("L", (width, height), 255)
    d = ImageDraw.Draw(im)
    y = pitch
    while y + glyph < height - pitch:
        if not any(a <= y <= b for a, b in gaps):
            for x in range(int(width * left), int(width * right), 70):
                d.rectangle([x, y, x + 40, y + glyph], fill=0)
        y += pitch
    return im.point(lambda v: 255 if v < 145 else 0)


# ------------------------------------------------------------ measuring density
def test_the_pitch_of_close_lines_is_measured_as_lines_not_paragraphs():
    dark = lined_page(pitch=50, glyph=30)
    assert 45 <= line_pitch(dark) <= 55


def test_a_dense_sheet_is_split_and_a_sparse_one_is_not():
    """Measured on real pages: the dense summaries sit at 28-30px per line once
    downscaled to 1600px, the topology notes at 65-75px."""
    assert needs_strips(lined_page(pitch=90))          # 90 * 1600/4961 = 29px
    assert not needs_strips(lined_page(pitch=230))     # 230 * 1600/4961 = 74px


def test_a_blank_page_is_not_split():
    blank = Image.new("L", (1000, 1400), 0)
    assert line_pitch(blank) is None
    assert not needs_strips(blank)


# -------------------------------------------------------------- where to cut
def test_every_cut_goes_through_an_empty_row():
    dark = lined_page(pitch=50, glyph=30)
    ink = list(dark.resize((1, dark.size[1]), Image.BOX).tobytes())
    strips = plan_strips(dark)
    assert len(strips) >= 2
    for s in strips[1:]:
        assert ink[s.top] == 0, f"cut at {s.top} passes through ink"


def test_the_strips_cover_the_page_without_gaps_or_overlap():
    dark = lined_page(pitch=50)
    strips = plan_strips(dark)
    assert strips[0].top == 0 and strips[-1].bottom == dark.size[1]
    for a, b in zip(strips, strips[1:]):
        assert a.bottom == b.top
    assert [s.index for s in strips] == list(range(1, len(strips) + 1))


def test_a_strip_holds_a_handful_of_lines():
    """Long strips get lines skipped (five theorems at a time from 16-line
    strips of the real page), so a strip holds about six counted lines."""
    dark = lined_page(pitch=60)                       # ~80 lines
    n = len(plan_strips(dark))
    assert 11 <= n <= 16


def test_a_table_is_never_cut_through():
    dark = lined_page(pitch=50)
    table = (1500, 2600)
    for s in plan_strips(dark, keep_whole=[table])[1:]:
        assert not (table[0] < s.top < table[1]), f"cut at {s.top} splits the table"


def test_a_page_without_measurable_lines_stays_whole():
    blank = Image.new("L", (1000, 1400), 0)
    assert plan_strips(blank) == [Strip(1, 0, 1400)]


# ---------------------------------------------------------- figures in strips
def test_a_figbox_in_a_strip_is_moved_into_page_coordinates():
    """The model saw only the strip, so its 0..1000 is the strip's height."""
    strip = Strip(2, 1000, 2000)                     # rows 1000-2000 of a 4000px page
    out = remap_figbox("a \\figbox{100}{0}{900}{1000} b", strip, 4000)
    x0, y0, x1, y1 = (int(g) for g in FIGBOX.search(out).groups())
    assert (x0, x1) == (100, 900)                    # full-width strips: x unchanged
    assert (y0, y1) == (250, 500)                    # 1000/4000 .. 2000/4000


def test_a_strip_without_figures_is_unchanged():
    assert remap_figbox("plain", Strip(1, 0, 10), 100) == "plain"


# ----------------------------------------------------------------- joining
def test_strips_are_joined_in_order_as_paragraphs():
    assert join_strips(["top\n", "", "  middle ", "bottom"]) == "top\n\nmiddle\n\nbottom"


def test_a_figure_split_by_a_cut_is_joined_back_into_one():
    """Cuts go through blank rows, and a drawing has blank rows inside it."""
    strips = [Strip(1, 0, 1000), Strip(2, 1000, 2000)]
    parts = ["text \\figbox{100}{300}{600}{498}",       # touches the cut at 500
             "\\figbox{120}{502}{580}{700} more"]
    out = join_strips(parts, strips, 2000)
    boxes = [tuple(int(g) for g in m.groups()) for m in FIGBOX.finditer(out)]
    assert boxes == [(100, 300, 600, 700)]
    assert "text" in out and "more" in out


def test_figures_that_merely_sit_in_neighbouring_strips_stay_separate():
    strips = [Strip(1, 0, 1000), Strip(2, 1000, 2000)]
    parts = ["\\figbox{100}{100}{300}{200}", "\\figbox{100}{800}{300}{900}"]
    assert len(list(FIGBOX.finditer(join_strips(parts, strips, 2000)))) == 2


# ------------------------------------------------------------ what is protected
def test_colored_ink_counts_as_ink_when_choosing_cuts():
    """Pink ink is lighter than the dark threshold; a darkness mask alone saw
    the glossary's rows as blank paper."""
    page = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(page).rectangle([50, 100, 350, 130], fill=(235, 120, 170))
    mask = planning_mask(page)
    rows = list(mask.resize((1, 400), Image.BOX).tobytes())
    assert rows[115] > 0 and rows[300] == 0


def test_a_colored_block_is_protected_from_cuts():
    page = Image.new("RGB", (2000, 3000), "white")
    d = ImageDraw.Draw(page)
    for y in range(100, 2900, 60):
        for x in range(200, 1800, 70):
            d.rectangle([x, y, x + 40, y + 30], fill="black")
    d.rectangle([1500, 1200, 1900, 1700], fill=(40, 150, 60))    # a green margin label
    mask = planning_mask(page)
    bands = protected_bands(page, mask, line_pitch(mask))
    assert any(a <= 1200 and b >= 1700 for a, b in bands)
    for s in plan_strips(mask, keep_whole=bands)[1:]:
        assert not (1200 <= s.top <= 1700)


# ----------------------------------------------------------- line structure
def test_each_handwritten_line_becomes_its_own_paragraph():
    assert lines_as_paragraphs("定理 1 a\n定理 2 b") == "定理 1 a\n\n定理 2 b"


def test_lines_inside_environments_and_displays_are_left_alone():
    src = ("\\begin{tabular}{ll}\nx & y \\\\\nz & w \\\\\n\\end{tabular}\n"
           "\\[\nx=1\n\\]\nd \\\\\ne")
    out = lines_as_paragraphs(src)
    assert "x & y \\\\\nz & w \\\\\n\\end{tabular}" in out
    assert "\\[\nx=1\n\\]" in out
    assert "d \\\\\ne" in out                        # an explicit break is kept as is
    assert "\\end{tabular}\n\n\\[" in out


def test_a_spaced_line_break_is_not_taken_for_display_math():
    out = lines_as_paragraphs("a \\\\[2pt]\nb\nc")
    assert out.endswith("b\n\nc")


# ------------------------------------------------------------- corner tables
def test_a_glossary_written_as_spaced_lines_becomes_a_tabular():
    src = ("\\begin{wraptable}{r}{0.4\\textwidth}\n中英词汇对照表\n"
           "振幅 \\quad Oscillation\n测度 \\quad Measure\n\\end{wraptable}")
    out = tabularize_wraptables(src)
    assert "\\begin{tabular}{ll}" in out
    assert "振幅 & Oscillation \\\\" in out and "测度 & Measure \\\\" in out
    assert out.index("中英词汇对照表") < out.index("\\begin{tabular}")


def test_a_corner_table_is_moved_up_beside_the_title():
    src = ("\\section*{实变函数总结}\n\n第一段\n\n"
           "\\begin{wraptable}{r}{0.4\\textwidth}\nT\n\\end{wraptable}\n\n第二段")
    out = wraptable_after_title(src)
    assert out.index("实变函数总结") < out.index("wraptable") < out.index("第一段")


# ----------------------------------------------------------- judging reads
def theorem(k: int) -> str:
    return f"定理 1.{k}. 设 $x_{k} \\in E$ 可测, 则 $m(E_{k}) \\le m^*(E)$ 且 $E_{k}$ 为闭集."


def test_a_complete_read_passes():
    read = "\n".join(theorem(k) for k in range(10))
    assert judge_strip(read, 10)[0]


def test_a_read_that_skipped_lines_fails():
    """Real case: two of three reads of a 17-line strip kept 10 and 12 lines."""
    read = "\n".join(theorem(k) for k in range(6))
    ok, _, why = judge_strip(read, 10)
    assert not ok and "skipped" in why


def test_fluent_garbage_fails():
    """Verbatim shape of answers the real endpoint gave for strips of the page."""
    garbage = "UL\n\n se五代 European European s\n\n se五代se\n\n Portuguese ULe\n\n ULs se\n\n Indonesian ULULULul"
    assert not judge_strip(garbage, 7)[0]
    assert not judge_strip("httpdef", 11)[0]
    assert not judge_strip("", 5)[0]


def test_a_margin_label_line_is_not_counted_as_content():
    read = "\\textbf{② 测度}\n" + "\n".join(f"定理 2.{k}. $m^*E=0$" for k in range(5))
    ok, _, _ = judge_strip(read, 7)
    assert not ok                       # 5 content lines for a 7-line strip


def test_a_better_read_scores_higher():
    short = "\n".join(theorem(k) for k in range(5))
    full = "\n".join(theorem(k) for k in range(10))
    assert judge_strip(full, 10)[1] > judge_strip(short, 10)[1] > 0
    assert judge_strip("httpdef", 10)[1] == 0          # garbage is never kept


def test_the_lines_of_a_strip_are_counted_from_its_ink():
    dark = lined_page(width=2000, height=600, pitch=60)
    assert 8 <= count_lines(dark, 60) <= 9


# ------------------------------------------------------------------ re-reads
def test_a_re_read_changes_the_image_but_not_what_it_shows():
    strip = Image.new("RGB", (1000, 200), "white")
    ImageDraw.Draw(strip).rectangle([100, 50, 300, 80], fill="black")
    assert strip_variant(strip, 0).size == (1000, 200)
    padded = strip_variant(strip, 3)
    assert padded.size == (1000 + 2 * 96, 200 + 2 * 96)
    assert padded.getpixel((0, 0)) == (255, 255, 255)
    assert strip_variant(strip, 1).size == (600, 120)
    # every variant differs from the plain strip
    assert len({strip_variant(strip, v).size for v in range(len(STRIP_VARIANTS))}) \
        == len(STRIP_VARIANTS)


def test_figure_coordinates_read_from_a_padded_strip_map_back_onto_it():
    pad = STRIP_VARIANTS[3][0]           # 96px of white, full size
    w, h = 1000, 200
    # a figure covering the whole strip, as the model sees it inside the frame
    x0 = round(pad / (w + 2 * pad) * 1000)
    y0 = round(pad / (h + 2 * pad) * 1000)
    out = unpad_figbox(f"\\figbox{{{x0}}}{{{y0}}}{{{1000 - x0}}}{{{1000 - y0}}}", 3, w, h)
    box = [int(g) for g in FIGBOX.search(out).groups()]
    assert all(abs(a - b) <= 3 for a, b in zip(box, [0, 0, 1000, 1000]))
    assert unpad_figbox("\\figbox{1}{2}{3}{4}", 0, w, h) == "\\figbox{1}{2}{3}{4}"


def test_tidy_strip_remaps_figures_and_splits_lines():
    out = tidy_strip("a\nb \\figbox{0}{0}{1000}{1000}", Strip(2, 500, 1000), 1000)
    assert out == "a\n\nb \\figbox{0}{500}{1000}{1000}"


def test_adjacent_inline_formulas_do_not_stop_the_paragraph_split():
    src = "定理 5.12. $\\big($$f$ 连续$\\big)$.\n定理 5.13 (FTC). x\n定理 5.14. y"
    assert lines_as_paragraphs(src).count("\n\n") == 2

