from __future__ import annotations

from PIL import Image, ImageDraw

from n2lh.pipeline.segment import segment_lines


def make_page() -> Image.Image:
    """White 240x620 page with two text bands and one tall-glyph 'math' band."""
    img = Image.new("L", (240, 620), 255)
    d = ImageDraw.Draw(img)
    black = 0

    def small_band(y: int) -> None:
        for i in range(8):
            x = 20 + i * 25
            d.rectangle([x, y, x + 14, y + 12], fill=black)  # 12px 'glyphs'

    small_band(20)    # text band 1
    small_band(120)   # text band 2
    # math band: tall fraction bar glyph + small glyphs around it
    d.rectangle([100, 260, 106, 350], fill=black)            # 90px tall bar
    for i, x in enumerate([40, 60, 140, 160]):
        d.rectangle([x, 295, x + 12, 307], fill=black)
    small_band(480)   # text band 3
    return img


def test_segment_lines_counts_bands():
    regions = segment_lines(make_page())
    # four distinct bands expected
    assert len(regions) == 4
    ys = [r.bbox[1] for r in regions]
    assert ys == sorted(ys)


def test_segment_lines_classifies_tall_glyphs_as_math():
    regions = segment_lines(make_page())
    kinds = [r.kind for r in regions]
    assert kinds.count("math") == 1
    math_region = next(r for r in regions if r.kind == "math")
    assert math_region.bbox[1] > 200  # it is the tall-bar band
    assert math_region.image is not None
    assert math_region.image.width > 0


def test_segment_lines_empty_page():
    img = Image.new("L", (100, 100), 255)
    assert segment_lines(img) == []
