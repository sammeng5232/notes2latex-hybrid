"""Ingestion must keep colored ink and keep grayscale pages grayscale.

A real one-page summary carried red strikes over margin marks and blue
underlines on defined terms; ingestion flattened every page to "L", so the
recognizer never saw a single color and was blamed for not transcribing them.
"""

from __future__ import annotations

from PIL import Image

from n2lh.pipeline.ingest import _has_color_ink, _save_page, color_ink_names


def _page(background, ink, size=(400, 600)):
    img = Image.new("RGB", size, background)
    px = img.load()
    for x in range(40, 300):
        for y in range(100, 140):
            px[x, y] = ink
    return img


def test_a_page_with_red_ink_keeps_its_color(tmp_path):
    # Red pen stroke (real page: mean RGB (255,78,225)) beside black text.
    img = _page((255, 255, 255), (40, 40, 40))
    px = img.load()
    for x in range(40, 90):
        for y in range(200, 204):
            px[x, y] = (255, 78, 225)
    assert _has_color_ink(img)
    path = _save_page(img, tmp_path, 1)
    with Image.open(path) as saved:
        assert saved.mode == "RGB"
        assert saved.convert("RGB").getpixel((60, 202)) == (255, 78, 225)


def test_a_green_pen_counts_as_color_too(tmp_path):
    img = _page((255, 255, 255), (40, 40, 40))
    px = img.load()
    for x in range(40, 90):
        for y in range(200, 204):
            px[x, y] = (91, 164, 128)
    assert _has_color_ink(img)
    path = _save_page(img, tmp_path, 1)
    with Image.open(path) as saved:
        assert saved.mode == "RGB"


def test_a_highlighter_swipe_counts_as_color(tmp_path):
    # Bright marker over white paper: light, but saturated and not paper-white.
    img = _page((255, 255, 255), (40, 40, 40))
    px = img.load()
    for x in range(40, 300):
        for y in range(200, 212):
            px[x, y] = (255, 255, 120)
    assert _has_color_ink(img)


def test_a_black_ink_page_stays_grayscale(tmp_path):
    img = _page((255, 255, 255), (30, 30, 30))
    assert not _has_color_ink(img)
    path = _save_page(img, tmp_path, 1)
    with Image.open(path) as saved:
        assert saved.mode == "L"


def test_tinted_paper_is_not_mistaken_for_color(tmp_path):
    # A yellowed scan: warm background, dark gray ink, no colored pen.
    img = _page((255, 250, 235), (60, 58, 52))
    assert not _has_color_ink(img)
    path = _save_page(img, tmp_path, 1)
    with Image.open(path) as saved:
        assert saved.mode == "L"


def test_a_grayscale_input_image_is_accepted(tmp_path):
    # Image inputs (photos, scans) may already be single-channel.
    img = Image.new("L", (400, 600), 255)
    for x in range(40, 300):
        for y in range(100, 140):
            img.putpixel((x, y), 30)
    path = _save_page(img, tmp_path, 1)
    with Image.open(path) as saved:
        assert saved.mode == "L"


def test_a_stray_few_colored_pixels_do_not_flip_the_page(tmp_path):
    # Dust or a JPEG artifact: a couple of saturated specks are not content.
    img = _page((255, 255, 255), (30, 30, 30))
    px = img.load()
    for x, y in [(50, 50), (51, 50), (52, 51), (200, 300)]:
        px[x, y] = (255, 78, 225)
    assert not _has_color_ink(img)


# --------------------------------------------------------- naming the colors
def test_the_color_families_are_named():
    """The recognizer is told WHAT colors to look for: a real title written in
    magenta and green pen came out black when the directive was generic."""
    img = _page((255, 255, 255), (40, 40, 40))
    px = img.load()
    for x in range(40, 140):           # magenta pen (real page: (255,78,225))
        for y in range(200, 210):
            px[x, y] = (255, 78, 225)
    for x in range(160, 260):          # green pen (real page: (91,164,128))
        for y in range(200, 210):
            px[x, y] = (91, 164, 128)
    assert color_ink_names(img) == ["pink", "green"]   # family order, not sorted


def test_a_single_pen_family_is_named_alone():
    img = _page((255, 255, 255), (40, 40, 40))
    px = img.load()
    for x in range(40, 200):
        for y in range(200, 210):
            px[x, y] = (30, 60, 200)
    assert color_ink_names(img) == ["blue"]


def test_a_grayscale_page_has_no_color_names():
    img = _page((255, 255, 255), (30, 30, 30))
    assert color_ink_names(img) == []


# ------------------------------------------------------------- region crops
def test_a_color_region_crop_excludes_neighbouring_black_text():
    """A wide crop of a green margin strip pulled in the beginnings of the
    black body lines, and the model then marked those green: the crop must
    contain nothing the colored pen did not write."""
    from n2lh.pipeline.ingest import colored_ink_regions

    im = Image.new("RGB", (1000, 1400), (255, 255, 255))
    px = im.load()
    for x in range(50, 200):            # green margin marks
        for y in range(100, 400):
            px[x, y] = (91, 164, 128)
    for x in range(260, 600):           # black body text beside them
        for y in range(100, 1200):
            px[x, y] = (40, 40, 40)
    regions = colored_ink_regions(im)
    assert len(regions) == 1
    box, names = regions[0]
    assert names == ["green"]
    assert box[2] < 260, "the black body text stays out of the color crop"
    assert box[0] < 50 and box[1] < 100, "the marks themselves are inside"


def test_a_color_crop_whitens_the_black_text_running_through_it():
    """The margin marks sit right on the beginnings of the body lines; a plain
    crop showed the model black words and it marked those green."""
    from n2lh.pipeline.ingest import color_only_crop

    im = Image.new("RGB", (300, 200), (255, 255, 255))
    px = im.load()
    for x in range(10, 60):            # green margin mark
        for y in range(50, 90):
            px[x, y] = (91, 164, 128)
    for x in range(40, 220):           # black body line through the same box
        for y in range(62, 74):
            px[x, y] = (30, 30, 30)
    crop = color_only_crop(im, (0, 0, 250, 150))
    px = crop.load()
    assert px[20, 55] == (91, 164, 128), "the green stroke itself survives"
    assert px[150, 68] == (255, 255, 255), "black text far from green is gone"
