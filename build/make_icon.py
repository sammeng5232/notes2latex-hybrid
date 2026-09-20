"""Generate build/icon.ico (simple branded icon) using Pillow."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "icon.ico"


def glyph_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (r"C:\Windows\Fonts\arialbd.ttf",
                      r"C:\Windows\Fonts\segoeuib.ttf",
                      r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(size: int) -> Image.Image:
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = max(1, s // 16)
    # dark rounded tile
    d.rounded_rectangle([m, m, s - m, s - m], radius=s // 6,
                        fill=(16, 20, 24, 255), outline=(76, 166, 255, 255), width=m)
    if s >= 48:
        f_big = glyph_font(int(s * 0.30))
        f_small = glyph_font(int(s * 0.20))
        d.text((s * 0.50, s * 0.36), "\u2211", font=f_big,      # sum sign
               fill=(230, 237, 243, 255), anchor="mm")
        d.text((s * 0.50, s * 0.70), "TeX", font=f_small,
               fill=(76, 166, 255, 255), anchor="mm")
    else:
        d.text((s * 0.5, s * 0.5), "T", font=glyph_font(int(s * 0.5)),
               fill=(230, 237, 243, 255), anchor="mm")
    return img


def main() -> None:
    sizes = [16, 24, 32, 48, 64, 128, 256]
    base = render(256)
    base.save(OUT, format="ICO",
              sizes=[(s, s) for s in sizes],
              append_images=[render(s) for s in sizes if s != 256])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
