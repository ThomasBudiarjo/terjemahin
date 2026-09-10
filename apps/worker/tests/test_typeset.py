from PIL import Image, ImageDraw

from terjemahin_worker.models import BBox
from terjemahin_worker.typeset import MIN_FONT, Typesetter


def _ts() -> Typesetter:
    return Typesetter(font_path=None)


def test_layout_fits_inside_box():
    ts = _ts()
    draw = ImageDraw.Draw(Image.new("RGB", (400, 400)))
    lay = ts.layout(draw, "Aku tidak akan menyerah begitu saja", 120, 80)
    font = ts._font(lay.font_size)
    assert lay.lines
    assert len(lay.lines) * lay.line_height <= 80
    assert all(draw.textlength(line, font=font) <= 120 for line in lay.lines)
    assert lay.font_size > MIN_FONT


def test_single_long_word_is_broken_instead_of_dropped():
    ts = _ts()
    draw = ImageDraw.Draw(Image.new("RGB", (400, 400)))
    word = "Supercalifragilisticexpialidocious"
    lay = ts.layout(draw, word, 30, 200)
    assert "".join(lay.lines).replace(" ", "") == word


def test_render_picks_white_text_on_dark_background():
    ts = _ts()
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    bbox = BBox(x1=10, y1=10, x2=190, y2=90)
    ts.render(img, bbox, "HALO", bg_rgb=(0, 0, 0))
    import numpy as np

    assert (np.asarray(img) == 255).all(axis=2).any()


def test_render_empty_text_is_noop():
    ts = _ts()
    img = Image.new("RGB", (50, 50), (255, 255, 255))
    lay = ts.render(img, BBox(x1=0, y1=0, x2=50, y2=50), "   ")
    assert lay.lines == []
