"""mask/refine.balloon_sweep: leftover ink near the text in flat closed balloons is erased."""

import cv2
import numpy as np

from terjemahin_worker.mask.refine import GlyphMask, balloon_sweep
from terjemahin_worker.models import BBox, TextRegion


def balloon(texture=False):
    img = np.full((300, 300, 3), 140, np.uint8)  # mid-grey art around the balloon
    cv2.ellipse(img, (150, 150), (90, 130), 0, 0, 360, (255, 255, 255), -1)
    if texture:  # screentone inside the balloon
        inside = cv2.ellipse(np.zeros((300, 300), np.uint8), (150, 150), (88, 128), 0, 0, 360, 1, -1)
        dots = np.zeros((300, 300), bool)
        for dy in (0, 1):
            for dx in (0, 1):
                dots[dy::4, dx::4] = True
        img[dots & (inside > 0)] = 90  # 2x2 dots every 4 px: a 25% tone
    cv2.ellipse(img, (150, 150), (90, 130), 0, 0, 360, (0, 0, 0), 3)
    return img


def column(img):
    """A vertical text column (x 140..160) with furigana beside it and a faint JPEG-like halo."""
    for k in range(5):
        y = 70 + k * 24
        cv2.rectangle(img, (142, y), (158, y + 18), (20, 20, 20), 3)
        for x in range(139, 162, 4):  # grey JPEG-ringing specks around the strokes
            img[y - 2, x] = img[y + 20, x] = 200
    for k in range(4):  # furigana: small strokes right of the column
        cv2.line(img, (166, 74 + k * 12), (171, 80 + k * 12), (30, 30, 30), 1)
    glyph_mask = np.zeros(img.shape[:2], np.uint8)
    for k in range(5):
        y = 70 + k * 24
        cv2.rectangle(glyph_mask, (141, y - 1), (159, y + 19), 255, -1)
    region = TextRegion(id=0, bbox=BBox(x1=140, y1=68, x2=161, y2=190), font_size=20,
                        lines=[[(140, 68), (161, 68), (161, 190), (140, 190)]])
    return region, glyph_mask


def test_sweep_takes_furigana_and_halo_but_not_the_outline():
    img = balloon()
    region, glyph_mask = column(img)
    out = balloon_sweep(img, [region], glyph_mask)
    assert out[80, 168] and out[110, 169]  # furigana
    assert out[68, 139] and out[90, 143]  # ringing just outside the glyph mask
    outline = cv2.ellipse(np.zeros((300, 300), np.uint8), (150, 150), (90, 130), 0, 0, 360, 255, 3)
    assert not (out & outline).any()
    assert not out[250:, :].any()  # far from the text: untouched


def test_sweep_skips_textured_balloons_and_text_over_art():
    img = balloon(texture=True)
    region, glyph_mask = column(img)
    assert not balloon_sweep(img, [region], glyph_mask).any()
    art = np.full((300, 300, 3), 140, np.uint8)
    region, glyph_mask = column(art)
    assert not balloon_sweep(art, [region], glyph_mask).any()


def test_default_refiner_sweeps_and_option_turns_it_off():
    img = balloon()
    region, _ = column(img)
    on = GlyphMask().refine(img, [region], None)
    off = GlyphMask(sweep=False).refine(img, [region], None)
    assert on[80, 168] and on.sum() > off.sum()
