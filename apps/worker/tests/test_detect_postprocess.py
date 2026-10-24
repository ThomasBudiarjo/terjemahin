"""detect/postprocess.py: boxes split over two balloons, grown over glyphs they cut off;
ctd's two-page spread tiling; reading order on spreads and 4-koma strips."""

import cv2
import numpy as np

from terjemahin_worker.detect import order as O
from terjemahin_worker.detect import postprocess as P
from terjemahin_worker.detect.base import Detection
from terjemahin_worker.detect.ctd_full import plan_tiles, union_detections
from terjemahin_worker.models import BBox, TextRegion

F = 20  # glyph size


def page(h=400, w=400, bg=255):
    return np.full((h, w), bg, np.uint8)


def glyph(img, x, y, s=F, v=0):
    """A glyph-like blob: a square ring with a bar (one connected dark component)."""
    cv2.rectangle(img, (x + 2, y + 2), (x + s - 3, y + s - 3), v, 2)
    cv2.line(img, (x + 2, y + s // 2), (x + s - 3, y + s // 2), v, 2)


def column(img, x, y, n, s=F, gap=4):
    for k in range(n):
        glyph(img, x, y + k * (s + gap), s)
    return y + n * (s + gap) - gap


def region(x1, y1, x2, y2, lines=(), font=F, direction="v"):
    return TextRegion(
        id=0, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), font_size=font, direction=direction,
        lines=[P._quad(*q) for q in lines],
    )


def test_box_grows_over_the_glyph_it_cut_off_and_not_over_furigana():
    img = page()
    cv2.ellipse(img, (200, 200), (70, 140), 0, 0, 360, 0, 2)  # balloon outline
    bottom = column(img, 190, 110, 4)  # 4 glyphs, y 110..200
    cv2.circle(img, (200, bottom + 10), 3, 0, -1)  # "…" dot below, in line
    for k in range(3):  # furigana beside the column: half-size glyphs
        glyph(img, 214, 115 + k * 13, 10)
    r = region(190, 134, 210, bottom, lines=[(190, 134, 210, bottom)])  # misses the first glyph
    out = P.complete_region(r, img)
    assert out.bbox.y1 <= 111 and out.bbox.y2 >= bottom + 12
    assert out.bbox.x2 <= 212  # the furigana stays out of the box
    (q,) = out.lines
    assert P._qbox(q)[1] <= 111  # the line quad follows the box


def test_box_grows_over_a_full_size_column_and_an_ellipsis_column_beside_it():
    img = page()
    cv2.ellipse(img, (200, 200), (90, 150), 0, 0, 360, 0, 2)
    column(img, 200, 110, 4)
    column(img, 176, 110, 3)  # a second column the detector missed
    for k in range(3):  # "……" set as a column of dots
        cv2.circle(img, (162, 116 + k * 8), 2, 0, -1)
    r = region(200, 110, 220, 202, lines=[(200, 110, 220, 202)])
    out = P.complete_region(r, img)
    assert out.bbox.x1 <= 161 and out.bbox.x2 >= 219
    assert len(out.lines) == 3  # one new quad per new column


def test_box_is_left_alone_on_textured_art():
    rng = np.random.default_rng(0)
    img = (200 + rng.integers(0, 55, (400, 400))).astype(np.uint8)  # light, noisy art
    column(img, 190, 110, 4)
    r = region(190, 134, 210, 202, lines=[(190, 134, 210, 202)])
    assert P.complete_region(r, img) == r


def test_lines_that_do_not_run_side_by_side_are_two_blocks():
    # ARMS_015: "ドクン" in one balloon, "ドクン" below-left in another, boxed as one block
    r = region(100, 100, 150, 200, lines=[(125, 100, 148, 150), (100, 148, 122, 200)])
    parts = P.split_unaligned(r)
    assert len(parts) == 2
    assert {(p.bbox.x1, p.bbox.y1) for p in parts} == {(125, 100), (100, 148)}
    # columns of one block share their height (the shorter last column too): kept whole
    one = region(100, 100, 150, 200, lines=[(125, 100, 148, 200), (100, 100, 122, 160)])
    assert P.split_unaligned(one) == [one]


def test_two_halves_of_one_block_in_one_balloon_are_merged():
    img = page()
    cv2.ellipse(img, (200, 200), (90, 150), 0, 0, 360, 0, 2)
    column(img, 200, 110, 4)
    column(img, 176, 110, 4)
    a = region(200, 110, 220, 202, lines=[(200, 110, 220, 202)])
    b = region(176, 110, 198, 202, lines=[(176, 110, 198, 202)])
    (m,) = P.merge_adjacent([a, b], img)
    assert (m.bbox.x1, m.bbox.x2) == (176, 220) and len(m.lines) == 2
    # the same two columns in two balloons, separated by their outlines, stay apart
    img2 = page()
    cv2.ellipse(img2, (212, 156), (14, 70), 0, 0, 360, 0, 2)
    cv2.ellipse(img2, (186, 156), (14, 70), 0, 0, 360, 0, 2)
    column(img2, 202, 110, 4, s=16)
    column(img2, 178, 110, 4, s=16)
    a2 = region(202, 110, 218, 198, lines=[(202, 110, 218, 198)], font=16)
    b2 = region(178, 110, 194, 198, lines=[(178, 110, 194, 198)], font=16)
    assert len(P.merge_adjacent([a2, b2], img2)) == 2


def test_refine_regions_keeps_clean_boxes_unchanged():
    img = page()
    cv2.ellipse(img, (200, 200), (70, 140), 0, 0, 360, 0, 2)
    bottom = column(img, 190, 110, 4)
    r = region(188, 108, 212, bottom + 2, lines=[(188, 108, 212, bottom + 2)])
    (out,) = P.refine_regions([r], cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    assert out.bbox == r.bbox


def test_spread_runs_as_two_halves():
    ratio, tiles = plan_tiles(1654, 1170, 1024, 2.5)
    assert len(tiles) == 2 and tiles[0][0] == 0 and tiles[1][2] == 1654
    assert tiles[0][2] > 827 > tiles[1][0]  # they overlap across the middle
    assert ratio > 1024 / 1654 + 0.2  # each half much closer to single-page scale
    assert len(plan_tiles(1654, 1170, 1024, 2.5, spread=0)[1]) == 1
    assert len(plan_tiles(900, 700, 1024, 2.5)[1]) == 1  # small enough for one pass


def test_union_detections_adds_only_lines_nothing_covers():
    main = Detection([region(100, 100, 120, 150, lines=[(100, 100, 120, 150)])])
    extra = Detection([
        region(95, 100, 125, 200, lines=[(100, 100, 120, 150), (96, 152, 118, 200)]),
        region(300, 300, 320, 350),
    ])
    out = union_detections(main, extra)
    boxes = [(r.bbox.x1, r.bbox.y1, r.bbox.x2, r.bbox.y2) for r in out.regions]
    assert boxes == [(100, 100, 120, 150), (96, 152, 119, 201), (300, 300, 320, 350)]


def _frames(img, cells):
    rng = np.random.default_rng(1)
    for x1, y1, x2, y2 in cells:  # textured art inside, so only the gutters are flat
        img[y1:y2, x1:x2] = rng.integers(60, 200, (y2 - y1, x2 - x1, 1), dtype=np.uint8)
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 0), 5)


def test_spread_is_read_right_page_first():
    img = np.full((600, 1000, 3), 255, np.uint8)
    cells = []
    for x0 in (20, 520):  # two pages, each two tiers of two panels
        for y0 in (20, 310):
            cells += [(x0, y0, x0 + 220, y0 + 270), (x0 + 240, y0, x0 + 460, y0 + 270)]
    _frames(img, cells)
    panels = O.find_panels(img)
    assert len(panels) == 8
    # right page (x >= 500) first: its four panels before any of the left page's
    assert all(p.x1 >= 500 for p in panels[:4]) and all(p.x2 <= 500 for p in panels[4:])


def test_4koma_strips_are_read_one_strip_at_a_time():
    img = np.full((1000, 520, 3), 255, np.uint8)
    cells = [(x0, 20 + k * 240, x0 + 220, 240 + k * 240) for x0 in (20, 280) for k in range(4)]
    _frames(img, cells)
    panels = O.find_panels(img)
    assert len(panels) == 8
    assert all(p.x1 >= 260 for p in panels[:4])  # the right strip, top to bottom
    assert [p.y1 for p in panels[:4]] == sorted(p.y1 for p in panels[:4])
