import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from terjemahin_worker.config import WORKER_ROOT
from terjemahin_worker.detect import DETECTORS, sort_reading_order
from terjemahin_worker.detect.classical import ClassicalDetector
from terjemahin_worker.detect.merge import (
    Line,
    group_to_region,
    merge_lines,
    poly_distance,
    sort_points,
)
from terjemahin_worker.detect.order import find_panels, order_within, sort_regions
from terjemahin_worker.models import BBox, TextRegion

FIXTURES = WORKER_ROOT / "eval" / "fixtures"
PAGE = FIXTURES / "page01.png"
FONT = WORKER_ROOT / "fonts" / "ComicShanns2.ttf"

# Text boxes of the 13 GT bubbles of page01, measured by hand on the page, in GT order.
# Bubble 11 spans two touching balloons ("あっ、あとこないだね、" + "髪型変えたら...").
PAGE01_BOXES = [
    (1314, 14, 1459, 307),
    (1153, 227, 1217, 327),
    (172, 11, 289, 187),
    (34, 191, 98, 377),
    (1339, 476, 1459, 664),
    (632, 941, 669, 1029),
    (22, 1058, 105, 1310),
    (1409, 1438, 1481, 1673),
    (1095, 1476, 1121, 1598),
    (852, 1496, 975, 1817),
    (707, 1594, 741, 1826),
    (209, 1433, 459, 1773),
    (46, 1796, 120, 1948),
]


def region(x1, y1, x2, y2, i=0) -> TextRegion:
    return TextRegion(id=i, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2))


# ---------------------------------------------------------------- merge


def test_sort_points_orders_clockwise_and_detects_direction():
    pts, vertical = sort_points(np.array([[10, 100], [0, 0], [10, 0], [0, 100]]))
    assert vertical
    assert pts.tolist() == [[0, 0], [10, 0], [10, 100], [0, 100]]
    pts, vertical = sort_points(np.array([[100, 10], [0, 0], [0, 10], [100, 0]]))
    assert not vertical
    assert pts.tolist() == [[0, 0], [100, 0], [100, 10], [0, 10]]


def test_poly_distance():
    a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert poly_distance(a, a + [5, 5]) == 0
    assert poly_distance(a, a + [30, 0]) == pytest.approx(20)
    assert poly_distance(a, a + [13, 14]) == pytest.approx(5)


def test_merge_vertical_columns_into_one_block_right_to_left():
    cols = [Line.from_box(x, 100, x + 30, 100 + h) for x, h in ((200, 300), (160, 240), (120, 120))]
    far = Line.from_box(600, 100, 630, 300)
    lines = [cols[2], far, cols[0], cols[1]]
    blocks = merge_lines(lines)
    assert len(blocks) == 2
    big = max(blocks, key=lambda b: len(b[0]))
    assert big[1] == "v"
    assert [lines[i] for i in big[0]] == cols  # rightmost column first
    reg = group_to_region(lines, *big)
    assert (reg.bbox.x1, reg.bbox.y1, reg.bbox.x2, reg.bbox.y2) == (120, 100, 230, 400)
    assert reg.direction == "v" and len(reg.lines) == 3 and reg.font_size == 30 and reg.angle == 0


def test_merge_horizontal_lines_top_to_bottom_and_splits_font_size_outliers():
    rows = [Line.from_box(100, y, 400, y + 24) for y in (100, 130, 160)]
    title = Line.from_box(100, 200, 500, 290)  # 90 px glyphs: different block
    lines = [rows[1], title, rows[2], rows[0]]
    blocks = merge_lines(lines)
    groups = sorted(([lines[i] for i in g], d) for g, d in blocks if len(g) > 1)
    assert len(blocks) == 2
    assert groups[0][1] == "h" and groups[0][0] == rows


# ---------------------------------------------------------------- order


def _texture(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, rng: np.random.Generator) -> None:
    noise = rng.integers(60, 200, size=(y2 - y1, x2 - x1, 1), dtype=np.uint8).repeat(3, axis=2)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(noise, (5, 5), 0)


def synthetic_page() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Three tiers (1 / 2 / 3 panels) with black frames on a light gutter. The top panel has a
    flat white band across its middle with a wavy edge, which must not be cut."""
    rng = np.random.default_rng(0)
    img = np.full((1400, 1000, 3), 235, np.uint8)
    panels = [
        (20, 20, 980, 400),
        (510, 420, 980, 900),
        (20, 420, 490, 900),
        (680, 920, 980, 1380),
        (350, 920, 660, 1380),
        (20, 920, 330, 1380),
    ]
    for x1, y1, x2, y2 in panels:
        _texture(img, x1, y1, x2, y2, rng)
    xs = np.arange(20, 980)
    top = (170 + 25 * np.sin(xs / 37)).astype(int)
    bottom = (260 + 25 * np.cos(xs / 41)).astype(int)
    for x, t, b in zip(xs, top, bottom):
        img[t:b, x] = 250
    for x1, y1, x2, y2 in panels:
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 0), 5)
    return img, panels


def test_find_panels_reads_tiers_then_right_to_left():
    img, panels = synthetic_page()
    found = find_panels(img)
    assert len(found) == len(panels)
    for f, (x1, y1, x2, y2) in zip(found, panels):  # cells contain their panel in order
        assert f.x1 <= x1 + 5 and f.y1 <= y1 + 5 and f.x2 >= x2 - 5 and f.y2 >= y2 - 5
    ltr = find_panels(img, rtl=False)
    assert [(p.x1, p.y1) for p in ltr[1:3]] == [(p.x1, p.y1) for p in reversed(found[1:3])]


def test_sort_regions_follows_panels_not_rows():
    img, _ = synthetic_page()
    # The left middle panel's balloon sits higher than the right one's, so a page-wide row
    # sort would read it first; manga order is right panel first.
    boxes = [
        (820, 60, 900, 300),  # top panel, right
        (100, 80, 180, 260),  # top panel, left
        (800, 640, 880, 860),  # middle right (low)
        (300, 440, 380, 560),  # middle left (high)
        (760, 960, 840, 1200),
        (420, 1100, 500, 1300),
        (60, 940, 140, 1100),
    ]
    shuffled = [region(*b, i) for i, b in enumerate(boxes)]
    random.Random(1).shuffle(shuffled)
    ordered = sort_regions(shuffled, img)
    assert [(r.bbox.x1, r.bbox.y1) for r in ordered] == [b[:2] for b in boxes]
    assert [r.id for r in ordered] == list(range(len(boxes)))
    naive = [(r.bbox.x1, r.bbox.y1) for r in sort_regions(shuffled)]  # no image: rows only
    assert naive.index((300, 440)) < naive.index((800, 640))


def test_order_within_rows_by_vertical_overlap():
    a, b, c = region(500, 0, 560, 300), region(300, 200, 360, 400), region(100, 500, 160, 600)
    assert order_within([c, b, a]) == [a, b, c]
    assert order_within([c, b, a], rtl=False) == [b, a, c]


def test_sort_reading_order_accepts_image_and_keeps_old_call():
    img, _ = synthetic_page()
    regs = [region(300, 440, 380, 560, 0), region(800, 640, 880, 860, 1)]
    assert [r.bbox.x1 for r in sort_reading_order(regs, image=img)] == [800, 300]
    assert [r.bbox.x1 for r in sort_reading_order(regs)] == [300, 800]


# ---------------------------------------------------------------- classical (synthetic)


def balloon_page(texts: list[tuple[str, str, tuple[int, int]]]) -> np.ndarray:
    """Grey textured page with white outlined ellipse balloons holding text set as tightly as
    manga lettering ("|" separates columns / rows)."""
    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur(rng.integers(90, 190, (1200, 900, 3), dtype=np.uint8), (7, 7), 0)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(str(FONT), 30)
    step_col, step_char, step_row = 24, 32, 32
    for text, direction, (cx, cy) in texts:
        cols = text.split("|")
        if direction == "v":
            w, h = step_col * len(cols), step_char * max(len(c) for c in cols)
        else:
            w, h = 17 * max(len(c) for c in cols), step_row * len(cols)
        draw.ellipse(
            (cx - w * 0.8 - 30, cy - h * 0.8 - 30, cx + w * 0.8 + 30, cy + h * 0.8 + 30),
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=5,
        )
        for i, col in enumerate(cols):
            if direction == "v":
                x = cx + w / 2 - step_col * (i + 1)
                for j, ch in enumerate(col):
                    draw.text((x, cy - h / 2 + step_char * j), ch, font=font, fill=(0, 0, 0))
            else:
                draw.text((cx - w / 2, cy - h / 2 + step_row * i), col, font=font, fill=(0, 0, 0))
    return np.asarray(pil).copy()


def test_classical_finds_vertical_and_horizontal_balloons_on_synthetic_page():
    img = balloon_page(
        [("ABCDE|FGH|IJKLM", "v", (650, 300)), ("HELLO THERE|HOW ARE YOU", "h", (300, 800))]
    )
    # a white blob with one big stroke is not text
    cv2.circle(img, (200, 250), 110, (255, 255, 255), -1)
    cv2.line(img, (130, 180), (270, 330), (0, 0, 0), 14)
    det = DETECTORS.build("classical", None)
    out = det.detect(img)
    regs = sort_reading_order(out.regions, image=img)
    assert len(regs) == 2
    v, h = regs
    assert v.direction == "v" and len(v.lines) == 3
    assert [p[0][0] for p in v.lines] == sorted((p[0][0] for p in v.lines), reverse=True)
    assert 560 < v.bbox.cx < 740 and 200 < v.bbox.cy < 400
    assert h.direction == "h" and len(h.lines) == 2 and h.bbox.cy > 700
    assert v.fg_color is not None and max(v.fg_color) < 90 and min(v.bg_color) > 240
    assert out.mask is not None and out.mask.shape == img.shape[:2]
    assert out.mask[v.bbox.y1 : v.bbox.y2, v.bbox.x1 : v.bbox.x2].mean() > 10
    assert out.mask[:, :400][:600].max() == 0  # nothing from the stroke blob


def test_classical_empty_page():
    out = ClassicalDetector().detect(np.full((800, 600, 3), 255, np.uint8))
    assert out.regions == [] and out.mask is not None and out.mask.max() == 0


# ---------------------------------------------------------------- page01 (real fixture)

needs_page = pytest.mark.skipif(not PAGE.exists(), reason="page01 fixture missing")


def _gt_index(r: TextRegion) -> int:
    def inter(b):
        x1, y1, x2, y2 = b
        return max(0, min(x2, r.bbox.x2) - max(x1, r.bbox.x1)) * max(
            0, min(y2, r.bbox.y2) - max(y1, r.bbox.y1)
        )

    best = max(range(len(PAGE01_BOXES)), key=lambda i: inter(PAGE01_BOXES[i]))
    return best if inter(PAGE01_BOXES[best]) > 0.5 * r.bbox.w * r.bbox.h else -1


@needs_page
def test_page01_hand_boxes_sort_into_gt_order():
    img = cv2.imread(str(PAGE))
    gt = json.loads(Path(FIXTURES / "page01.gt.json").read_text())["bubbles"]
    assert len(gt) == len(PAGE01_BOXES)
    regs = [region(*b, i) for i, b in enumerate(PAGE01_BOXES)]
    random.Random(7).shuffle(regs)
    ordered = sort_reading_order(regs, image=img)
    assert [
        PAGE01_BOXES.index((r.bbox.x1, r.bbox.y1, r.bbox.x2, r.bbox.y2)) for r in ordered
    ] == list(range(13))


@needs_page
def test_page01_panels():
    img = cv2.imread(str(PAGE))
    panels = find_panels(img)
    # top strip, middle right, middle left, bottom right, bottom left (borderless)
    centres = [(p.cx, p.cy) for p in panels]
    assert len(panels) == 5
    expected = [(740, 200), (1080, 890), (330, 890), (1140, 1710), (390, 1710)]
    for (cx, cy), (ex, ey) in zip(centres, expected):
        assert abs(cx - ex) < 80 and abs(cy - ey) < 80


@needs_page
def test_page01_classical_detects_every_bubble_in_reading_order():
    img = cv2.imread(str(PAGE))
    out = ClassicalDetector().detect(img)
    regs = sort_reading_order(out.regions, image=img)
    idx = [_gt_index(r) for r in regs]
    assert -1 not in idx  # precision: every detection is a GT bubble
    assert sorted(set(idx)) == list(range(13))  # recall: every GT bubble found
    assert idx == sorted(idx)  # reading order matches GT
    assert len(regs) == 14  # bubble 11 is two balloons -> two blocks
    assert all(r.direction == "v" for r in regs)
    assert out.mask is not None and 0.005 < (out.mask > 0).mean() < 0.05
