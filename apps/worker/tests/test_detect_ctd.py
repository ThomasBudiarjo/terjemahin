import itertools
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import latency_budget

from terjemahin_worker.config import Settings
from terjemahin_worker.detect.ctd_full import (
    CtdFullDetector,
    merge_tile_blocks,
    plan_tiles,
    scope_mask,
)
from terjemahin_worker.detect.ctd_utils import (
    db_lines,
    decode_blocks,
    group_lines,
    line_geometry,
    mask_lines,
)
from terjemahin_worker.models import BBox, TextRegion

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = Settings().weights_dir / "koharu-comictextdetector.onnx"
PAGE = ROOT / "eval" / "fixtures" / "page01.png"
needs_model = pytest.mark.skipif(
    not (WEIGHTS.exists() and PAGE.exists()), reason="ctd weights or sample page missing"
)

# centres of the 13 dialogue bubbles on page01 (original pixels), checked by eye
BUBBLES = [
    (1387, 160), (232, 98), (1184, 276), (68, 281), (1400, 569), (649, 985), (63, 1186),
    (1444, 1554), (1106, 1536), (335, 1604), (908, 1652), (724, 1710), (88, 1870),
]


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def test_decode_blocks_filters_and_nms():
    blk = np.zeros((1, 4, 7), np.float32)
    blk[0, 0] = [100, 100, 40, 80, 0.9, 0.0, 1.0]
    blk[0, 1] = [102, 101, 40, 80, 0.8, 0.0, 1.0]  # duplicate
    blk[0, 2] = [300, 300, 20, 20, 0.9, 1.0, 0.0]
    blk[0, 3] = [500, 500, 20, 20, 0.2, 0.0, 1.0]  # low confidence
    xyxy, scores, cls = decode_blocks(blk, 0.4, 0.35)
    assert len(xyxy) == 2
    i = int(np.argmin(xyxy[:, 0]))
    np.testing.assert_allclose(xyxy[i], [80, 60, 120, 140])
    assert cls[i] == 1 and abs(scores[i] - 0.9) < 1e-6


def test_db_lines_unclips_shrunk_stripe():
    prob = np.zeros((200, 200), np.float32)
    prob[40:160, 95:105] = 0.9  # 10 x 120 shrink core of a vertical line
    quads, scores = db_lines(prob, unclip=1.5)
    assert len(quads) == 1 and scores[0] > 0.8
    vertical, angle, length, thick = line_geometry(quads[0])
    assert vertical and angle == pytest.approx(0, abs=1)
    w, h = 9, 119  # contour through pixel centres, as in the reference
    d = w * h * 1.5 / (2 * (w + h))  # area * ratio / perimeter
    assert thick == pytest.approx(w + 2 * d, abs=0.5)
    assert length == pytest.approx(h + 2 * d, abs=0.5)


def test_line_geometry_angle_sign():
    t = np.radians(10)
    v = np.array([np.cos(t), -np.sin(t)])  # horizontal line, right end raised
    n = np.array([np.sin(t), np.cos(t)])
    q = np.array([[0, 0], 100 * v, 100 * v + 20 * n, 20 * n], np.float32) + 50
    vertical, angle, length, thick = line_geometry(q)
    assert not vertical and angle == pytest.approx(10, abs=0.1)
    assert length == pytest.approx(100, abs=0.1) and thick == pytest.approx(20, abs=0.1)


def test_group_lines_assigns_orders_and_filters():
    mask = np.zeros((400, 400), np.float32)
    mask[20:180, 20:180] = 1.0
    mask[300:330, 200:380] = 1.0
    blocks_xyxy = np.array([[20, 20, 180, 180], [250, 20, 300, 60]], np.float32)
    lines = [rect(30, 20, 60, 170), rect(75, 20, 105, 175), rect(200, 300, 380, 330)]
    out = group_lines(blocks_xyxy, np.array([0.9, 0.8]), lines, [0.9, 0.9, 0.9], mask)
    # second YOLO block has no line and no mask support -> dropped; the horizontal line
    # outside any block becomes its own block because the mask confirms text there
    assert len(out) == 2
    v = next(b for b in out if b.vertical)
    h = next(b for b in out if not b.vertical)
    assert v.lines[0][:, 0].mean() > v.lines[1][:, 0].mean()  # right-to-left
    assert v.font_size == pytest.approx(30, abs=0.5)
    np.testing.assert_allclose(h.xyxy, [200, 300, 380, 330])


def test_group_lines_splits_far_columns():
    blocks_xyxy = np.array([[0, 0, 400, 200]], np.float32)
    lines = [rect(360, 10, 390, 190), rect(320, 10, 350, 190), rect(10, 10, 40, 190)]
    out = group_lines(blocks_xyxy, np.array([0.9]), lines, [0.9] * 3, None)
    assert sorted(len(b.lines) for b in out) == [1, 2]


def test_line_inside_block_or_block_inside_line_is_assigned():
    # DB line fused two columns and is bigger than the tight YOLO box
    out = group_lines(
        np.array([[50, 50, 116, 153]], np.float32), np.array([0.9]),
        [rect(20, 27, 140, 177)], [0.8], None,
    )
    assert len(out) == 1


def test_mask_lines_columns():
    mask = np.zeros((200, 200), np.float32)
    mask[20:180, 140:170] = 1
    mask[20:120, 90:120] = 1
    quads = mask_lines(mask, np.array([80, 10, 180, 190], np.float32))
    assert len(quads) == 2 and all(line_geometry(q)[0] for q in quads)


def test_plan_tiles():
    ratio, tiles = plan_tiles(1487, 2048, 1024, 2.5)
    assert tiles == [(0, 0, 1487, 2048)] and ratio == pytest.approx(0.5)
    ratio, tiles = plan_tiles(800, 9000, 1024, 2.5)
    assert len(tiles) > 1 and tiles[0][1] == 0 and tiles[-1][3] == 9000
    assert all(t[2] - t[0] == 800 and t[3] - t[1] == 1200 for t in tiles)
    assert all(b[1] < a[3] for a, b in itertools.pairwise(tiles))  # overlapping
    assert max(t[3] - t[1] for t in tiles) * ratio <= 1024
    _, wide = plan_tiles(9000, 800, 1024, 2.5)
    assert wide[0] == (0, 0, 1200, 800)


def test_merge_tile_blocks_drops_clipped_duplicates():
    xyxy = np.array([[0, 90, 50, 100], [0, 60, 50, 140], [200, 0, 250, 50]], np.float32)
    at_cut = np.array([True, False, True])
    keep = merge_tile_blocks(xyxy, np.array([0.9, 0.8, 0.7]), at_cut, 0.35)
    assert sorted(keep.tolist()) == [1, 2]


def test_scope_mask_keeps_text_near_regions():
    mask = np.full((100, 100), 200, np.uint8)
    r = TextRegion(id=0, bbox=BBox(x1=40, y1=40, x2=60, y2=60), font_size=8)
    out = scope_mask(mask, [r])
    assert out[50, 50] == 200 and out[37, 37] == 200 and out[5, 5] == 0


class FakeSession:
    """Emits one block, one DB line and one seg blob at known letterbox coords."""

    def run(self, names, feeds):
        x = feeds["images"]
        assert x.shape == (1, 3, 1024, 1024) and x.dtype == np.float32
        assert x[0, :, :, 600:].max() == 0  # bottom/right padding
        blk = np.zeros((1, 3, 7), np.float32)
        blk[0, 0] = [200, 300, 40, 160, 0.95, 0.0, 1.0]
        seg = np.zeros((1, 1, 1024, 1024), np.float32)
        seg[0, 0, 230:370, 185:215] = 1.0
        det = np.zeros((1, 2, 1024, 1024), np.float32)
        det[0, 0, 235:365, 195:205] = 0.9
        out = {"blk": blk, "seg": seg, "det": det}
        return [out[n] for n in names]


def test_detector_maps_letterbox_back_to_page():
    det = CtdFullDetector(Path("unused"), session=FakeSession())
    page = np.full((2048, 1200, 3), 255, np.uint8)  # ratio 0.5, resized to 600x1024
    res = det.detect(page)
    assert len(res.regions) == 1
    r = res.regions[0]
    assert r.direction == "v" and len(r.lines) == 1
    assert 360 <= r.bbox.x1 <= 390 and 410 <= r.bbox.x2 <= 440
    assert 440 <= r.bbox.y1 <= 470 and 730 <= r.bbox.y2 <= 760
    assert res.mask.shape == (2048, 1200) and res.mask.dtype == np.uint8
    assert res.mask[600, 400] == 255 and res.mask[100, 100] == 0


@pytest.fixture(scope="module")
def ctd():
    return CtdFullDetector(WEIGHTS)


@needs_model
def test_page01_all_bubbles_separate_lines_and_fast(ctd):
    page = cv2.imread(str(PAGE))
    ctd.detect(page)  # warm-up
    t = time.perf_counter()
    res = ctd.detect(page)
    elapsed = time.perf_counter() - t
    assert elapsed < latency_budget(5.0), f"ctd took {elapsed:.2f}s"  # ~0.6s on 4 idle cores
    hits = []
    for cx, cy in BUBBLES:
        inside = [
            i for i, r in enumerate(res.regions)
            if r.bbox.x1 <= cx <= r.bbox.x2 and r.bbox.y1 <= cy <= r.bbox.y2
        ]
        assert len(inside) == 1, f"bubble at {(cx, cy)} -> regions {inside}"
        hits.append(inside[0])
    assert len(set(hits)) == 13
    for i in hits:
        r = res.regions[i]
        assert r.direction == "v" and r.lines and r.angle == 0
        assert 25 <= r.font_size <= 60
        assert r.fg_color is not None and max(r.fg_color) < 100
        assert r.bg_color is not None and min(r.bg_color) > 200
        for q in r.lines:
            xs, ys = [p[0] for p in q], [p[1] for p in q]
            assert r.bbox.x1 <= min(xs) and max(xs) <= r.bbox.x2
            assert r.bbox.y1 <= min(ys) and max(ys) <= r.bbox.y2


@needs_model
def test_page01_mask_covers_glyphs_not_outlines():
    page = cv2.imread(str(PAGE))
    res = CtdFullDetector(WEIGHTS, mask_scope="page").detect(page)
    mask = res.mask
    assert mask.shape == page.shape[:2] and mask.dtype == np.uint8
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    in_text = np.zeros_like(mask, bool)
    for r in res.regions:
        for q in r.lines:
            cv2.fillPoly(in_text.view(np.uint8), [np.array(q, np.int32)], 1)
    ink = (gray < 90) & in_text
    assert (mask[ink] > 127).mean() > 0.9  # glyph strokes are covered
    # dark pixels well away from any text block (balloon outlines, line art, hair)
    near = np.zeros_like(mask)
    for r in res.regions:
        b = r.bbox.pad(15).clamp(*page.shape[1::-1])
        near[b.y1 : b.y2, b.x1 : b.x2] = 1
    art = (gray < 90) & (near == 0)
    assert (mask[art] > 127).mean() < 0.01


@needs_model
def test_tall_page_is_tiled(ctd):
    page = cv2.imread(str(PAGE))
    tall = np.vstack([page, page])  # 1487 x 4096, aspect 2.75 -> tiles
    res = ctd.detect(tall)
    for dy in (0, page.shape[0]):
        for cx, cy in BUBBLES:
            assert any(
                r.bbox.x1 <= cx <= r.bbox.x2 and r.bbox.y1 <= cy + dy <= r.bbox.y2
                for r in res.regions
            ), (cx, cy + dy)
    assert len(res.regions) <= 2 * 15  # no duplicates from overlapping tiles
    assert res.mask.shape == tall.shape[:2]


@needs_model
def test_blk_only_head_runs_without_mask():
    page = cv2.imread(str(PAGE))
    res = CtdFullDetector(WEIGHTS, heads=("blk",)).detect(page)
    assert res.mask is None and len(res.regions) >= 13
