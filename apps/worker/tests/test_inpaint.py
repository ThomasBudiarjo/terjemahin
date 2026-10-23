from pathlib import Path

import cv2
import numpy as np
import pytest

from terjemahin_worker.config import Settings
from terjemahin_worker.inpaint import INPAINTERS
from terjemahin_worker.inpaint import lama as L
from terjemahin_worker.weights import WEIGHTS as SPECS

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "eval" / "fixtures" / "page01.png"
WEIGHTS_DIR = ROOT / "weights"


def _text_page(h: int = 700, w: int = 600) -> tuple[np.ndarray, np.ndarray]:
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(img, "ABC", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)
    mask = cv2.dilate(((img[..., 0] < 128) * 255).astype(np.uint8), np.ones((5, 5), np.uint8))
    return img, mask


def _stripes(h: int, w: int) -> np.ndarray:
    x = np.arange(w)[None, :, None]
    return np.broadcast_to(((x // 6) % 2) * 200 + 30, (h, w, 3)).astype(np.uint8).copy()


def test_registered():
    assert {"lama_manga", "lama", "telea", "ns"} <= set(INPAINTERS.names())


def test_plan_tiles_merges_near_and_splits_far():
    hole = np.zeros((2000, 1500), bool)
    hole[100:140, 100:120] = hole[100:140, 150:170] = True  # 30px apart: one cluster
    hole[1500:1600, 1200:1250] = True
    tiles = L.plan_tiles(hole, size=512, margin=2.0, merge=48)
    assert len(tiles) == 2
    for t in tiles:
        x1, y1, x2, y2 = t.win
        assert x2 - x1 == 512 and y2 - y1 == 512  # small clusters run at native resolution
        assert 0 <= x1 and 0 <= y1 and x2 <= 1500 and y2 <= 2000
        ox1, oy1, ox2, oy2 = t.own
        assert x1 <= ox1 and y1 <= oy1 and ox2 <= x2 and oy2 <= y2


def test_plan_tiles_packs_clusters_that_share_a_window():
    hole = np.zeros((1000, 1000), bool)
    hole[100:120, 100:120] = hole[100:120, 300:320] = True  # separate clusters, 220px union
    assert len(L.plan_tiles(hole, size=512, margin=2.0, merge=48)) == 1
    hole[100:120, 600:620] = True  # 520px union: needs its own window
    assert len(L.plan_tiles(hole, size=512, margin=2.0, merge=48)) == 2


def test_plan_tiles_cells_cover_huge_hole_with_bounded_downscale():
    hole = np.zeros((3000, 2000), bool)
    hole[100:2900:7, 100:1900] = True  # a page full of text lines
    tiles = L.plan_tiles(hole, size=512, margin=2.0, max_down=2.0)
    covered = np.zeros_like(hole)
    for t in tiles:
        x1, y1, x2, y2 = t.win
        assert max(x2 - x1, y2 - y1) <= 1024
        ox1, oy1, ox2, oy2 = t.own
        covered[oy1:oy2, ox1:ox2] = True
    assert covered[hole].all()


def test_window_clamps_to_small_page():
    t = L.plan_tiles(np.pad(np.ones((10, 10), bool), 50), size=512)[0]
    assert t.win == (0, 0, 110, 110)


def test_flat_fill_only_on_flat_background():
    img, mask = _text_page()
    img[:, 300:] = _stripes(img.shape[0], 300)
    cv2.putText(img, "X", (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)
    mask[350:560, 380:520] = 255
    out, rest = L.flat_fill(img, mask > 0, tol=3.0)
    assert (out[:300][mask[:300] > 0] == 255).all()
    assert not rest[:300].any() and rest[350:560, 380:520].all()


class FakeSession:
    """Mimics the graphs: zero the hole, fill it with mid grey; records what it was fed."""

    def __init__(self, scale: float) -> None:
        self.scale, self.calls = scale, []

    def run(self, _, feeds):
        x, m = feeds["image"], feeds["mask"]
        self.calls.append((x.shape, float(x.min()), float(x.max()), set(np.unique(m))))
        return [(x * (1 - m) + 0.5 * m) * self.scale]


@pytest.mark.parametrize("scale", [1.0, 255.0])
def test_lama_pastes_only_holes(monkeypatch, scale):
    fake = FakeSession(scale)
    monkeypatch.setattr(L, "_session", lambda *a: fake)
    img = _stripes(900, 700)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:160, 100:300] = mask[700:760, 400:600] = 255
    inp = L.LamaInpainter(Path("x.onnx"), out_scale=scale, batch=2, feather=0, dilate=0)
    out = inp.inpaint(img, mask)
    assert inp.stats["tiles"] == 2 and len(fake.calls) == 1  # one batch of two tiles
    shape, lo, hi, mvals = fake.calls[0]
    assert shape == (2, 3, 512, 512) and 0 <= lo and hi <= 1 and mvals <= {0.0, 1.0}
    assert (out[mask == 0] == img[mask == 0]).all()
    assert np.abs(out[mask > 0].astype(int) - 128).max() <= 1


def test_lama_downscaled_window_round_trips(monkeypatch):
    fake = FakeSession(1.0)
    monkeypatch.setattr(L, "_session", lambda *a: fake)
    img = _stripes(1400, 1300)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[300:800, 300:700] = 255  # 500px hole -> 1000px window -> 0.512x into the model
    inp = L.LamaInpainter(Path("x.onnx"), out_scale=1.0, feather=0, dilate=0)
    out = inp.inpaint(img, mask)
    assert inp.stats["tiles"] == 1 and fake.calls[0][0] == (1, 3, 512, 512)
    assert (out[mask == 0] == img[mask == 0]).all()
    assert np.abs(out[mask > 0].astype(int) - 128).max() <= 1


def test_lama_feather_stays_near_hole(monkeypatch):
    monkeypatch.setattr(L, "_session", lambda *a: FakeSession(1.0))
    img = _stripes(600, 600)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[200:260, 200:400] = 255
    out = L.LamaInpainter(Path("x.onnx"), out_scale=1.0, feather=2, dilate=0).inpaint(img, mask)
    changed = np.abs(out.astype(int) - img).max(axis=2) > 0
    near = cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0
    assert changed[mask > 0].all() and not changed[~near].any()


def test_lama_skips_model_on_flat_background(monkeypatch):
    def boom(*a):
        raise AssertionError("model should not load")

    monkeypatch.setattr(L, "_session", boom)
    img, mask = _text_page()
    out = L.LamaInpainter(Path("x.onnx"), out_scale=1.0).inpaint(img, mask)
    assert (out == 255).all()


@pytest.mark.parametrize("name", ["telea", "ns"])
def test_opencv_inpainters(name):
    img, mask = _text_page()
    out = INPAINTERS.build(name, Settings()).inpaint(img, mask)
    assert (out[mask == 0] == img[mask == 0]).all()
    assert out[mask > 0].min() > 200


def _art_case() -> tuple[np.ndarray, np.ndarray]:
    """Strokes painted over real artwork (face/hair) and a bubble on page01: known truth."""
    page = cv2.imread(str(PAGE))
    crop = page[440:1240, 0:640].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    for x, y in [(150, 300), (300, 420), (420, 250), (200, 600)]:
        cv2.line(mask, (x, y), (x + 60, y + 40), 255, 7)
    return crop, mask


@pytest.mark.parametrize("name,key", [("lama_manga", "lama_manga"), ("lama", "lama")])
def test_lama_real_weights_reconstruct_art(name, key):
    if not (WEIGHTS_DIR / SPECS[key].filename).exists():
        pytest.skip(f"{key} weights missing")
    if not PAGE.exists():
        pytest.skip("page01 fixture missing")
    img, mask = _art_case()
    inp = INPAINTERS.build(name, Settings(weights_dir=WEIGHTS_DIR), flat_tol=0)
    out = inp.inpaint(img, mask)
    hole = mask > 0
    err = np.abs(out.astype(int) - img)[hole].mean()
    blur = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    assert (out[~hole] == img[~hole]).sum() > 0.97 * (~hole).sum() * 3
    assert err < 12, err
    assert err <= np.abs(blur.astype(int) - img)[hole].mean() + 2
