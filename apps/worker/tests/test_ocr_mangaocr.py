import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from terjemahin_worker.config import Settings
from terjemahin_worker.models import BBox, TextRegion
from terjemahin_worker.ocr.common import (
    crop_region,
    estimate_colors,
    normalize_ja,
    split_long,
    text_pieces,
    to_fullwidth,
)
from terjemahin_worker.ocr.manga_ocr_onnx import (
    EOS,
    MangaOcrBase,
    banned_ngram_tokens,
    decode_tokens,
    preprocess,
)
from terjemahin_worker.registry import OCRS, load_all

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "eval" / "fixtures" / "page01.png"
GT = ROOT / "eval" / "fixtures" / "page01.gt.json"
WEIGHTS = Settings().weights_dir
ONNX = WEIGHTS / "koharu-manga-ocr.onnx"
VOCAB = WEIGHTS / "koharu-vocab.txt"
needs_onnx = pytest.mark.skipif(
    not (ONNX.exists() and VOCAB.exists() and PAGE.exists()),
    reason="manga-ocr onnx or page missing",
)

# page01 dialogue blocks in gt.json order: (x1, y1, x2, y2, glyph size), from ctd, checked by eye
BOXES = [
    (1313, 10, 1462, 311, 43),
    (1144, 218, 1225, 334, 45),
    (170, 4, 294, 192, 33),
    (31, 186, 106, 377, 36),
    (1334, 471, 1467, 667, 37),
    (621, 931, 677, 1039, 55),
    (18, 1056, 108, 1316, 41),
    (1406, 1434, 1482, 1675, 35),
    (1091, 1471, 1121, 1601, 28),
    (841, 1485, 976, 1820, 39),
    (704, 1590, 744, 1830, 38),
    (210, 1430, 461, 1779, 37),
    (43, 1788, 133, 1951, 43),
]


def regions() -> list[TextRegion]:
    return [
        TextRegion(id=i, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), font_size=fs)
        for i, (x1, y1, x2, y2, fs) in enumerate(BOXES)
    ]


def lev(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def cer(pred: str, ref: str) -> float:
    ref = normalize_ja(ref)
    return lev(normalize_ja(pred), ref) / max(1, len(ref))


# --- fast, offline ---------------------------------------------------------------------------


def test_normalize_matches_manga_ocr_postprocess():
    assert to_fullwidth("ｶﾞｷ ab1?｡") == "ガキ　ａｂ１？。"
    assert normalize_ja("え …? そう・・ ね..") == "え．．．？そう．．ね．．"
    assert normalize_ja("中・黒") == "中・黒"  # a single middle dot is kept
    assert normalize_ja(" \n") == ""


def test_banned_ngrams_like_hf():
    assert banned_ngram_tokens([5, 6, 7, 5, 6], 3) == {7}
    assert banned_ngram_tokens([5, 6], 3) == set()
    assert banned_ngram_tokens([5, 6, 7, 5, 6], 0) == set()
    assert banned_ngram_tokens([9, 9, 9], 2) == {9}


def test_decode_tokens_drops_specials_and_widens():
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "え", "?", ".", "##ね"]
    assert decode_tokens([5, 7, 7, 6, 8, EOS, 0], vocab) == "え．．？ね"


def test_preprocess_is_grey_normalised():
    img = np.zeros((40, 20, 3), np.uint8)
    img[:, :, 2] = 255  # pure red in BGR
    x = preprocess([img])
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert np.allclose(x[0, 0], x[0, 1]) and np.allclose(x[0, 1], x[0, 2])
    assert abs(float(x.mean()) - (76 / 127.5 - 1)) < 0.01  # PIL L of red = 76


def test_estimate_colors_dark_and_light_text():
    paper = np.full((60, 60, 3), (240, 250, 255), np.uint8)  # BGR
    paper[20:40, 10:50] = (30, 20, 10)
    fg, bg = estimate_colors(paper)
    assert fg == (10, 20, 30) and bg == (255, 250, 240)
    inv = np.full((60, 60, 3), 20, np.uint8)
    inv[25:35, 5:55] = 250
    fg, bg = estimate_colors(inv)
    assert fg == (250, 250, 250) and bg == (20, 20, 20)
    assert estimate_colors(np.zeros((0, 0, 3), np.uint8)) == ((0, 0, 0), (255, 255, 255))


def test_crop_pads_and_clamps():
    img = np.zeros((100, 100, 3), np.uint8)
    r = TextRegion(id=0, bbox=BBox(x1=10, y1=0, x2=40, y2=90), font_size=20)
    c = crop_region(img, r, pad=0.25, min_pad=2)
    assert c.shape[:2] == (95, 40)  # 5 px each side, clamped at the top


def test_crop_deskews_tilted_block():
    img = np.full((300, 300, 3), 255, np.uint8)
    quad = cv2.boxPoints(((150, 150), (30, 160), -15)).astype(np.int32)  # leaning column
    cv2.fillPoly(img, [quad], (0, 0, 0))
    x1, y1 = quad.min(0)
    x2, y2 = quad.max(0)
    r = TextRegion(
        id=0,
        bbox=BBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)),
        lines=[[tuple(map(int, p)) for p in quad]],
        font_size=30,
    )
    for angle in (15, -15):
        c = crop_region(img, r.model_copy(update={"angle": angle}), pad=0.0, min_pad=0)
        ink = (c[..., 0] < 128).mean()
        if ink > 0.8:  # the right sign makes the column upright, filling its tight crop
            assert c.shape[0] > 4 * c.shape[1]
            break
    else:
        raise AssertionError("no rotation sign straightened the column")
    upright = crop_region(img, r.model_copy(update={"angle": 15.0}), pad=0.0, min_pad=0)
    assert (upright[..., 0] < 128).mean() > 0.8


def _column(n: int, glyph: int = 30, gap: int = 8) -> np.ndarray:
    img = np.full((n * (glyph + gap) + gap, glyph + 10, 3), 255, np.uint8)
    for k in range(n):
        y = gap + k * (glyph + gap)
        cv2.rectangle(img, (5, y), (5 + glyph, y + glyph), (0, 0, 0), 3)
    return img


def test_split_long_cuts_between_glyphs():
    col = _column(16)
    pieces = split_long(col, vertical=True, chunk_aspect=4.0)
    assert len(pieces) == int(np.ceil(col.shape[0] / (4 * col.shape[1])))
    assert sum(p.shape[0] for p in pieces) == col.shape[0]
    for p in pieces[1:]:
        assert (p[0, :, 0] < 128).sum() == 0  # every cut lands on blank paper
    rows = split_long(
        np.ascontiguousarray(col.transpose(1, 0, 2)), vertical=False, chunk_aspect=4.0
    )
    assert len(rows) == len(pieces) and all(r.shape[0] == col.shape[1] for r in rows)
    assert len(split_long(_column(3), True)) == 1


def test_text_pieces_only_splits_extreme_blocks():
    page = np.full((1000, 400, 3), 255, np.uint8)
    col = _column(20)
    page[50 : 50 + col.shape[0], 100 : 100 + col.shape[1]] = col
    tall = TextRegion(id=0, bbox=BBox(x1=100, y1=50, x2=140, y2=50 + col.shape[0]), font_size=30)
    assert len(text_pieces(page, tall, split_aspect=10)) > 1
    assert len(text_pieces(page, tall, split_aspect=0)) == 1
    short = tall.model_copy(update={"bbox": BBox(x1=100, y1=50, x2=140, y2=250)})
    assert len(text_pieces(page, short)) == 1


class FakeOcr(MangaOcrBase):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "あ"]
        self.batches: list[int] = []

    def generate(self, pixels):
        self.batches.append(len(pixels))
        return [([5, EOS], 0.5) for _ in pixels]


def test_recognize_groups_pieces_and_batches():
    page = np.full((1000, 400, 3), 255, np.uint8)
    col = _column(20)
    page[50 : 50 + col.shape[0], 100 : 100 + col.shape[1]] = col
    tall = TextRegion(id=0, bbox=BBox(x1=100, y1=50, x2=140, y2=50 + col.shape[0]), font_size=30)
    small = TextRegion(id=1, bbox=BBox(x1=200, y1=50, x2=260, y2=150))
    ocr = FakeOcr(batch_size=2)
    out = ocr.recognize(page, [small, tall, small])
    n = len(text_pieces(page, tall))
    assert [o.text for o in out] == ["あ", "あ" * n, "あ"]
    assert sum(ocr.batches) == n + 2 and max(ocr.batches) <= 2
    assert out[1].fg_color == (0, 0, 0) and out[1].bg_color == (255, 255, 255)
    assert FakeOcr().recognize(page, []) == []


def test_methods_registered():
    load_all()
    assert "manga_ocr" in OCRS.methods and "manga_ocr_hf" in OCRS.methods
    assert OCRS.get("manga_ocr").weights == ("manga_ocr_onnx", "manga_ocr_vocab")
    assert Settings().ocr == "manga_ocr"


# --- real weights ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def page():
    return cv2.imread(str(PAGE))


@pytest.fixture(scope="module")
def gt():
    return [b["ja"] for b in json.loads(GT.read_text(encoding="utf-8"))["bubbles"]]


@pytest.fixture(scope="module")
def onnx_ocr():
    load_all()
    return OCRS.build("manga_ocr", Settings())


@pytest.fixture(scope="module")
def onnx_results(onnx_ocr, page):
    t = time.perf_counter()
    res = onnx_ocr.recognize(page, regions())
    print(f"manga_ocr page01: {time.perf_counter() - t:.2f}s for {len(res)} regions")
    return res


@needs_onnx
def test_onnx_cer_on_page01(onnx_results, gt):
    cers = [cer(r.text, g) for r, g in zip(onnx_results, gt)]
    print("manga_ocr CER per bubble", [round(c, 3) for c in cers], "mean", np.mean(cers))
    assert len(onnx_results) == len(gt)
    assert np.mean(cers) < 0.1
    assert max(cers) < 0.2
    for r in onnx_results:
        assert 0.5 < r.confidence <= 1.0
        assert max(r.fg_color) < 90 and min(r.bg_color) > 230  # black ink on white balloons


@needs_onnx
def test_onnx_split_graph_matches_full_graph(onnx_ocr, page):
    from terjemahin_worker.ocr.manga_ocr_onnx import MangaOcrOnnx

    assert onnx_ocr.decoder is not None, "encoder/decoder split failed"
    full = MangaOcrOnnx(ONNX, VOCAB, split=False, max_len=24)
    pixels = preprocess([crop_region(page, r) for r in regions()[1:3]])
    split = MangaOcrOnnx.generate(onnx_ocr, pixels)
    ref = full.generate(pixels)
    assert [s[0][:24] for s in split] == [r[0] for r in ref]


@needs_onnx
def test_onnx_reads_extremely_tall_column(onnx_ocr, page):
    """Four bubbles stacked into one 1:15 column: read whole it is garbage, in pieces it is not."""
    rs = regions()
    crops = [crop_region(page, rs[i]) for i in (8, 10, 8, 10)]
    w = max(c.shape[1] for c in crops)
    col = np.vstack(
        [cv2.copyMakeBorder(c, 0, 0, 0, w - c.shape[1], cv2.BORDER_REPLICATE) for c in crops]
    )
    canvas = np.full((col.shape[0] + 40, w + 40, 3), 255, np.uint8)
    canvas[20 : 20 + col.shape[0], 20 : 20 + w] = col
    region = TextRegion(
        id=0, bbox=BBox(x1=20, y1=20, x2=20 + w, y2=20 + col.shape[0]), font_size=30
    )
    ref = "うんうんわあ、いいね！" * 2
    text = onnx_ocr.recognize(canvas, [region])[0].text
    assert cer(text, ref) < 0.15, text


def _hf_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache("kha-white/manga-ocr-base", "pytorch_model.bin"), str)


@needs_onnx
@pytest.mark.skipif(not _hf_cached(), reason="kha-white/manga-ocr-base not in the HF cache")
def test_hf_backend_agrees_with_onnx(onnx_results, page, gt):
    load_all()
    hf = OCRS.build("manga_ocr_hf", Settings())
    t = time.perf_counter()
    res = hf.recognize(page, regions())
    print(f"manga_ocr_hf page01: {time.perf_counter() - t:.2f}s")
    assert [r.text for r in res] == [r.text for r in onnx_results]
    for a, b in zip(res, onnx_results):
        assert abs(a.confidence - b.confidence) < 0.02
    assert np.mean([cer(r.text, g) for r, g in zip(res, gt)]) < 0.1
