"""Synthetic data generator, OCR fine-tuning and ONNX export (training/)."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

import synth

REF_FONTS = Path("/home/user/zyddnys/manga-image-translator/fonts")
CANDIDATES = [
    REF_FONTS / "msgothic.ttc",
    REF_FONTS / "NotoSansMonoCJK-VF.ttf.ttc",
    Path("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"),
    Path("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"),
]


def _fonts():
    faces = synth.find_fonts([p for p in CANDIDATES if p.exists()][:2])
    return [f for f in faces if f.vertical][:2]


FONTS = _fonts()
needs_font = pytest.mark.skipif(not FONTS, reason="no vertical-capable Japanese font")


def _hf_cached(repo: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


needs_mangaocr = pytest.mark.skipif(
    not _hf_cached("kha-white/manga-ocr-base"), reason="manga-ocr-base not in HF cache"
)


# --- fast, offline ---------------------------------------------------------------------------


def test_wrap_keeps_closing_punctuation_off_line_starts():
    lines = synth.wrap("ちょっと待ってよ！！うん", 4)
    assert "".join(lines) == "ちょっと待ってよ！！うん"
    assert all(ln[0] not in synth.NO_START for ln in lines)
    assert synth.wrap("あいう\nえお", 10) == ["あいう", "えお"]


def test_corpus_is_filtered_by_vocab_after_nfkc():
    vocab = synth.load_vocab(synth.DEFAULT_VOCAB)
    if vocab is None:
        pytest.skip("vocab missing")
    lines = synth.load_corpus(None, vocab)
    assert len(lines) > 100  # ！？… are in the vocab once NFKC'd to !?...
    assert synth.ocr_target("え？…！") == "え?...!"


def test_split_corpus_is_disjoint_and_deterministic():
    lines = [f"l{i}" for i in range(40)]
    tr, va = synth.split_corpus(lines, 0.25, 3)
    assert len(va) == 10 and not set(tr) & set(va) and set(tr) | set(va) == set(lines)
    assert synth.split_corpus(lines, 0.25, 3) == (tr, va)
    assert synth.split_corpus(["a"], 0.5, 0) == (["a"], ["a"])


def test_procedural_background_shape_and_range():
    rng = np.random.default_rng(0)
    page = synth.procedural_background(rng, 640, 900)
    assert page.shape == (900, 640, 3) and page.dtype == np.uint8
    assert 30 < page.mean() < 250


def test_shape_mask_contains_text_box():
    rng = np.random.default_rng(1)
    for kind in ("ellipse", "cloud", "spiky", "rect"):
        m, ox, oy = synth.shape_mask(rng, kind, 120, 200)
        assert m[oy : oy + 200, ox : ox + 120].mean() > 0.97, kind


@needs_font
def test_vertical_columns_share_a_centre_line():
    face, size = FONTS[0], 40
    guo = synth._guo_width(face, size)
    shift = synth._col_shift(face, size, "ーー")
    bar = synth._render(synth.load_font(face.path, face.index, size), "ー", "ttb")
    bar_w = np.ptp(np.nonzero(bar.any(0))[0]) + 1
    assert abs(shift + bar_w / 2 - guo / 2) <= 3  # ー centred under 国, not flush left
    assert synth._col_shift(face, size, "、") > guo / 3  # vertical comma sits top-right
    fill, _ = synth.render_block(["あいう", "えお"], face, size, True, 0,
                                 np.random.default_rng(0))
    h, w = fill.shape
    assert h > 2.5 * size and 1.8 * size < w < 3.5 * size


@needs_font
def test_generate_emits_consistent_datasets(tmp_path):
    t0 = time.time()
    meta = synth.generate(tmp_path, 6, 7, FONTS, synth.load_corpus(None, None), [], 0.34,
                          preview=4)
    assert time.time() - t0 < 30
    assert meta["pages"] == 6 and meta["blocks"] > 6 and meta["val_pages"] == 2

    for split in ("train", "val"):
        imgs = sorted((tmp_path / "det" / "images" / split).glob("*.jpg"))
        assert imgs
        for img in imgs:
            label = tmp_path / "det" / "labels" / split / (img.stem + ".txt")
            rows = [r.split() for r in label.read_text().splitlines()]
            assert rows and all(r[0] == "0" and len(r) == 5 for r in rows)
            vals = np.array([[float(v) for v in r[1:]] for r in rows])
            assert ((vals > 0) & (vals <= 1)).all()
    assert "names:\n  0: text" in (tmp_path / "det" / "dataset.yaml").read_text()

    pages = [json.loads(x) for x in (tmp_path / "pages.jsonl").read_text().splitlines()]
    texted = cv2.imread(str(tmp_path / "inpaint" / "texted" / "000000.png"))
    clean = cv2.imread(str(tmp_path / "inpaint" / "clean" / "000000.png"))
    mask = cv2.imread(str(tmp_path / "inpaint" / "mask" / "000000.png"), 0)
    assert texted.shape == clean.shape and mask.shape == texted.shape[:2]
    diff = np.abs(texted.astype(int) - clean.astype(int)).max(2) > 10
    # every changed pixel is covered by the mask, and every block box holds text pixels
    assert diff.sum() > 100 and (mask[diff] > 0).mean() > 0.999
    for b in pages[0]["blocks"]:
        x1, y1, x2, y2 = b["bbox"]
        assert diff[y1:y2, x1:x2].sum() > 0

    rows = [json.loads(x) for s in ("train", "val")
            for x in (tmp_path / "ocr" / f"{s}.jsonl").read_text().splitlines()]
    assert len(rows) == meta["blocks"]
    for r in rows:
        crop = cv2.imread(str(tmp_path / "ocr" / r["image"]))
        assert crop is not None and crop.size and r["text"]
    assert (tmp_path / "preview.jpg").exists() and (tmp_path / "ocr_preview.jpg").exists()


@needs_font
def test_generate_is_deterministic(tmp_path):
    corpus = synth.load_corpus(None, None)
    synth.generate(tmp_path / "a", 3, 11, FONTS, corpus, [], 0.34, preview=0)
    synth.generate(tmp_path / "b", 3, 11, FONTS, corpus, [], 0.34, preview=0, workers=2)
    for f in sorted((tmp_path / "a").rglob("*")):
        if f.is_file() and f.name not in ("dataset.yaml", "meta.json"):
            assert f.read_bytes() == (tmp_path / "b" / f.relative_to(tmp_path / "a")).read_bytes()


def test_tokenizer_matches_manga_ocr_targets():
    import train_ocr

    vocab = train_ocr.load_vocab(synth.DEFAULT_VOCAB) if synth.DEFAULT_VOCAB.exists() else None
    if vocab is None:
        pytest.skip("vocab missing")
    tok = train_ocr.Tokenizer(vocab)
    ids = tok.encode("え？ …", 64)
    assert ids[0] == 2 and ids[-1] == 3  # [CLS] ... [SEP], as manga-ocr was trained
    assert tok.decode(ids) == "え？．．．"  # what the worker outputs after normalize_ja
    assert len(tok.encode("あ" * 500, 16)) == 16
    assert train_ocr.edit_distance("abc", "abd") == 1 and train_ocr.edit_distance("", "ab") == 2


# --- real model (HF cache) -------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    if not FONTS:
        pytest.skip("no font")
    out = tmp_path_factory.mktemp("synth")
    synth.generate(out, 3, 5, FONTS, synth.load_corpus(None, None), [], 0.34, preview=0,
                   emit=frozenset({"ocr"}))
    return out / "ocr"


@needs_font
@needs_mangaocr
def test_one_training_step_and_onnx_export(tiny_dataset, tmp_path):
    import train_ocr

    t0 = time.time()
    args = train_ocr.parse_args([
        "--data", str(tiny_dataset), "--out", str(tmp_path / "run"), "--steps", "1",
        "--batch", "2", "--eval-every", "1", "--max-eval", "2", "--threads", "2",
        "--max-len", "48",
    ])
    info = train_ocr.train(args)
    assert len(info["history"]) == 1 and np.isfinite(info["history"][0]["loss"])
    assert info["history"][0]["loss"] < 1.0  # stock model already reads clean synthetic text
    assert info["final"]["val"]["cer"] is not None
    ckpt = tmp_path / "run" / "final"
    assert (ckpt / "vocab.txt").exists() and (ckpt / "config.json").exists()
    assert time.time() - t0 < 90

    import export_ocr_onnx

    onnx_path = tmp_path / "ft.onnx"
    crops = export_ocr_onnx.load_crops([tiny_dataset / "val.jsonl"], 2)
    export_ocr_onnx.export(str(ckpt), onnx_path)
    rep = export_ocr_onnx.check_logits(str(ckpt), onnx_path, crops)
    assert rep["max_abs"] < 1e-3 and rep["argmax_agree"] == 1.0
    dec = export_ocr_onnx.check_decode(str(ckpt), onnx_path, crops)
    assert dec["split_graph"] and dec["same"] == dec["n"] == len(crops)
    assert export_ocr_onnx.vocab_path(onnx_path).read_text() == (ckpt / "vocab.txt").read_text()
