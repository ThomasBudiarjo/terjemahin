"""Synthetic data generator, OCR fine-tuning and ONNX export (training/)."""

import json
import shutil
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


UNIFONT = Path("/usr/share/fonts/opentype/unifont")
WQY = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def test_cmap_is_exact_and_tofu_fonts_are_rejected():
    csur = UNIFONT / "unifont_csur.otf"
    if not csur.exists():
        pytest.skip("unifont not installed")
    # unifont_csur defines U+F0000 (CSUR private use) but no kana: its tofu used to pass
    assert synth.has_glyph(str(csur), 0, "\U000f0000")
    assert not synth.has_glyph(str(csur), 0, "あ")
    assert synth.probe_font(str(csur), 0) is None
    jp = UNIFONT / "unifont_jp.otf"
    if jp.exists():  # draws kana, but as 16px bitmaps
        assert synth.has_glyph(str(jp), 0, "あ")
        assert synth.is_pixel_font(synth.load_font(str(jp), 0, 32))
        assert synth.probe_font(str(jp), 0) is None


def test_default_font_search_finds_only_japanese_outline_faces():
    faces = synth.find_fonts(synth.FONT_DIRS)
    for f in faces:
        assert synth.is_japanese_face(f.path, f.index), f
        assert not synth.is_pixel_font(synth.load_font(f.path, f.index, 32)), f
        assert "unifont" not in f.path and "wqy" not in f.path, f
    assert len({(Path(f.path).resolve(), f.index) for f in faces}) == len(faces)


def test_chinese_faces_need_opt_in_and_every_ttc_face_is_probed():
    if WQY.exists():
        assert synth.face_count(str(WQY)) == 3
        assert not synth.is_japanese_face(str(WQY), 0)
        assert synth.find_fonts([WQY]) == []
        assert len(synth.find_fonts([WQY], any_cjk=True)) == 3
    noto = REF_FONTS / "NotoSansMonoCJK-VF.ttf.ttc"
    if noto.exists():  # JP, KR, SC, TC, HK: only the JP face, but all five draw kana
        assert [f.index for f in synth.find_fonts([noto])] == [0]
        assert len(synth.find_fonts([noto], any_cjk=True)) == 5
    ms = REF_FONTS / "msgothic.ttc"
    if ms.exists():
        assert [f.index for f in synth.find_fonts([ms])] == [0, 1, 2]


MSGOTHIC = REF_FONTS / "msgothic.ttc"


@pytest.mark.skipif(not MSGOTHIC.exists(), reason="msgothic.ttc not available")
def test_bitmap_strike_sizes_are_supersampled_from_outlines():
    """msgothic ships 1-bit strikes for 7-22px; drawing them gives aliased glyphs and
    horizontal-position 、。 in vertical text, so those sizes are rendered 2x and downscaled."""
    path = str(MSGOTHIC)
    assert synth.bitmap_strikes(path, 0) == frozenset(range(7, 23))
    assert synth.render_scale(path, 0, 16) == 2 and synth.render_scale(path, 0, 23) == 1
    face = synth.FontFace(path, 0, True)
    for size in (12, 16, 22):
        assert synth.is_pixel_font(synth.load_font(path, 0, size))
        fill, _ = synth.render_block(["ごめん、遅く"], face, size, True, 0,
                                     np.random.default_rng(0))
        ink = fill > 0
        assert ((fill > 0) & (fill < 255)).sum() > 0.2 * ink.sum()  # antialiased edges
        assert 0.7 * size < fill.shape[1] < 1.4 * size  # downscaled back to the asked size
    for size in (16, 22):  # the strike's 、 is a 3px dot that vanishes in a degraded crop
        fill, _ = synth.render_block(["ん、"], face, size, True, 0, np.random.default_rng(0))
        rows = np.nonzero(fill.any(1))[0]
        gap = np.nonzero(np.diff(rows) > 1)[0][-1]
        comma = fill[rows[gap + 1] :]
        assert np.ptp(np.nonzero(comma.any(1))[0]) + 1 >= 0.25 * size
        assert np.nonzero(comma.any(0))[0].mean() > 0.6 * fill.shape[1]  # top-right of cell


@needs_font
def test_emit_rejects_unknown_stages_and_only_creates_what_it_writes(tmp_path):
    corpus = synth.load_corpus(None, None)
    with pytest.raises(ValueError, match="ocrr"):
        synth.generate(tmp_path / "x", 1, 0, FONTS, corpus, emit=frozenset({"det", "ocrr"}))
    with pytest.raises(SystemExit):
        synth.main(["--out", str(tmp_path / "y"), "--n", "1", "--emit", "det,ocrr"])
    synth.generate(tmp_path / "z", 2, 0, FONTS, corpus, preview=0, emit=frozenset({"ocr"}))
    assert sorted(p.name for p in (tmp_path / "z").iterdir()) == ["meta.json", "ocr",
                                                                   "pages.jsonl"]
    assert (tmp_path / "z" / "ocr" / "train.jsonl").read_text().strip()


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
    long = tok.encode("あ" * 500, 16)
    assert len(long) == 16 and long[-1] != 3  # a cut target must not teach "stop here"
    assert len(tok.encode("あ" * 14, 16)) == 16 and tok.encode("あ" * 14, 16)[-1] == 3
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
    """Train 1 step, export, and load the export the way the product does. The ~1.3GB of
    checkpoint + ONNX is deleted afterwards, since pytest keeps old basetemps around."""
    try:
        _train_export_and_load(tiny_dataset, tmp_path)
    finally:
        shutil.rmtree(tmp_path / "run", ignore_errors=True)
        (tmp_path / "ft.onnx").unlink(missing_ok=True)


def _train_export_and_load(tiny_dataset, tmp_path):
    import export_ocr_onnx
    import train_ocr

    empty = tmp_path / "empty.jsonl"  # an empty extra eval set is skipped, not a crash
    empty.touch()
    t0 = time.time()
    args = train_ocr.parse_args([
        "--data", str(tiny_dataset), "--out", str(tmp_path / "run"), "--steps", "1",
        "--batch", "2", "--eval-every", "1", "--max-eval", "2", "--threads", "2",
        "--max-len", "48", "--eval", str(empty),
    ])
    info = train_ocr.train(args)
    assert len(info["history"]) == 1 and np.isfinite(info["history"][0]["loss"])
    assert info["history"][0]["loss"] < 1.0  # stock model already reads clean synthetic text
    assert info["final"]["val"]["cer"] is not None
    ckpt = tmp_path / "run" / "final"
    assert (ckpt / "vocab.txt").exists() and (ckpt / "config.json").exists()
    assert ckpt.is_symlink() and not (tmp_path / "run" / "best").is_symlink()  # one copy
    assert time.time() - t0 < 90

    onnx_path = tmp_path / "ft.onnx"
    crops = export_ocr_onnx.load_crops([tiny_dataset / "val.jsonl"], 2)
    export_ocr_onnx.export(str(ckpt), onnx_path)
    rep = export_ocr_onnx.check_logits(str(ckpt), onnx_path, crops)
    assert rep["max_abs"] < 1e-3 and rep["argmax_agree"] == 1.0
    dec = export_ocr_onnx.check_decode(str(ckpt), onnx_path, crops)  # via OCRS weights=
    assert dec["split_graph"] and dec["same"] == dec["n"] == len(crops)
    assert export_ocr_onnx.vocab_path(onnx_path).read_text() == (ckpt / "vocab.txt").read_text()

    from terjemahin_worker.config import Settings
    from terjemahin_worker.models import BBox, TextRegion
    from terjemahin_worker.registry import OCRS, load_all

    load_all()
    page = np.full((400, 400, 3), 255, np.uint8)
    h, w = crops[0].shape[:2]
    s = min(1.0, 360 / max(h, w))
    crop = cv2.resize(crops[0], (max(1, int(w * s)), max(1, int(h * s))))
    page[20 : 20 + crop.shape[0], 20 : 20 + crop.shape[1]] = crop
    region = TextRegion(id=0, bbox=BBox(x1=20, y1=20, x2=20 + crop.shape[1],
                                        y2=20 + crop.shape[0]))
    for kw in ({"weights": str(onnx_path)}, {}):
        ocr = OCRS.build("manga_ocr", Settings(), **kw)
        (res,) = ocr.recognize(page, [region])
        assert res.text, kw
    with pytest.raises(FileNotFoundError, match="nope.onnx"):
        OCRS.build("manga_ocr", Settings(), weights=str(tmp_path / "nope.onnx"))
