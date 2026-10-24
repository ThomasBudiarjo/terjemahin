"""Regression tests for the robustness sweep: odd inputs, hand-edited results, bad options and
long-lived servers. Each test names the bug it guards against."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from terjemahin_worker.config import Settings
from terjemahin_worker.detect import Detection
from terjemahin_worker.imageio import ImageError, decode_image, read_image, to_bgr8
from terjemahin_worker.models import BBox, Bubble, PageResult, PipelineConfig, TextRegion
from terjemahin_worker.ocr.base import OcrResult
from terjemahin_worker.pipeline import Pipeline
from terjemahin_worker.registry import (
    DETECTORS,
    INPAINTERS,
    OCRS,
    RENDERERS,
    TRANSLATORS,
    load_all,
)
from terjemahin_worker.translate import textproc as tp
from terjemahin_worker.translate.base import TranslationItem

load_all()

FIXTURE = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "page01.png"

# --- fake stages: a page with two boxes whose OCR confidence the test chooses ---------------

BOXES = [BBox(x1=40, y1=40, x2=140, y2=100), BBox(x1=220, y1=40, x2=320, y2=100)]
READS: dict[str, list[tuple[str, float]]] = {"reads": [("テスト", 0.95), ("テスト", 0.95)]}


class FakeDetector:
    def detect(self, img):
        return Detection([TextRegion(id=i, bbox=b, direction="h") for i, b in enumerate(BOXES)])


class FakeOCR:
    def recognize(self, img, regions):
        out = []
        for r in regions:
            i = BOXES.index(r.bbox)
            text, conf = READS["reads"][i]
            out.append(OcrResult(text=text, confidence=conf))
        return out


class EmptyTranslator:
    vision = False

    def translate(self, req):
        return [TranslationItem(source=s, target="") for s in req.sources]


def _register() -> None:
    fakes = [
        (DETECTORS, "_rb_fake", lambda s, **_: FakeDetector()),
        (OCRS, "_rb_fake", lambda s, **_: FakeOCR()),
        (TRANSLATORS, "_rb_empty", lambda s, **_: EmptyTranslator()),
    ]
    for reg, name, factory in fakes:
        if name not in reg.methods:
            reg.register(name, description="test fake")(factory)


_register()
CHEAP = {"detector": "_rb_fake", "ocr": "_rb_fake", "translator": "original",
         "mask_refiner": "bbox", "inpainter": "fill", "renderer": "basic"}


def _page() -> np.ndarray:
    img = np.full((160, 380, 3), 255, np.uint8)
    for b in BOXES:
        cv2.putText(img, "TXT", (b.x1 + 10, b.y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    return img


def _changed(a: np.ndarray, b: np.ndarray, box: BBox) -> int:
    sl = (slice(box.y1, box.y2), slice(box.x1, box.x2))
    return int((np.abs(a[sl].astype(int) - b[sl].astype(int)).max(axis=2) > 0).sum())


@pytest.fixture(scope="module")
def pipe() -> Pipeline:
    return Pipeline(Settings())


# --- pipeline / detect+OCR gating (text-free balloons, rotated pages) -----------------------

def test_low_confidence_ocr_read_leaves_the_region_untouched(pipe):
    # an empty balloon read as 'そういう' at confidence 0.30 was erased, smeared and typeset
    READS["reads"] = [("そういう", 0.30), ("テスト", 0.95)]
    page = _page()
    out = pipe.run(page, config=PipelineConfig(**CHEAP))
    assert [b.region.bbox for b in out.result.bubbles] == [BOXES[1]]
    assert _changed(page, out.image, BOXES[0]) == 0
    assert _changed(page, out.image, BOXES[1]) > 0
    # the threshold is a per-job setting
    out = pipe.run(page, config=PipelineConfig(**CHEAP, min_ocr_confidence=0.1))
    assert len(out.result.bubbles) == 2
    READS["reads"] = [("テスト", 0.95), ("テスト", 0.95)]


# --- translate textproc + compose (punctuation-only bubbles, empty translations) ------------

@pytest.mark.parametrize(
    ("src", "want"),
    [("？", "?"), ("……", "..."), ("！！", "!!"), ("！？", "!?"), ("・・・", "..."),
     ("．．．．．", "..."), ("？？", "??")],
)
def test_punctuation_only_bubbles_keep_their_punctuation(src, want):
    p = tp.prepare(src, "ja", "id", [], True, True, True)
    assert p.text == "" and tp.finish(p, "", "id") == want
    assert tp.finish(tp.prepare(src, "en", "id"), "", "id") == want


def test_punctuation_only_bubble_goes_through_seq2seq_without_the_model(monkeypatch):
    tr = TRANSLATORS.build("m2m100", Settings())
    monkeypatch.setattr(tr, "generate", lambda texts, *a, **k: [] if not texts else pytest.fail("model called"))
    assert tr.translate_texts(["？", "……"], "ja", "id") == ["?", "..."]


def test_letterless_translation_of_words_is_bad():
    # chain m2m100:en,nllb -> zh turned 'Uh-huh' into ',.', which finish() reduced to ''
    assert tp.is_bad("Uh-huh", ",.", "zh")
    assert tp.is_bad("うんうん", "...", "id")
    assert not tp.is_bad("？", "?", "id")
    assert not tp.is_bad("うんうん", "Iya, iya", "id")


def test_empty_translation_keeps_the_original_text(pipe, caplog):
    page = _page()
    with caplog.at_level(logging.WARNING):
        out = pipe.run(page, config=PipelineConfig(**{**CHEAP, "translator": "_rb_empty"}))
    assert all(b.erase is False for b in out.result.bubbles)
    assert np.array_equal(out.image, page)  # not erased and left blank
    assert "empty translation" in caplog.text
    # translator 'none' blanks on purpose: it still erases
    out = pipe.run(page, config=PipelineConfig(**{**CHEAP, "translator": "none"}))
    assert all(b.erase for b in out.result.bubbles)
    assert _changed(page, out.image, BOXES[0]) > 0


# --- compose: hand-edited and user-added bubbles --------------------------------------------

def test_user_added_bubble_without_source_is_drawn_not_erased(pipe):
    page = cv2.imread(str(FIXTURE))
    note = Bubble(region=TextRegion(id=0, bbox=BBox(x1=150, y1=650, x2=450, y2=900)),
                  target_text="Catatan penerjemah")
    dbg: dict = {}
    pipe.compose(page, [note], PipelineConfig(inpainter="fill", renderer="basic"), debug=dbg)
    assert not dbg["mask"].any()  # it erased 12.5k px of hair highlights
    assert note.model_copy(update={"erase": True}).erases()
    assert Bubble(region=note.region, source_text="x").erases()


def test_compose_bounds_a_hand_edited_font_size(pipe):
    page = cv2.imread(str(FIXTURE))
    box = BBox(x1=1144, y1=218, x2=1225, y2=334)
    b = Bubble(region=TextRegion(id=0, bbox=box, font_size=5000), source_text="坂本くん？",
               target_text="Sakamoto-kun?")
    (s,) = Pipeline.sanitize(page, [b])
    assert s.region.font_size <= round(1.5 * min(box.w, box.h))
    t = time.perf_counter()
    pipe.compose(page, [b], PipelineConfig(inpainter="fill"))  # default mask refiner
    assert time.perf_counter() - t < 20  # font_size=5000 ran for > 10 min


def test_inverted_and_partly_off_page_boxes(pipe):
    page = cv2.imread(str(FIXTURE))
    inverted = Bubble(region=TextRegion(id=0, bbox=BBox(x1=450, y1=900, x2=150, y2=650)),
                      source_text="x", target_text="Terbalik")
    (s,) = Pipeline.sanitize(page, [inverted])
    assert (s.region.bbox.x1, s.region.bbox.y1, s.region.bbox.x2, s.region.bbox.y2) == (150, 650, 450, 900)
    for renderer in ("basic", "default"):  # basic crashed with 'cannot convert float NaN'
        out = pipe.compose(page, [inverted], PipelineConfig(inpainter="fill", renderer=renderer))
        assert _changed(page, out, s.region.bbox) > 0
    off = Bubble(region=TextRegion(id=0, bbox=BBox(x1=-100, y1=-50, x2=150, y2=120)),
                 source_text="x", target_text="Sebagian di luar")
    (s,) = Pipeline.sanitize(page, [off])
    assert (s.region.bbox.x1, s.region.bbox.y1) == (0, 0)
    gone = Bubble(region=TextRegion(id=0, bbox=BBox(x1=-300, y1=10, x2=-100, y2=90)), source_text="x")
    assert Pipeline.sanitize(page, [gone]) == []


def test_basic_renderer_lays_out_in_the_visible_part_of_the_box():
    from terjemahin_worker.render.basic import _on_page

    assert _on_page(BBox(x1=-100, y1=-50, x2=150, y2=120), 400, 300) == BBox(x1=0, y1=0, x2=150, y2=120)
    assert _on_page(BBox(x1=150, y1=120, x2=-100, y2=-50), 400, 300) == BBox(x1=0, y1=0, x2=150, y2=120)
    assert _on_page(BBox(x1=500, y1=0, x2=600, y2=50), 400, 300) is None


def test_estimate_background_of_an_empty_box_is_white():
    from terjemahin_worker.clean import estimate_background

    img = np.full((1000, 1000, 3), 128, np.uint8)
    assert estimate_background(img, BBox(x1=450, y1=900, x2=150, y2=650)) == (255, 255, 255)


# --- input decoding and the Pipeline API contract -------------------------------------------

def test_transparent_png_is_composited_over_white(tmp_path):
    page = cv2.imread(str(FIXTURE))
    rgba = cv2.cvtColor(page, cv2.COLOR_BGR2BGRA)
    white = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) > 245
    rgba[white] = (0, 0, 0, 0)  # how exporters store transparent paper
    p = tmp_path / "rgba.png"
    cv2.imwrite(str(p), rgba)
    assert cv2.imread(str(p)).mean() < 100  # what IMREAD_COLOR made of it: black balloons
    img = read_image(p)
    assert img.shape == page.shape and abs(img.mean() - page.mean()) < 2


def test_decode_rejects_empty_unreadable_and_oversized():
    with pytest.raises(ImageError, match="empty"):
        decode_image(b"")
    with pytest.raises(ImageError, match="not a readable image"):
        decode_image(b"x\n")
    big = cv2.imencode(".png", np.full((300, 400, 3), 255, np.uint8))[1].tobytes()
    with pytest.raises(ImageError, match="limit"):
        decode_image(big, max_pixels=100_000)
    assert decode_image(big, max_pixels=200_000).shape == (300, 400, 3)


@pytest.mark.parametrize(
    "arr",
    [
        np.full((20, 30), 200, np.uint8),
        np.full((20, 30, 1), 200, np.uint8),
        np.dstack([np.full((20, 30, 3), 200, np.uint8), np.full((20, 30), 255, np.uint8)]),
        np.full((20, 30, 3), 200 * 257, np.uint16),
        np.full((20, 30, 3), 200 / 255, np.float32),
    ],
)
def test_to_bgr8_normalises_api_inputs(arr):
    out = to_bgr8(arr)
    assert out.shape == (20, 30, 3) and out.dtype == np.uint8 and abs(int(out[0, 0, 0]) - 200) <= 1


def test_pipeline_run_accepts_greyscale_and_float_pages(pipe):
    page = cv2.cvtColor(_page(), cv2.COLOR_BGR2GRAY)
    out = pipe.run(page, config=PipelineConfig(**CHEAP))
    assert out.image.shape == (*page.shape, 3) and len(out.result.bubbles) == 2
    out = pipe.run(_page().astype(np.float32) / 255, config=PipelineConfig(**CHEAP))
    assert out.image.dtype == np.uint8 and len(out.result.bubbles) == 2


# --- render/default (tall webtoon strips, CJK, typed line breaks) ---------------------------

def test_page_minimum_font_size_ignores_strip_height():
    from terjemahin_worker.render.default import page_min_size

    assert page_min_size(1487, 2048) == 18  # page01 as before
    assert page_min_size(800, 12000) == 10  # was 64 px: every balloon overflowed
    assert page_min_size(12000, 800) == 10


def test_tall_strip_bubbles_fit():
    from terjemahin_worker.render.default import DefaultRenderer

    page = np.full((12000, 800, 3), 255, np.uint8)
    cv2.ellipse(page, (400, 6000), (150, 90), 0, 0, 360, (0, 0, 0), 3)
    b = Bubble(region=TextRegion(id=0, bbox=BBox(x1=330, y1=5950, x2=470, y2=6050), font_size=20),
               source_text="x", target_text="Hei, hei, apa yang kau pikirkan?")
    r = DefaultRenderer(Settings().font_path)
    r.render(page, [b], page)
    assert r.plans[0].layout.fits


def test_typed_line_breaks_are_kept():
    from terjemahin_worker.render.layout import FontMeasure, Shape, fit
    from terjemahin_worker.typeset import Typesetter

    m = FontMeasure(Settings().font_path)
    lay = fit("Baris satu\nBaris dua\n\nBaris empat", Shape.rect(600, 400), m, max_size=30)
    assert [ln.text for ln in lay.lines] == ["Baris satu", "Baris dua", "", "Baris empat"]
    assert lay.fits and lay.text == "Baris satu\nBaris dua\n\nBaris empat"
    assert [ln.text for ln in fit("a b c", Shape.rect(600, 400), m, max_size=30).lines] == ["a b c"]
    from PIL import Image, ImageDraw

    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    t = Typesetter(Settings().font_path).layout(d, "Baris satu\nBaris dua\n\nBaris empat", 600, 400)
    assert t.lines == ["Baris satu", "Baris dua", "", "Baris empat"]


def test_horizontal_cjk_text_uses_a_cjk_font():
    from terjemahin_worker.render.default import DefaultRenderer
    from terjemahin_worker.render.vertical import find_cjk_font

    if not find_cjk_font():
        pytest.skip("no CJK font installed")
    page = np.full((400, 400, 3), 255, np.uint8)
    b = Bubble(region=TextRegion(id=0, bbox=BBox(x1=50, y1=50, x2=350, y2=150), direction="h"),
               source_text="x", target_text="我-我... 不是这样...")
    r = DefaultRenderer(Settings().font_path)
    r.render(page, [b], page)
    assert r.plans[0].measure is not r.measure and r.plans[0].measure.has_glyph("我")


def test_single_line_sits_in_the_middle_of_its_balloon():
    from terjemahin_worker.render.layout import FontMeasure, Shape, fit

    mask = np.zeros((150, 200), np.uint8)
    cv2.ellipse(mask, (100, 75), (95, 70), 0, 0, 360, 255, -1)
    lay = fit("?", Shape.from_mask(mask > 0), FontMeasure(Settings().font_path), max_size=40)
    assert 40 < lay.lines[0].y + lay.size / 2 < 110  # a lone '?' sat at the top


def test_punctuation_only_balloons_are_drawn_at_the_source_size():
    # '?' and '……' were drawn at a third of the source marks (size from the balloon alone)
    from terjemahin_worker.render.default import DefaultRenderer

    page = np.full((400, 700, 3), 255, np.uint8)
    cv2.ellipse(page, (150, 200), (120, 70), 0, 0, 360, (0, 0, 0), 2)
    cv2.ellipse(page, (500, 200), (100, 70), 0, 0, 360, (0, 0, 0), 2)
    q = Bubble(region=TextRegion(id=0, bbox=BBox(x1=480, y1=170, x2=519, y2=231), direction="v",
                                 font_size=38), source_text="？", target_text="?")
    dots = Bubble(region=TextRegion(id=1, bbox=BBox(x1=94, y1=195, x2=207, y2=206),
                                    direction="h", font_size=11),
                  source_text="．．．．．", target_text="...")
    r = DefaultRenderer(Settings().font_path)
    r.render(page, [q, dots], page)
    pq, pd = r.plans
    _, t, _, b = pq.measure.font(pq.layout.size).getbbox("?")
    assert 0.75 * 61 <= b - t <= 1.25 * 61, pq.layout.size  # was ~20 px of ink for 61
    assert pd.text == "..."  # three dots, not one narrow ellipsis glyph
    l, _, rr, _ = pd.measure.font(pd.layout.size).getbbox(pd.text)
    assert 0.75 * 113 <= rr - l <= 1.25 * 113, pd.layout.size


# --- options: CLI parsing and validation at build time --------------------------------------

def test_cli_opt_booleans_aliases_and_unknown_stages():
    import typer

    from terjemahin_worker.cli import _parse_opts

    got = _parse_opts(["renderer.uppercase=False", "render.hyphenate=off", "mask.pad=40",
                       "inpaint.ring=5", "translator.threads=None", "detector.conf=0.3"])
    assert got == {"renderer": {"uppercase": False, "hyphenate": False},
                   "mask_refiner": {"pad": 40}, "inpainter": {"ring": 5},
                   "translator": {"threads": None}, "detector": {"conf": 0.3}}
    with pytest.raises(typer.BadParameter, match="unknown stage"):
        _parse_opts(["detecter.conf=0.3"])


def test_renderer_bool_strings_are_parsed():
    from terjemahin_worker.render.default import DefaultRenderer

    r = DefaultRenderer(Settings().font_path, uppercase="False", hyphenate="false")
    assert r.uppercase is False and r.hyphenate is False
    with pytest.raises(ValueError, match="uppercase"):
        DefaultRenderer(Settings().font_path, uppercase="maybe")


@pytest.mark.parametrize(
    ("stage", "method", "opts", "match"),
    [
        ("detector", "ctd", {"conf": "abc"}, "detector conf"),
        ("detector", "ctd", {"conf": -1}, "detector conf"),
        ("detector", "ctd", {"tile_aspect": 0}, "tile_aspect"),
        ("detector", "ctd", {"box_padding": -50}, "box_padding"),
        ("detector", "ctd_blk", {"iou": "x"}, "detector iou"),
        ("detector", "classical", {"bright": "x"}, "bright"),
        ("ocr", "manga_ocr", {"pad": "x"}, "ocr pad"),
        ("ocr", "manga_ocr", {"max_len": 0}, "max_len"),
        ("inpainter", "fill", {"ring": -3}, "ring"),
        ("inpainter", "fill", {"ring": 2.5}, "ring"),
        ("inpainter", "telea", {"radius": "x"}, "radius"),
        ("inpainter", "lama_manga", {"size": 1024}, "size must be 512"),
        ("inpainter", "lama_manga", {"max_down": 0}, "max_down"),
        ("mask_refiner", "bbox", {"pad": -500}, "pad"),
        ("renderer", "default", {"line_spacing": 0}, "line_spacing"),
        ("renderer", "default", {"stroke_width": -3}, "stroke_width"),
        ("renderer", "default", {"fg_color": "red"}, "fg_color"),
        ("renderer", "default", {"fg_color": [255, 0]}, "fg_color"),
        ("renderer", "default", {"fg_color": [300, -5, 0]}, "fg_color"),
        ("renderer", "default", {"min_font_size": 40, "max_font_size": 10}, "larger than"),
        ("renderer", "default", {"font_path": "/nonexistent.ttf"}, "no such font"),
        ("renderer", "default", {"cjk_font_path": "/nonexistent.ttf"}, "no such font"),
        ("renderer", "basic", {"font_path": "/nonexistent.ttf"}, "no such font"),
        ("translator", "nllb", {"max_tokens": 0}, "max_tokens"),
        ("translator", "nllb", {"model": "facebook/m2m100_418M"}, "m2m100 checkpoint"),
        ("translator", "m2m100", {"pre_dict": "x"}, "pre_dict"),
        ("translator", "nllb", {"names": "maybe"}, "names"),
        ("translator", "chain", {"steps": "[1,2]"}, "chain step"),
        ("translator", "chain", {"steps": "nllb:klingon,nllb"}, "unsupported language"),
        ("translator", "ollama", {"fallback": "bogus"}, "unknown translator 'bogus'"),
        ("translator", "ollama", {"base_url": "notaurl"}, "base_url"),
        ("translator", "local_llm", {"pivot_translator": "bogus"}, "unknown translator"),
    ],
)
def test_bad_option_values_fail_at_build_with_the_option_named(stage, method, opts, match):
    from terjemahin_worker.registry import REGISTRIES

    with pytest.raises(ValueError, match=match):
        REGISTRIES[stage].build(method, Settings(), **opts)


def test_unknown_options_are_reported(caplog):
    with caplog.at_level(logging.WARNING):
        INPAINTERS.build("fill", Settings(), rnig=5)
        RENDERERS.build("basic", Settings(), font_size=3)
    assert "rnig" in caplog.text and "font_size" in caplog.text


def test_check_fails_fast_before_any_stage_runs(pipe):
    with pytest.raises(ValueError, match="unknown renderer 'bogus'"):
        pipe.check(pipe.resolve(PipelineConfig(**{**CHEAP, "renderer": "bogus"})))
    with pytest.raises(ValueError, match="unsupported language 'klingon'"):
        pipe.check(pipe.resolve(PipelineConfig(**CHEAP, target_lang="klingon")))
    # ocr 'none' with a translator that cannot read the image: an untouched page, exit 0
    with pytest.raises(ValueError, match="vision"):
        pipe.check(pipe.resolve(PipelineConfig(**{**CHEAP, "ocr": "none"})))


def test_non_japanese_source_gets_an_ocr_that_reads_it(pipe):
    assert pipe.resolve(PipelineConfig(source_lang="en")).ocr == "48px"
    assert pipe.resolve(PipelineConfig(source_lang="ko")).ocr == "48px"
    assert pipe.resolve(PipelineConfig()).ocr == "manga_ocr"
    assert pipe.resolve(PipelineConfig(source_lang="auto")).ocr == "manga_ocr"
    assert pipe.resolve(PipelineConfig(source_lang="en", ocr="manga_ocr")).ocr == "manga_ocr"


def test_reading_order_follows_the_source_language(pipe):
    out = pipe.run(_page(), config=PipelineConfig(**CHEAP, source_lang="en"))
    assert [b.region.bbox.x1 for b in sorted(out.result.bubbles, key=lambda b: b.region.id)] == [40, 220]
    out = pipe.run(_page(), config=PipelineConfig(**CHEAP))
    assert [b.region.bbox.x1 for b in sorted(out.result.bubbles, key=lambda b: b.region.id)] == [220, 40]


# --- worker model cache ---------------------------------------------------------------------

def test_stage_cache_is_bounded_per_stage():
    p = Pipeline(Settings(stage_cache_size=2))
    for ring in range(1, 7):
        p.stage("inpainter", PipelineConfig(inpainter="fill", options={"inpainter": {"ring": ring}}))
    p.stage("renderer", PipelineConfig(renderer="basic"))
    stages = [k[0] for k in p._cache]
    assert stages.count("inpainter") == 2 and stages.count("renderer") == 1
    # most recently used survive
    assert [k[2] for k in p._cache if k[0] == "inpainter"] == ["[('ring', 5)]", "[('ring', 6)]"]


def test_manga_ocr_instances_share_onnx_sessions():
    a = OCRS.build("manga_ocr", Settings(), batch_size=2)
    b = OCRS.build("manga_ocr", Settings(), batch_size=3)
    assert a is not b and a.batch_size != b.batch_size
    assert (a.encoder, a.decoder, a.full) == (b.encoder, b.decoder, b.full)  # ~1 GB per copy


def test_seq2seq_model_cache_evicts_the_least_recently_used(monkeypatch):
    from terjemahin_worker.translate import hf_seq2seq as H

    class FakeModel:
        def __init__(self):
            self.generation_config = type("G", (), {})()

        def to(self, _):
            return self

        def eval(self):
            return self

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: object()))
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained",
                        staticmethod(lambda *a, **k: FakeModel()))
    monkeypatch.setattr(H, "_MODELS", type(H._MODELS)())
    monkeypatch.setattr(H, "MAX_MODELS", 2)
    H.load_model("a")
    H.load_model("b")
    H.load_model("a")  # a is now the most recent
    H.load_model("c")
    assert [k[0] for k in H._MODELS] == ["a", "c"]


# --- translate: CJK targets, mixed pages, LLM connection errors -----------------------------

def test_chinese_target_writes_surnames_in_kanji(monkeypatch):
    p = tp.prepare("坂本くん？", "ja", "zh", [])
    assert p.terms == [] and p.kanji == {"Sakamoto": "坂本"}  # no Latin term to insist on
    assert tp.prepare("坂本くん？", "ja", "id", []).terms == ["Sakamoto"]
    assert tp.prepare("坂本のこと", "ja", "ko", []).terms == []  # Hangul output, no Latin term
    tr = TRANSLATORS.build("m2m100", Settings())
    monkeypatch.setattr(tr, "generate", lambda texts, *a, **k: ["你对Sakamoto有什么看法?" for _ in texts])
    out = tr.translate_texts(["坂本くん？", "めいは坂本のことどう思う？"], "ja", "zh")
    assert out == ["坂本君？", "你对坂本有什么看法？"]  # was Latin 'Sakamoto-kun?' on a Chinese page


@pytest.mark.parametrize(
    ("out", "romaji", "kanji", "src", "want"),
    [
        # NLLB / M2M100 transliterate the romaji they were given instead of copying it
        ("你对萨卡莫托有什么看法?", "Sakamoto", "坂本", "めいは Sakamoto のことどう思う？",
         "你对坂本有什么看法?"),
        ("萨卡莫托?", "Sakamoto", "坂本", "Sakamoto-kun？", "坂本君?"),  # dropped -kun put back
        ("萨卡莫 - 昆?", "Sakamoto", "坂本", "Sakamoto-kun？", "坂本君?"),  # last mora lost
        ("萨卡莫托·库恩", "Sakamoto", "坂本", "Sakamoto-kun", "坂本君"),
        ("塔纳卡-桑在哪里?", "Tanaka", "田中", "Tanaka-san はどこ？", "田中桑在哪里?"),
        ("我喜欢太卡哈西-昆.", "Takahashi", "高橋", "", "我喜欢高橋君."),  # one mora differs
        ("哈特里 - 桑", "Hattori", "服部", "", "服部桑"),  # small っ written
        ("哈托里-桑", "Hattori", "服部", "", "服部桑"),  # ... or not
        ("孔多桑说", "Kondo", "近藤", "", "近藤桑说"),  # mora + n in one character
        ("亚马莫托先生", "Yamamoto", "山本", "Yamamoto-sensei", "山本先生"),  # Chinese honorific
        ("你對薩卡莫托有什麼看法?", "Sakamoto", "坂本", "", "你對坂本有什麼看法?"),  # Traditional
        ("您对Sakamoto有什么看法?", "Sakamoto", "坂本", "", "您对坂本有什么看法?"),  # kept Latin
        # nothing that spells the name: left alone
        ("刚刚说:", "Kondo", "近藤", "", "刚刚说:"),
        ("他去了阿拉伯。", "Arai", "新井", "", "他去了阿拉伯。"),
    ],
)
def test_chinese_output_transliterated_surnames_become_kanji(out, romaji, kanji, src, want):
    assert tp.restore_kanji(out, {romaji: kanji}, src) == want


def test_surname_transliteration_does_not_match_ordinary_chinese():
    text = ("你对这个有什么看法？我们今天去吃饭吧。别担心，一切都会好起来的。他说他明天会来。"
            "这是我第一次来日本。快跑！这里撑不住了！真的吗？太好了！老师来了，大家安静。"
            "今天天气很好，我们去公园散步吧。对不起，我迟到了。那个女孩是谁？谢谢你帮我。"
            "我叫萨拉，来自加拿大。克里斯蒂娜和玛丽亚在巴黎。莫斯科很冷。泰国菜很辣。")
    hits = {r: s for r in set(tp.SURNAMES.values()) if (s := tp.find_transliteration(text, r))}
    assert hits == {}
    assert tp.restore_kanji("你對薩卡莫托?", {"Sakamoto": "坂本"}, "Sakamoto-chan?", hant=True) \
        == "你對坂本醬?"


class _ZhTranslit:
    """A pivot hop that transliterates romaji the way NLLB en->zh does."""

    vision = False

    def translate(self, req):
        return [TranslationItem(source=s, target="萨卡莫托?" if s.startswith("Sakamoto-kun")
                                else "你对萨卡莫托有什么看法?") for s in req.sources]


class _Echo:
    vision = False

    def translate(self, req):
        return [TranslationItem(source=s, target=s) for s in req.sources]


def test_chinese_target_surnames_in_kanji_through_seq2seq_and_chain(monkeypatch):
    # the verifier's repro: nllb --to zh gave '你对萨卡莫托有什么看法？', chain m2m100:en,nllb
    # gave '萨卡莫托？' for 坂本くん？
    tr = TRANSLATORS.build("m2m100", Settings())
    monkeypatch.setattr(tr, "generate", lambda texts, *a, **k: ["你对萨卡莫托有什么看法?" for _ in texts])
    assert tr.translate_texts(["めいは坂本のことどう思う？"], "ja", "zh") == ["你对坂本有什么看法？"]
    for name, cls in (("_rb_echo", _Echo), ("_rb_zh_translit", _ZhTranslit)):
        if name not in TRANSLATORS.methods:
            TRANSLATORS.register(name, description="test fake")(lambda s, cls=cls, **_: cls())
    chain = TRANSLATORS.build("chain", Settings(), steps="_rb_echo:en,_rb_zh_translit")
    out = chain.translate_texts(["坂本くん？", "めいは坂本のことどう思う？"], "ja", "zh")
    assert out == ["坂本君？", "你对坂本有什么看法？"]


def test_chinese_output_that_lost_a_surname_is_retranslated(monkeypatch):
    tr = TRANSLATORS.build("m2m100", Settings())
    calls = []

    def gen(texts, source, target, **overrides):
        calls.append(overrides)
        return ["刚刚说:" if not overrides else "孔多桑说" for _ in texts]

    monkeypatch.setattr(tr, "generate", gen)
    assert tr.translate_texts(["近藤さんが言った"], "ja", "zh") == ["近藤桑说"]
    assert len(calls) == 2 and calls[1]  # the strict retry ran


def test_chinese_output_punctuation_and_stutter():
    assert tp.han_punct("你好, 世界... 是吗?") == "你好，世界……是吗？"
    assert tp.stutter("我不是这样", "zh") == "我、我不是这样"
    assert tp.stutter("aku tidak", "id") == "A-aku tidak"


def test_auto_source_language_is_per_bubble(monkeypatch):
    assert tp.detect_langs(["Hey, what do you think?", "Huh?", "、らz", "..."]) == ["en", "en", "ja", "en"]
    tr = TRANSLATORS.build("m2m100", Settings())
    seen = []

    def fake(texts, source, target, **_):
        seen.append(source)
        return [f"<{source}>{t}" for t in texts]

    monkeypatch.setattr(tr, "generate", fake)
    out = tr.translate_texts(["Hey, what do you think of Sakamoto?", "Huh?", "、らz"], "auto", "id")
    assert sorted(set(seen)) == ["en", "ja"]  # one stray kana kept every English bubble English
    assert out[0].lower().startswith("<en>") and out[1].lower().startswith("<en>")


def test_llm_connection_refused_fails_fast_and_names_the_url():
    import httpx

    from terjemahin_worker.translate.llm_common import CONNECT_RETRIES, Http, LLMError

    calls = {"n": 0}

    def refuse(request):
        calls["n"] += 1
        raise httpx.ConnectError("[Errno 111] Connection refused", request=request)

    http = Http(transport=httpx.MockTransport(refuse), sleep=lambda s: None)
    with pytest.raises(LLMError, match="localhost:11434.*is the server running"):
        http.post("http://localhost:11434/v1/chat/completions", {}, {})
    assert calls["n"] == CONNECT_RETRIES + 1  # was max_retries + 1 = 5, ~20 s of backoff
    http = Http(sleep=lambda s: pytest.fail("retried a malformed URL"))
    with pytest.raises(LLMError, match="bad URL"):
        http.post("notaurl/chat/completions", {}, {})


# --- detect/ctd_full plan_tiles (extreme aspect ratios) -------------------------------------

def test_thin_strips_are_not_upscaled_without_bound():
    from terjemahin_worker.detect.ctd_full import MAX_UPSCALE, plan_tiles

    ratio, tiles = plan_tiles(8, 300, 1024, 2.5)
    assert len(tiles) == 1  # was ratio 85 and ~32 tiles, 34 s
    ratio, tiles = plan_tiles(100, 12000, 1024, 2.5)
    assert ratio <= MAX_UPSCALE and len(tiles) <= 30
    assert plan_tiles(1487, 2048, 1024, 2.5) == (1024 / 2048, [(0, 0, 1487, 2048)])
    ratio, tiles = plan_tiles(800, 12000, 1024, 2.5)
    assert ratio == pytest.approx(1024 / 1200)  # webtoons tile as before


# --- CLI ------------------------------------------------------------------------------------

def _cli(*args):
    from terjemahin_worker.cli import app

    r = CliRunner().invoke(app, list(args))
    r.text = " ".join(r.output.replace("│", " ").split())  # rich wraps errors in a box
    return r


def test_cli_rejects_unreadable_images_and_mismatched_results(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("x\n")
    res = PageResult(bubbles=[], width=1487, height=2048)
    rj = tmp_path / "r.json"
    rj.write_text(res.model_dump_json())
    for args in (["compose", str(bad), str(rj)], ["detect", str(bad), "-d", "classical"]):
        r = _cli(*args, "-o", str(tmp_path / "o.png"))
        assert r.exit_code == 2 and "not a readable image" in r.text, r.output
    small = tmp_path / "small.png"
    cv2.imwrite(str(small), np.full((64, 64, 3), 255, np.uint8))
    r = _cli("compose", str(small), str(rj), "-o", str(tmp_path / "o.png"))
    assert r.exit_code == 2 and "1487x2048" in r.text


def test_cli_reports_bad_configuration_without_a_traceback(tmp_path):
    page = tmp_path / "p.png"
    cv2.imwrite(str(page), _page())
    r = _cli("translate", str(page), "-o", str(tmp_path / "o.png"), "-d", "_rb_fake",
             "--ocr", "_rb_fake", "-t", "original", "-r", "bogus")
    assert r.exit_code == 2 and "unknown renderer 'bogus'" in r.text
    r = _cli("translate", str(page), "-o", str(tmp_path / "o.png"), "--opt", "detecter.conf=1")
    assert r.exit_code == 2 and "unknown stage" in r.text


class _LLMDown:
    vision = False

    def translate(self, req):
        from terjemahin_worker.translate.llm_common import LLMError

        raise LLMError("cannot connect to http://127.0.0.1:9/v1/chat/completions: [Errno 111] "
                       "Connection refused (is the server running and base_url right?)")


def test_cli_llm_connection_failure_is_one_line(tmp_path):
    if "_rb_llm_down" not in TRANSLATORS.methods:
        TRANSLATORS.register("_rb_llm_down", description="test fake")(lambda s, **_: _LLMDown())
    page = tmp_path / "p.png"
    cv2.imwrite(str(page), _page())
    r = _cli("translate", str(page), "-o", str(tmp_path / "o.png"), "-d", "_rb_fake",
             "--ocr", "_rb_fake", "-t", "_rb_llm_down", "--mask", "bbox", "-i", "fill",
             "-r", "basic")
    assert r.exit_code == 2 and "error: cannot connect to http://127.0.0.1:9" in r.output
    assert "Traceback" not in r.output


# --- consumer -------------------------------------------------------------------------------

def test_consumer_reports_bad_messages_and_creates_result_dirs(tmp_path, pipe):
    from terjemahin_worker.consumer import bad_message_status, process
    from terjemahin_worker.models import JobMessage

    st = bad_message_status(json.dumps({"job_id": "j1", "page_id": "p", "context": 123}).encode(),
                            ValueError("context Input should be an object"))
    assert st and st["job_id"] == "j1" and st["status"] == "failed" and "context" in st["error"]
    assert bad_message_status(b"not json", ValueError("x")) is None

    settings = Settings(storage_dir=tmp_path)
    cv2.imwrite(str(tmp_path / "in.png"), _page())
    job = JobMessage(job_id="x4", page_id="p", input_path="in.png", output_path="o4/out.png",
                     result_path="r4/result.json", pipeline=PipelineConfig(**CHEAP))
    st = process(job, pipe, settings)
    assert st["status"] == "done", st
    assert (tmp_path / "r4" / "result.json").exists()
    (tmp_path / "txt.png").write_text("hello")
    st = process(job.model_copy(update={"input_path": "txt.png"}), pipe, settings)
    assert st["status"] == "failed" and "not a readable image" in st["error"]


# --- web ------------------------------------------------------------------------------------

def _client(tmp_path, **kw):
    from fastapi.testclient import TestClient

    from terjemahin_worker.web.app import create_app

    return TestClient(create_app(Settings(storage_dir=tmp_path, **kw)))


def _post(c, data: bytes, pipeline=None):
    return c.post("/api/translate", files={"file": ("p.png", data, "image/png")},
                  data={"pipeline": json.dumps(pipeline or CHEAP)})


def test_web_upload_checks(tmp_path):
    with _client(tmp_path, max_pixels=10_000) as c:
        assert _post(c, b"").status_code == 400  # was a 500 from cv2.imdecode
        big = cv2.imencode(".png", np.full((200, 200, 3), 255, np.uint8))[1].tobytes()
        r = _post(c, big)
        assert r.status_code == 413 and "limit" in r.json()["detail"]
        r = _post(c, cv2.imencode(".png", np.full((50, 50, 3), 255, np.uint8))[1].tobytes(),
                  {**CHEAP, "target_lang": "klingon"})
        assert r.status_code == 400 and "klingon" in r.json()["detail"]
    with _client(tmp_path) as c:
        rgba = np.zeros((160, 380, 4), np.uint8)  # fully transparent: white paper
        r = _post(c, cv2.imencode(".png", rgba)[1].tobytes())
        assert r.status_code == 200, r.text
        img = cv2.imread(str(tmp_path / "web" / r.json()["id"] / "input.png"))
        assert img.min() == 255


def test_web_job_interrupted_by_a_restart_is_failed(tmp_path):
    from terjemahin_worker.web.app import Job, JobStore

    store = JobStore(tmp_path / "web")
    store.save(Job(id="0123456789abcdef", status="running", stage="detector"))
    fresh = JobStore(tmp_path / "web")  # a new process
    j = fresh.get("0123456789abcdef")
    assert j.status == "failed" and "interrupted" in (j.error or "")
    assert JobStore(tmp_path / "web").get("0123456789abcdef").status == "failed"


def test_web_prunes_old_jobs(tmp_path):
    from terjemahin_worker.web.app import Job, JobStore

    store = JobStore(tmp_path / "web")
    ids = [f"{i:016x}" for i in range(5)]
    for i in ids:
        store.save(Job(id=i, status="done"))
        time.sleep(0.01)
    store.save(Job(id=ids[0], status="running"))  # running: kept although oldest... now newest
    removed = store.prune(3)
    assert sorted(removed) == ids[1:3]
    assert sorted(d.name for d in (tmp_path / "web").iterdir()) == sorted([ids[0], *ids[3:]])


def test_web_erase_key_ignores_target_only_bubbles():
    from terjemahin_worker.web.app import erase_key

    cfg = PipelineConfig(mask_refiner="default", inpainter="fill")
    r = TextRegion(id=0, bbox=BBox(x1=0, y1=0, x2=10, y2=10))
    base = [Bubble(region=r, source_text="a", target_text="b")]
    note = Bubble(region=TextRegion(id=1, bbox=BBox(x1=20, y1=0, x2=40, y2=10)), target_text="n")
    assert erase_key(base + [note], cfg) == erase_key(base, cfg)
