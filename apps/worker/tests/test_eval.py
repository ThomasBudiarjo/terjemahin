import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from terjemahin_worker.config import Settings
from terjemahin_worker.detect.base import Detection
from terjemahin_worker.eval import bench as B
from terjemahin_worker.eval import metrics as M
from terjemahin_worker.models import BBox, Bubble, PipelineConfig, TextRegion
from terjemahin_worker.ocr.base import OcrResult
from terjemahin_worker.registry import DETECTORS, INPAINTERS, OCRS, TRANSLATORS
from terjemahin_worker.translate.base import TranslationItem

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "eval" / "fixtures" / "page01.png"
GT = ROOT / "eval" / "fixtures" / "page01.gt.json"
WEIGHTS = ROOT / "weights"


# -- text metrics -------------------------------------------------------------------------


def test_hungarian_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n, m = (int(v) for v in rng.integers(1, 6, 2))
        c = rng.random((n, m))
        pairs = M.hungarian(c)
        assert len(pairs) == min(n, m)
        assert len({i for i, _ in pairs}) == len({j for _, j in pairs}) == len(pairs)
        if n <= m:
            best = min(sum(c[i, q[i]] for i in range(n)) for q in itertools.permutations(range(m), n))
        else:
            best = min(sum(c[q[j], j] for j in range(m)) for q in itertools.permutations(range(n), m))
        assert sum(c[i, j] for i, j in pairs) == pytest.approx(best)
    assert M.hungarian(np.zeros((0, 3))) == []


def test_chrf_matches_sacrebleu():
    # reference values computed with sacrebleu 2.4.3 CHRF() / CHRF(word_order=2)
    hyps = ["Hei, hei, apa yang kau pikirkan tentang Sakamoto?", "Eh, itu saja?", "Wah, baiklah!", ""]
    refs = ["Hei, hei, Mei, menurutmu Sakamoto itu gimana?", "Eh, cuma itu?", "Wah, bagus dong!", "Iya, iya"]
    assert M.chrf(hyps, refs) == pytest.approx(30.703794690431234, abs=1e-9)
    assert M.chrf(hyps, refs, word_order=2) == pytest.approx(33.26635443950463, abs=1e-9)
    assert M.chrf(["the cat sat on the mat"], ["the cat is on the mat"]) == pytest.approx(64.5779420625287)
    assert M.chrf(["sama"], ["sama"]) == pytest.approx(100.0)
    assert M.chrf(["abc"], ["xyz"]) == 0.0


def test_cer_normalisation_and_language():
    a, b = "うーんそうだなぁ・・優しいよね", "うーんそうだなぁ…優しいよね"
    assert M.normalize_ja(a) == M.normalize_ja(b)
    assert M.cer(M.normalize_ja("坂本くん?"), M.normalize_ja("坂本くん？")) == 0.0
    assert M.cer("abcd", "abce") == 0.25
    assert M.cer("", "") == 0.0 and M.cer("x", "") == 1.0
    assert M.is_japanese("坂本くん？") and not M.is_japanese("Sakamoto-kun?")
    assert M.count_chars("A-aku... nggak!") == 9


def test_match_detections_classifies_everything():
    gt = ["ねえねえ、めいは坂本のことどう思う？", "坂本くん？", "えー、それだけー？", "うんうん"]
    det = [
        "えーそれだけ？",  # gt 2, OCR noise
        "坂本くん？",  # gt 1
        "ねえねえ、めいは坂本の",  # first half of gt 0 (split bubble)
        "ことどう思う？",  # second half: fragment
        "ごろん",  # SFX
        "全然関係ない文章です",  # false positive
    ]
    s = M.match_detections(det, gt, sfx=["ごろん"])
    by_gt = {m.gt: m.det for m in s.matches}
    assert by_gt == {0: 2, 1: 1, 2: 0}
    assert s.missed == [3]
    assert s.sfx == [4] and s.fragments == [3] and s.false_positives == [5]
    assert s.mean_cer is not None and 0 < s.mean_cer < 0.5


# -- image metrics -----------------------------------------------------------------------


def _balloon_page() -> tuple[np.ndarray, BBox]:
    page = np.full((400, 400, 3), 120, np.uint8)
    cv2.ellipse(page, (200, 200), (120, 90), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (200, 200), (120, 90), 0, 0, 360, (0, 0, 0), 3)
    return page, BBox(x1=170, y1=150, x2=230, y2=250)


def _bubble(b: BBox, target: str = "Halo dunia") -> Bubble:
    return Bubble(region=TextRegion(id=0, bbox=b), source_text="こんにちは", target_text=target)


def test_outside_mask_change():
    a = np.full((50, 50, 3), 200, np.uint8)
    mask = np.zeros((50, 50), np.uint8)
    mask[10:20, 10:20] = 255
    b = a.copy()
    b[10:20, 10:20] = 0
    assert M.outside_mask_change(a, b, mask) == {"mad": 0.0, "max": 0, "changed_frac": 0.0, "far_changed": 0}
    b[21, 15] = 190  # feather ring: changed but near
    b[45, 45] = 0  # far away: a real violation
    r = M.outside_mask_change(a, b, mask)
    assert r["max"] == 200 and r["far_changed"] == 1 and r["changed_frac"] > 0


def test_dark_residue_sees_ghost_glyphs():
    page, box = _balloon_page()
    orig = page.copy()
    cv2.putText(orig, "AB", (172, 215), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)
    mask = np.zeros(page.shape[:2], np.uint8)
    mask[box.y1 : box.y2, box.x1 : box.x2] = 255
    clean = M.dark_residue(orig, page, mask, [box])
    assert clean["pixels"] > 0 and clean["inpainted"] == 0.0 and clean["original"] > 0.05
    assert clean["ghost_max"] == 0.0
    dirty = page.copy()
    dirty[190:200, 180:220] = 0
    assert M.dark_residue(orig, dirty, mask, [box])["inpainted"] > 0.05
    faint = page.copy()
    faint[190:200, 180:220] = 190  # light-grey ghost: not dark, but still readable
    r = M.dark_residue(orig, faint, mask, [box])
    assert r["inpainted"] == 0.0 and r["ghost_max"] > 0.05


def test_render_check_flags_ink_outside_balloon():
    page, box = _balloon_page()
    b = _bubble(box)
    good = page.copy()
    cv2.putText(good, "Halo", (150, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    [r], overlay = M.render_check(page, good, page, [b])
    assert r.closed and r.ink_px > 100 and r.outside_px == 0 and r.bbox_inside and not r.lost
    assert r.font_source == "estimate" and r.font_px and r.font_px > 10
    assert overlay.shape == page.shape

    bad = good.copy()
    cv2.putText(bad, "spill", (240, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    [r], _ = M.render_check(page, bad, page, [b])
    assert r.outside_px > 50 and r.inside_frac < 0.9 and not r.bbox_inside

    [r], _ = M.render_check(page, page.copy(), page, [b])
    assert r.ink_px == 0 and r.lost


# -- harness end to end with fake stages --------------------------------------------------


class _FakeDetector:
    def detect(self, img):
        return Detection([TextRegion(id=0, bbox=BBox(x1=170, y1=150, x2=230, y2=250))])


class _FakeOcr:
    def recognize(self, img, regions):
        return [OcrResult(text="坂本くん？") for _ in regions]


class _FakeTranslator:
    vision = False

    def translate(self, req):
        # a cleaned-up source (like `chain`) must not be mistaken for what OCR read
        return [TranslationItem(source="坂本", target="Sakamoto-kun?") for _ in req.sources]


class _WhiteFill:
    def inpaint(self, img, mask):
        out = img.copy()
        out[mask > 0] = 255
        return out


@pytest.fixture
def fakes():
    regs = [
        (DETECTORS, "_t_det", _FakeDetector),
        (OCRS, "_t_ocr", _FakeOcr),
        (TRANSLATORS, "_t_tr", _FakeTranslator),
        (INPAINTERS, "_t_inp", _WhiteFill),
    ]
    for reg, name, cls in regs:
        reg.register(name)(lambda settings, _c=cls, **_: _c())
    yield PipelineConfig(
        detector="_t_det", ocr="_t_ocr", translator="_t_tr", mask_refiner="bbox",
        inpainter="_t_inp", renderer="default",
    )
    for reg, name, _ in regs:
        reg.methods.pop(name)


def test_evaluate_scores_every_stage_and_memoises(tmp_path, fakes):
    page = _balloon_page()[0]
    cv2.putText(page, "X", (185, 215), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
    gt = B.GroundTruth(["坂本くん？", "うんうん"], ["Sakamoto-kun?", "Iya, iya"])
    pipe = B.BenchPipeline(Settings())
    m = B.evaluate(pipe, page, gt, fakes, tmp_path / "a", judge=fakes)
    assert m["detection"]["matched"] == 1 and m["detection"]["missed_gt"] == [1]
    assert m["ocr"]["mean_cer"] == 0.0
    assert 0 < m["translation"]["chrf"] < 100 and m["translation"]["japanese"] == 0
    assert m["inpaint"]["outside_mad"] == 0.0 and m["inpaint"]["outside_far_changed"] == 0
    # the fake judge always "reads" text in its box, so residue is reported, not hidden
    assert m["inpaint"]["residual_regions"] == 1
    r = m["render"]
    assert r["bubbles"] == 1 and r["all_inside"] and r["lost"] == 0
    assert r["chars_rendered"] == r["chars_target"] == 11 and r["font_source"] == "layout"
    assert m["checks"]["ocr"] and m["checks"]["render"] and not m["checks"]["detect"]
    for f in ("final", "inpainted", "mask", "detect", "render_check"):
        assert (tmp_path / "a" / f"{f}.png").exists()
    saved = json.loads((tmp_path / "a" / "metrics.json").read_text())["bubbles"][0]
    assert saved["gt"] == 0 and saved["source"] == "坂本くん？" and saved["translator_source"] == "坂本"

    m2 = B.evaluate(pipe, page, gt, fakes.model_copy(update={"renderer": "basic"}), None, judge=fakes)
    assert set(m2["cached_stages"]) >= {"detect", "ocr", "translate", "inpaint"}
    assert m2["latency_s"]["detect"] == m["latency_s"]["detect"]  # the real call's time
    assert m2["render"]["font_source"] == "estimate"
    warm = B.warm_latency(pipe, page, fakes)
    assert set(warm) == {"detect", "ocr", "translate", "mask", "inpaint", "render"}
    assert pipe.use_memo


def test_matrix_report_and_contact_sheet(tmp_path, fakes):
    page = _balloon_page()[0]
    gt = B.GroundTruth(["坂本くん？"], ["Sakamoto-kun?"])
    pipe = B.BenchPipeline(Settings())
    report = B.run_matrix(pipe, page, gt, tmp_path, fakes, stages=["renderer"])
    labels = {r["label"]: r for r in report["runs"]}
    assert labels["baseline"]["status"] == "ok"
    assert labels["renderer-basic"]["status"] == "ok"
    assert not any(lab.startswith("ocr-") for lab in labels)
    ocr = dict(B.available_methods(pipe, pipe.resolve(fakes), False, False)["ocr"])
    assert "vision" in ocr["none"]
    md = B.markdown(report)
    assert "| baseline | ok |" in md and "renderer-basic" in md
    sheet = B.contact_sheet(report["runs"], tmp_path, Settings().font_path)
    assert cv2.imread(str(sheet)) is not None


def test_available_methods_skips_unrunnable():
    pipe = B.BenchPipeline(Settings(openai_api_key=""))
    methods = B.available_methods(pipe, pipe.resolve(PipelineConfig(translator="original")), False, False)
    tr = dict(methods["translator"])
    assert tr["openai"] and "OPENAI_API_KEY" in tr["openai"]
    assert tr["original"] is None
    if "nllb_big" in tr:
        assert "download" in tr["nllb_big"]


# -- real weights --------------------------------------------------------------------------

REAL = [WEIGHTS / "koharu-comictextdetector.onnx", WEIGHTS / "koharu-manga-ocr.onnx"]


@pytest.mark.skipif(not all(p.exists() for p in REAL + [PAGE, GT]), reason="weights or fixture missing")
def test_bench_on_page01_real_models(tmp_path):
    """ctd + manga_ocr on the real page; translation passthrough and telea keep it fast."""
    cfg = PipelineConfig(detector="ctd", ocr="manga_ocr", translator="original", inpainter="telea")
    pipe = B.BenchPipeline(Settings())
    page = cv2.imread(str(PAGE))
    m = B.evaluate(pipe, page, B.GroundTruth.load(GT), cfg, tmp_path, judge=cfg)
    d = m["detection"]
    assert d["matched"] >= 12 and d["false_positives"] == 0
    assert m["ocr"]["mean_cer"] < 0.1
    assert m["translation"]["japanese"] == m["translation"]["outputs"]  # passthrough is caught
    assert m["inpaint"]["outside_far_changed"] == 0 and m["inpaint"]["outside_mad"] < 0.5
    assert m["render"]["bubbles"] >= 12 and m["render"]["lost"] == 0
