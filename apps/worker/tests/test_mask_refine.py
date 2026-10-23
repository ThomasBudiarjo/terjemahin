from pathlib import Path

import cv2
import numpy as np
import pytest

from terjemahin_worker.config import Settings
from terjemahin_worker.mask import MASK_REFINERS
from terjemahin_worker.mask.refine import DilateMask, GlyphMask, font_size
from terjemahin_worker.models import BBox, TextRegion

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = Settings().weights_dir / "koharu-comictextdetector.onnx"
PAGE = ROOT / "eval" / "fixtures" / "page01.png"


def balloon_page(text_bgr=(20, 20, 20), paper=(250, 250, 250), dark_paper=False, gap=5):
    """Grey 'art' page with one outlined balloon and anti-aliased text close to its outline.

    Returns image, region, glyph coverage (float 0..1) and the outline's solid pixels.
    """
    H, W = 260, 300
    img = np.full((H, W, 3), (150, 140, 145), np.uint8)
    cv2.line(img, (0, 30), (W, 60), (60, 60, 60), 3)  # an art line crossing the page
    center, axes = (150, 130), (110, 80)
    fill = (25, 25, 25) if dark_paper else paper
    cv2.ellipse(img, center, axes, 0, 0, 360, fill, -1)
    outline = np.zeros((H, W), np.uint8)
    cv2.ellipse(outline, center, axes, 0, 0, 360, 255, 4, cv2.LINE_AA)
    color = (235, 235, 235) if dark_paper else (0, 0, 0)
    img = blend(img, outline, (0, 0, 0) if not dark_paper else (230, 230, 230))

    glyphs = np.zeros((H, W), np.uint8)
    x0 = 150 - 110 + 2 + gap + 18  # first glyph a few px inside the left outline
    cv2.putText(glyphs, "AWM", (x0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 4, cv2.LINE_AA)
    cv2.putText(glyphs, "o", (x0 + 150, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2, cv2.LINE_AA)
    img = blend(img, glyphs, color if dark_paper else text_bgr)

    ys, xs = np.nonzero(glyphs)
    bbox = BBox(x1=int(xs.min()), y1=int(ys.min()), x2=int(xs.max()) + 1, y2=int(ys.max()) + 1)
    q = [(bbox.x1, bbox.y1), (bbox.x2, bbox.y1), (bbox.x2, bbox.y2), (bbox.x1, bbox.y2)]
    region = TextRegion(id=0, bbox=bbox, lines=[q], direction="h", font_size=40)
    solid_outline = outline > 200
    return img, region, glyphs.astype(np.float32) / 255.0, solid_outline


def blend(img, alpha_u8, color):
    a = (alpha_u8.astype(np.float32) / 255.0)[..., None]
    out = img.astype(np.float32) * (1 - a) + np.array(color, np.float32) * a
    return out.round().astype(np.uint8)


def fake_seg(glyphs: np.ndarray, spurious: bool = True) -> np.ndarray:
    """Detector-like soft mask: glyph cores, a fuzzy rim, plus a false blob on the art."""
    seg = cv2.GaussianBlur((glyphs > 0.5).astype(np.float32), (5, 5), 1.5)
    seg = np.clip(seg * 1.6, 0, 1)
    if spurious:
        cv2.circle(seg, (270, 20), 8, 0.9, -1)
    return (seg * 255).astype(np.uint8)


def covered(mask, glyphs, level=0.1):
    sel = glyphs >= level
    return float((mask[sel] > 0).mean())


@pytest.mark.parametrize("with_seg", [True, False])
def test_default_covers_glyphs_and_spares_outline(with_seg):
    img, region, glyphs, outline = balloon_page()
    raw = fake_seg(glyphs) if with_seg else None
    m = GlyphMask().refine(img, [region], raw)
    assert m.dtype == np.uint8 and m.shape == img.shape[:2]
    assert set(np.unique(m)) <= {0, 255}
    assert covered(m, glyphs, 0.05) == 1.0  # anti-aliased fringe included
    assert (m[outline] > 0).mean() < 0.01
    assert not m[:60].any()  # art line and spurious seg blob far from the text stay
    # glyph-shaped, not the box: plenty of paper inside the box is left alone
    b = region.bbox
    assert (m[b.y1 : b.y2, b.x1 : b.x2] > 0).mean() < 0.9


def test_default_light_text_on_dark_paper():
    img, region, glyphs, outline = balloon_page(dark_paper=True)
    m = GlyphMask().refine(img, [region], None)
    assert covered(m, glyphs, 0.05) == 1.0
    assert (m[outline] > 0).mean() < 0.01


def test_seg_blob_overlapping_art_is_dropped():
    img, region, glyphs, _ = balloon_page()
    raw = fake_seg(glyphs, spurious=False)
    # a big false positive beside the text, reaching into the text zone but mostly outside it
    b = region.bbox
    cv2.rectangle(raw, (b.x2 + 2, b.y1 + 20), (299, b.y2), 230, -1)
    m = GlyphMask().refine(img, [region], raw)
    assert not m[:, b.x2 + 25 :].any()
    assert covered(m, glyphs, 0.05) == 1.0


def test_busy_background_falls_back_to_text_zone():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    region = TextRegion(id=0, bbox=BBox(x1=40, y1=30, x2=120, y2=90), font_size=30)
    m = GlyphMask().refine(img, [region], None)
    assert (m[30:90, 40:120] > 0).all()
    assert not m[:5].any() and not m[:, :5].any()


def test_no_regions_and_size_mismatch():
    img, region, glyphs, _ = balloon_page()
    assert not GlyphMask().refine(img, [], None).any()
    raw = cv2.resize(fake_seg(glyphs), (150, 130))  # detector mask at a lower resolution
    m = GlyphMask().refine(img, [region], raw)
    assert m.shape == img.shape[:2] and covered(m, glyphs, 0.3) > 0.99


def test_dilate_variant_and_options():
    img, region, glyphs, _ = balloon_page()
    raw = fake_seg(glyphs)
    small = DilateMask(dilation=0.05).refine(img, [region], raw)
    big = DilateMask(dilation=0.3).refine(img, [region], raw)
    assert covered(small, glyphs, 0.3) > 0.99
    assert (big > 0).sum() > (small > 0).sum()
    assert not small[:40, 250:].any()  # spurious blob outside the text zone ignored
    assert covered(DilateMask().refine(img, [region], None), glyphs, 0.3) > 0.99


def test_method_option():
    img, region, glyphs, _ = balloon_page()
    raw = fake_seg(glyphs)
    pixels = GlyphMask(method="pixels").refine(img, [region], raw)
    assert np.array_equal(pixels, GlyphMask().refine(img, [region], None))
    assert not GlyphMask(method="seg").refine(img, [region], None).any()
    assert covered(GlyphMask(method="seg").refine(img, [region], raw), glyphs, 0.05) == 1.0
    with pytest.raises(ValueError):
        GlyphMask(method="crf")


def test_registry_and_font_size():
    s = Settings()
    assert isinstance(MASK_REFINERS.build("default", s, dilation="0.2", protect="false"), GlyphMask)
    assert isinstance(MASK_REFINERS.build("dilate", s), DilateMask)
    r = TextRegion(
        id=0, bbox=BBox(x1=0, y1=0, x2=40, y2=200), lines=[[(0, 0), (40, 0), (40, 200), (0, 200)]]
    )
    assert font_size(r) == 40


def text_zone(r: TextRegion, c, reach: float) -> np.ndarray:
    zone = np.zeros(c.near.shape, np.uint8)
    cv2.fillPoly(zone, [np.asarray(q, np.int32) - (c.x1, c.y1) for q in r.lines], 255)
    k = 2 * round(c.font * reach) + 1
    return cv2.dilate(zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


@pytest.mark.skipif(not (WEIGHTS.exists() and PAGE.exists()), reason="ctd weights missing")
def test_page01_with_ctd():
    """Real page: every glyph pixel of the 13 balloons erased, outlines kept, both modes."""
    from terjemahin_worker.detect.ctd_full import CtdFullDetector
    from terjemahin_worker.mask.refine import blend_ink, make_crop, ring_median, structure

    img = cv2.imread(str(PAGE))
    det = CtdFullDetector(WEIGHTS).detect(img)
    regions = [r for r in det.regions if r.bbox.y2 < 2020]  # skip the corner SFX over art
    assert len(regions) >= 13
    H, W = img.shape[:2]
    for raw in (det.mask, None):
        m = GlyphMask().refine(img, regions, raw) > 0
        ink_total = missed = outline_hit = 0
        for r in regions:
            c = make_crop(r, W, H, 0.5)
            lines = text_zone(r, c, 0.25)
            crop = img[c.sl]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            dark = gray < 128
            bg = ring_median(crop, dark & c.near, 5, 15)
            ink = blend_ink(crop, bg, np.zeros(3), 0.25)
            st = structure(ink, c)
            glyph = dark & ~st & lines
            ink_total += glyph.sum()
            missed += (glyph & ~m[c.sl]).sum()
            outline_hit += (st & (gray < 80) & m[c.sl]).sum()
        assert ink_total > 40000
        assert missed / ink_total < 0.002, (raw is None, missed, ink_total)
        assert outline_hit < 300, (raw is None, outline_hit)
        assert m.mean() < 0.08  # glyphs only: no page-sized blobs
