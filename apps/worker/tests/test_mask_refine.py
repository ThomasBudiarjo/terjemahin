from pathlib import Path

import cv2
import numpy as np
import pytest

from terjemahin_worker.config import Settings
from terjemahin_worker.mask import MASK_REFINERS
from terjemahin_worker.mask.refine import (
    DilateMask,
    GlyphMask,
    crossing_lines,
    find_paper,
    font_size,
    make_crop,
)
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
    """Real page: every glyph pixel of the 13 balloons erased, outlines kept, with the detector
    mask, without it, and with bare boxes (no lines, no font size: ctd_blk, yolo, hand-drawn)."""
    from terjemahin_worker.detect.ctd_full import CtdFullDetector
    from terjemahin_worker.mask.refine import blend_ink, make_crop, ring_median, structure

    img = cv2.imread(str(PAGE))
    det = CtdFullDetector(WEIGHTS).detect(img)
    regions = [r for r in det.regions if r.bbox.y2 < 2020]  # skip the corner SFX over art
    assert len(regions) >= 13
    boxes = [r.model_copy(update={"lines": [], "font_size": None}) for r in regions]
    for fs, r in zip([font_size(b, img) for b in boxes], regions, strict=True):
        assert 0.7 * r.font_size <= fs <= 1.3 * r.font_size  # measured from ink, not the box
    H, W = img.shape[:2]
    loose = [  # hand-drawn boxes 20 px too big: they cross balloon outlines and art
        b.model_copy(update={"bbox": BBox(x1=max(0, b.bbox.x1 - 20), y1=max(0, b.bbox.y1 - 20),
                                          x2=min(W, b.bbox.x2 + 20), y2=min(H, b.bbox.y2 + 20))})
        for b in boxes
    ]  # fmt: skip
    sizes = {}
    for name, regs, raw in (
        ("seg", regions, det.mask),
        ("pixels", regions, None),
        ("box", boxes, None),
        ("loose", loose, None),
    ):
        for refiner in (GlyphMask(), DilateMask()):
            m = refiner.refine(img, regs, raw) > 0
            ink_total = missed = outline_hit = 0
            for r in regions:
                c = make_crop(r, img, 0.5)
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
            tag = (name, type(refiner).__name__, missed, outline_hit)
            assert ink_total > 40000
            assert missed / ink_total < 0.002, tag
            if isinstance(refiner, GlyphMask):
                assert outline_hit < 300, tag
            else:
                assert outline_hit < 3000, tag
            assert m.mean() < 0.08, tag  # glyphs only: no page-sized blobs
            sizes[name, type(refiner).__name__] = m.sum()
    for cls in ("GlyphMask", "DilateMask"):  # a loose box must not turn into an erased box
        assert sizes["loose", cls] < 1.1 * sizes["box", cls], sizes


def multi_column_page():
    """Big white balloon with four columns of text and a dark art blob inside its bbox corner."""
    H, W = 420, 420
    img = np.full((H, W, 3), (120, 110, 115), np.uint8)
    cv2.ellipse(img, (210, 210), (180, 190), 0, 0, 360, (250, 250, 250), -1)
    cv2.ellipse(img, (210, 210), (180, 190), 0, 0, 360, (0, 0, 0), 4, cv2.LINE_AA)
    glyphs = np.zeros((H, W), np.uint8)
    for i, word in enumerate(("AB", "CD", "EF", "GH")):
        for j, ch in enumerate(word):
            org = (110 + 55 * i, 150 + 60 * j)
            cv2.putText(glyphs, ch, org, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 3, cv2.LINE_AA)
    img = blend(img, glyphs, (10, 10, 10))
    ys, xs = np.nonzero(glyphs)
    box = BBox(
        x1=int(xs.min()) - 4, y1=int(ys.min()) - 4, x2=int(xs.max()) + 5, y2=int(ys.max()) + 5
    )
    return img, TextRegion(id=0, bbox=box), glyphs.astype(np.float32) / 255.0


@pytest.mark.parametrize("cls", [GlyphMask, DilateMask])
def test_box_only_region_measures_glyph_size(cls):
    img, region, glyphs = multi_column_page()
    assert 25 <= font_size(region, img) <= 50  # not the 150 px short side of the box
    m = cls().refine(img, [region], None)
    assert covered(m, glyphs, 0.05 if cls is GlyphMask else 0.3) == 1.0
    b = region.bbox
    assert (m[b.y1 : b.y2, b.x1 : b.x2] > 0).mean() < 0.6
    grown = cv2.dilate((glyphs > 0).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    assert not (m.astype(bool) & ~grown).any()  # nothing far from a glyph


def test_weak_seg_bleed_does_not_drop_glyph():
    small, region, glyphs, _ = balloon_page()
    img = np.full((700, 600, 3), (150, 140, 145), np.uint8)
    oy, ox = 300, 250
    img[oy : oy + 260, ox : ox + 300] = small
    g = np.zeros((700, 600), np.float32)
    g[oy : oy + 260, ox : ox + 300] = glyphs
    seg = fake_seg(g, spurious=False)
    b = region.bbox
    region = region.model_copy(
        update={
            "bbox": BBox(x1=b.x1 + ox, y1=b.y1 + oy, x2=b.x2 + ox, y2=b.y2 + oy),
            "lines": [[(x + ox, y + oy) for x, y in region.lines[0]]],
        }
    )
    cy = (region.bbox.y1 + region.bbox.y2) // 2
    cv2.line(seg, (region.bbox.x1 + 10, cy), (0, cy), 110, 3)  # bleed along a speed line
    m = GlyphMask().refine(img, [region], seg)
    assert covered(m, g, 0.05) == 1.0
    assert not m[:, : ox - 20].any()


@pytest.mark.parametrize("cls", [GlyphMask, DilateMask])
def test_malformed_boxes_are_tolerated(cls):
    img, region, _, _ = balloon_page()
    b = region.bbox
    flipped = region.model_copy(update={"bbox": BBox(x1=b.x2, y1=b.y2, x2=b.x1, y2=b.y1)})
    assert np.array_equal(cls().refine(img, [flipped], None), cls().refine(img, [region], None))
    for bad in (BBox(x1=400, y1=400, x2=500, y2=500), BBox(x1=-50, y1=-50, x2=-10, y2=-10),
                BBox(x1=50, y1=50, x2=50, y2=50)):  # fmt: skip
        m = cls().refine(img, [TextRegion(id=1, bbox=bad)], None)
        assert m.shape == img.shape[:2] and not m.any()


@pytest.mark.parametrize("cls", [GlyphMask, DilateMask])
def test_raw_mask_dtypes(cls):
    img, region, glyphs, _ = balloon_page()
    u8 = fake_seg(glyphs)
    ref = cls().refine(img, [region], u8)
    assert covered(ref, glyphs, 0.3) > 0.99
    for raw in (u8.astype(np.float32) / 255.0, u8 > 127, u8.astype(np.uint16) * 257):
        m = cls().refine(img, [region], raw)
        assert covered(m, glyphs, 0.3) > 0.99, raw.dtype


def test_unknown_option_warns(caplog):
    with caplog.at_level("WARNING"):
        MASK_REFINERS.build("default", Settings(), dilatation="0.3")
        MASK_REFINERS.build("dilate", Settings(), flat_spread="30")
    assert len(caplog.records) == 1 and "dilatation" in caplog.records[0].getMessage()


def loosen(region: TextRegion, pad: int) -> TextRegion:
    b = region.bbox
    box = BBox(x1=b.x1 - pad, y1=b.y1 - pad, x2=b.x2 + pad, y2=b.y2 + pad)
    return region.model_copy(update={"bbox": box, "lines": [], "font_size": None})


@pytest.mark.parametrize("cls", [GlyphMask, DilateMask])
@pytest.mark.parametrize("pad", [12, 25, 45])
def test_loose_box_over_outline_keeps_outline_and_art(cls, pad):
    """A hand-drawn box that crosses the balloon outline onto the art: the border rings are
    mixed, so the paper comes from the balloon interior; no box-shaped erase."""
    img, region, glyphs, outline = balloon_page()
    m = cls().refine(img, [loosen(region, pad)], None)
    assert covered(m, glyphs, 0.05 if cls is GlyphMask else 0.3) == 1.0
    assert (m[outline] > 0).mean() < (0.01 if cls is GlyphMask else 0.05)
    inside = np.zeros(m.shape, np.uint8)
    cv2.ellipse(inside, (150, 130), (110, 80), 0, 0, 360, 255, -1)
    assert not (m > 0)[inside == 0].any()  # nothing on the art outside the balloon
    b = region.bbox
    assert (m[b.y1 : b.y2, b.x1 : b.x2] > 0).mean() < 0.9


@pytest.mark.parametrize("cls", [GlyphMask, DilateMask])
def test_loose_box_mostly_background_no_blob(cls):
    """Narrow balloon in a box that holds more grey background than balloon: the background is
    the box's dominant colour, but the balloon (in the box middle) is the paper."""
    H, W = 300, 200
    img = np.full((H, W, 3), (160, 150, 155), np.uint8)
    cv2.ellipse(img, (100, 150), (26, 90), 0, 0, 360, (252, 252, 252), -1)
    cv2.ellipse(img, (100, 150), (26, 90), 0, 0, 360, (0, 0, 0), 4, cv2.LINE_AA)
    glyphs = np.zeros((H, W), np.uint8)
    for j, ch in enumerate("ABC"):
        cv2.putText(glyphs, ch, (88, 115 + 40 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2,
                    cv2.LINE_AA)  # fmt: skip
    img = blend(img, glyphs, (10, 10, 10))
    ys, xs = np.nonzero(glyphs)
    tight = TextRegion(id=0, bbox=BBox(x1=int(xs.min()), y1=int(ys.min()),
                                       x2=int(xs.max()) + 1, y2=int(ys.max()) + 1))  # fmt: skip
    m = cls().refine(img, [loosen(tight, 45)], None) > 0
    g = glyphs.astype(np.float32) / 255.0
    assert covered(m.astype(np.uint8) * 255, g, 0.3) == 1.0
    grown = cv2.dilate((glyphs > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    assert (m & ~grown).sum() < 100  # glyph-shaped, not the balloon interior


FONT = ROOT / "fonts" / "ComicShanns2.ttf"


def pil_page(outline=None, tone=False):
    """400x300 white page, "Hello!" at 40 px, optional balloon outline or screentone."""
    from PIL import Image, ImageDraw, ImageFont

    a = np.full((300, 400, 3), 255, np.uint8)
    if tone:
        for y in range(0, 300, 6):
            for x in range(0, 400, 6):
                cv2.circle(a, (x + (y // 6) % 2 * 3, y), 2, (60, 60, 60), -1)
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    if outline:
        d.ellipse(outline, fill=(255, 255, 255), outline=(0, 0, 0), width=5)
    font = ImageFont.truetype(str(FONT), 40)
    d.text((100, 120), "Hello!", font=font, fill=(0, 0, 0))
    return np.array(im)[..., ::-1].copy(), d.textbbox((100, 120), "Hello!", font=font)


@pytest.mark.skipif(not FONT.exists(), reason="font missing")
@pytest.mark.parametrize("cls", [GlyphMask, DilateMask])
@pytest.mark.parametrize("pad", [0, 10, 20])
def test_screentone_falls_back_to_box(cls, pad):
    """Text on screentone passes the ring test (the gaps between dots are paper) but has no
    solid paper: erase the whole box rather than a mesh of dots."""
    img, (x1, y1, x2, y2) = pil_page(tone=True)
    region = TextRegion(id=0, bbox=BBox(x1=x1 - pad, y1=y1 - pad, x2=x2 + pad, y2=y2 + pad))
    m = cls().refine(img, [region], None)
    assert (m[y1 - pad : y2 + pad, x1 - pad : x2 + pad] > 0).all()


@pytest.mark.skipif(not FONT.exists(), reason="font missing")
def test_outline_dilation_does_not_hug_outline():
    """A tight balloon whose outline joins a glyph: the outline is cut before dilating, so no
    mask ring is left along it, outside the balloon."""
    o = (92, 112, 290, 172)
    img, (x1, y1, x2, y2) = pil_page(outline=o)
    region = TextRegion(id=0, bbox=BBox(x1=x1 - 4, y1=y1 - 4, x2=x2 + 4, y2=y2 + 4))
    m = GlyphMask().refine(img, [region], None) > 0
    inside = np.zeros(m.shape, np.uint8)
    c, ax = (
        ((o[0] + o[2]) // 2, (o[1] + o[3]) // 2),
        ((o[2] - o[0]) // 2 - 3, (o[3] - o[1]) // 2 - 3),
    )
    cv2.ellipse(inside, c, ax, 0, 0, 360, 1, -1)
    assert not (m & (inside == 0)).any()
    assert m[y1:y2, x1:x2].mean() > 0.25  # the glyphs clear of the outline are erased


@pytest.mark.parametrize("cls", [GlyphMask, DilateMask])
def test_box_partly_off_page(cls):
    img, region, glyphs, _ = balloon_page()
    shifted = np.zeros_like(img)
    dx = region.bbox.x1 - 2
    shifted[:, : img.shape[1] - dx] = img[:, dx:]
    g = np.zeros_like(glyphs)
    g[:, : img.shape[1] - dx] = glyphs[:, dx:]
    b = region.bbox
    r = TextRegion(id=0, bbox=BBox(x1=-30, y1=b.y1 - 5, x2=b.x2 - dx + 5, y2=b.y2 + 5))
    m = cls().refine(shifted, [r], None)
    assert covered(m, g, 0.3) > 0.99


def lettering_page(
    outline: int = 0, two_tone: bool = False, cross: bool = False, thin: bool = False
):
    """Text touching an art line that runs off the page window, on flat lavender paper; with an
    outline ring (black SFX lettering with a white edge), half the page a darker background,
    another art line crossing behind the text, or strokes as thin as the art lines.

    Returns image, box-only region, glyph alpha, ring alpha (glyph + outline) and the lines."""
    H, W = 240, 320
    img = np.full((H, W, 3), (172, 148, 158), np.uint8)
    if two_tone:
        img[:, 170:] = (63, 51, 51)
    lines = np.zeros((H, W), np.uint8)
    cv2.line(lines, (60, 0), (70, H), 255, 4, cv2.LINE_AA)
    if cross:
        cv2.line(lines, (0, 95), (W, 115), 255, 3, cv2.LINE_AA)
    glyphs = np.zeros((H, W), np.uint8)
    cv2.putText(glyphs, "AWB", (52, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 7, cv2.LINE_AA)
    if thin:  # OpenCV 5's putText ignores the thickness
        glyphs = cv2.erode(glyphs, np.ones((3, 3), np.uint8), iterations=2)
    img = blend(img, lines, (40, 30, 90))
    ring = glyphs
    if outline:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outline + 1, 2 * outline + 1))
        ring = cv2.dilate(glyphs, k)
        img = blend(img, ring, (255, 255, 255))
    img = blend(img, glyphs, (5, 5, 5))
    ys, xs = np.nonzero(ring)
    box = BBox(
        x1=int(xs.min()) - 16, y1=int(ys.min()) - 16, x2=int(xs.max()) + 17, y2=int(ys.max()) + 17
    )
    return img, TextRegion(id=0, bbox=box), glyphs, ring, lines


def far_from(alpha: np.ndarray, lines: np.ndarray, region: TextRegion, px: int = 24) -> np.ndarray:
    """Art line pixels more than `px` from the glyphs and outside the (slightly grown) box."""
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    out = (lines > 200) & ~(cv2.dilate((alpha > 0).astype(np.uint8), k) > 0)
    b = region.bbox.pad(4)
    out[max(0, b.y1) : b.y2, max(0, b.x1) : b.x2] = False
    return out


@pytest.mark.parametrize("cls", [GlyphMask, DilateMask])
@pytest.mark.parametrize(
    "kw",
    [
        {},  # plain glyph fused with the art line
        {"outline": 3},
        {"outline": 3, "two_tone": True},  # the paper detection can only take one background
        {"outline": 3, "two_tone": True, "cross": True},
        {"cross": True},
    ],
)
def test_box_only_glyphs_touching_art(cls, kw):
    """Without a detector mask, glyphs fused with an art line are split from it by stroke
    width, outlined lettering comes with its outline ring whatever background it is on, and
    the art away from the glyphs stays."""
    img, region, glyphs, ring, lines = lettering_page(**kw)
    m = cls().refine(img, [region], None) > 0
    assert m[glyphs > 30].all()
    assert m[ring > 30].all()
    far = far_from(ring, lines, region)
    if cls is GlyphMask:
        assert not (m & far).any()
    b = region.bbox
    assert m[b.y1 : b.y2, b.x1 : b.x2].mean() < 0.9  # glyph-shaped, not the box


def test_box_only_unsplittable_text_is_box_erased():
    """Strokes as thin as the art lines they are fused with cannot be split off: erase the box
    instead of leaving half erased glyphs."""
    img, region, glyphs, _, _ = lettering_page(cross=True, thin=True)
    for cls in (GlyphMask, DilateMask):
        m = cls().refine(img, [region], None) > 0
        assert m[glyphs > 30].all(), cls.__name__


def test_glyphs_option_off_keeps_first_pass_only():
    img, region, _, ring, _ = lettering_page(outline=3, two_tone=True)
    assert covered(GlyphMask(glyphs=False).refine(img, [region], None), ring / 255.0, 0.1) < 0.9
    s = Settings()
    assert not MASK_REFINERS.build("default", s, glyphs="false").glyphs
    assert not MASK_REFINERS.build("dilate", s, glyphs="0").glyphs


PAGE01_SFX = {"ごろん": (728, 1190, 934, 1300), "もじ…": (1226, 1872, 1330, 1974)}


def sfx_truth(img: np.ndarray, box: tuple[int, int, int, int]):
    """White outline, the black fill it encloses, and dark art / background away from both."""
    x1, y1, x2, y2 = box
    g = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    white = (g > 200).astype(np.uint8)
    reach = np.pad(white, 1)
    cv2.floodFill(reach, None, (0, 0), 2)
    fill = (reach[1:-1, 1:-1] == 0) & (g < 100)
    art = (g < 60) & ~(cv2.dilate(white, np.ones((21, 21), np.uint8)) > 0)
    return white > 0, fill, art


@pytest.mark.skipif(not PAGE.exists(), reason="page01 missing")
@pytest.mark.parametrize("cls", [GlyphMask, DilateMask])
@pytest.mark.parametrize("pad", [0, 12, 30])
def test_page01_outlined_sfx_without_detector_mask(cls, pad):
    """Black SFX with a white edge over art (ごろん crosses two backgrounds and art lines; も of
    もじ… touches the arm line): fill and outline erased from a bare box, art kept."""
    img = cv2.imread(str(PAGE))
    for name, (x1, y1, x2, y2) in PAGE01_SFX.items():
        r = TextRegion(id=0, bbox=BBox(x1=x1 - pad, y1=y1 - pad, x2=x2 + pad, y2=y2 + pad))
        m = cls().refine(img, [r], None)[y1:y2, x1:x2] > 0
        white, fill, art = sfx_truth(img, (x1, y1, x2, y2))
        assert white.sum() > 500 and fill.sum() > 400
        assert m[white].mean() > 0.995 and m[fill].mean() > 0.995, (name, pad)
        if cls is GlyphMask:
            assert m[art].mean() < 0.01, (name, pad)


def crossed_page(text: str, size: int, lines, line_bgr, pad: int):
    """Page-sized flat paper with text (ComicShanns, strokes about as thick as the art lines)
    and art lines running through it and well past the box, as on a real page: the line
    leaves the refiner's window. Returns image, box-only region, glyph alpha, line alpha."""
    from PIL import Image, ImageDraw, ImageFont

    H, W = 560, 800
    img = np.full((H, W, 3), (172, 148, 158), np.uint8)
    art = np.zeros((H, W), np.uint8)
    for x1, y1, x2, y2, t in lines:
        cv2.line(art, (x1, y1), (x2, y2), 255, t, cv2.LINE_AA)
    pil = Image.new("L", (W, H))
    ImageDraw.Draw(pil).text((300, 240), text, font=ImageFont.truetype(str(FONT), size), fill=255)
    glyphs = np.array(pil)
    img = blend(blend(img, art, line_bgr), glyphs, (5, 5, 5))
    ys, xs = np.nonzero(glyphs)
    box = BBox(
        x1=int(xs.min()) - pad,
        y1=int(ys.min()) - pad,
        x2=int(xs.max()) + 1 + pad,
        y2=int(ys.max()) + 1 + pad,
    )
    return img, TextRegion(id=0, bbox=box), glyphs, art


CROSSED = [  # text, px, lines (x1, y1, x2, y2, width), line colour: all left half erased before
    ("BAM", 69, [(0, 331, 800, 293, 3), (320, 0, 394, 560, 5)], (46, 75, 114)),
    ("Ok!", 59, [(0, 302, 800, 229, 4), (419, 0, 383, 560, 4)], (107, 61, 111)),
    ("Hello", 44, [(0, 259, 800, 308, 5), (342, 0, 384, 560, 4)], (81, 82, 104)),
    ("BAM", 48, [(0, 246, 800, 239, 2), (418, 0, 375, 560, 5)], (72, 116, 111)),
]


@pytest.mark.skipif(not FONT.exists(), reason="font missing")
@pytest.mark.parametrize("cls", [GlyphMask, DilateMask])
@pytest.mark.parametrize("case", range(len(CROSSED)))
@pytest.mark.parametrize("pad", [4, 24])
def test_box_only_glyphs_crossed_by_art_lines(cls, case, pad):
    """Without a detector mask, glyphs that art lines run through (strokes no thicker than
    the lines, so no stroke-width split) are erased on flat paper; the lines away from the
    glyphs stay."""
    img, region, glyphs, art = crossed_page(*CROSSED[case], pad=pad)
    m = cls().refine(img, [region], None) > 0
    assert m[glyphs > 30].mean() > 0.995
    far = far_from(glyphs, art, region)
    assert far.any() and m[far].mean() < 0.005


def test_crossing_lines_spare_straight_glyph_strokes():
    """Long straight glyph strokes end at the box; the art line goes on past it."""
    H, W = 500, 700
    img = np.full((H, W, 3), (172, 148, 158), np.uint8)
    art = np.zeros((H, W), np.uint8)
    cv2.line(art, (0, 240), (W, 262), 255, 4, cv2.LINE_AA)
    glyphs = np.zeros((H, W), np.uint8)
    cv2.putText(glyphs, "HIT", (260, 290), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 7, cv2.LINE_AA)
    img = blend(blend(img, art, (40, 30, 90)), glyphs, (5, 5, 5))
    ys, xs = np.nonzero(glyphs)
    b = BBox(
        x1=int(xs.min()) - 10, y1=int(ys.min()) - 10, x2=int(xs.max()) + 11, y2=int(ys.max()) + 11
    )
    c = make_crop(TextRegion(id=0, bbox=b), img, 0.5)
    crop = img[c.sl]
    _, dist, _, cn = find_paper(crop, c, 40.0)
    lines = crossing_lines(dist > 40, cn, cn.font)
    away = (glyphs[c.sl] > 128) & ~(cv2.dilate(art[c.sl], np.ones((9, 9), np.uint8)) > 0)
    assert lines[art[c.sl] > 128].mean() > 0.8
    assert not (lines & away).any()


@pytest.mark.parametrize("cls", [GlyphMask, DilateMask])
def test_box_only_keeps_neighbour_glyph_outside_box(cls):
    """A bare box's text zone reaches past it (a tight box may cut its glyphs), but ink
    lying wholly beyond the box, such as a neighbour balloon's dot, is not this box's text."""
    img = np.full((400, 600, 3), 250, np.uint8)
    cv2.putText(
        img, "Hello", (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (10, 10, 10), 4, cv2.LINE_AA
    )
    dot = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(dot, (260, 168), 6, 1, -1)
    img[dot > 0] = 10
    region = TextRegion(id=0, bbox=BBox(x1=195, y1=182, x2=355, y2=230))
    m = cls().refine(img, [region], None) > 0
    assert not m[dot > 0].any()
    text = (img[182:230, 195:355, 0] < 128).astype(np.float32)
    assert covered(m[182:230, 195:355].astype(np.uint8) * 255, text, 0.5) == 1.0
