import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from terjemahin_worker.config import Settings
from terjemahin_worker.models import BBox, Bubble, TextRegion
from terjemahin_worker.registry import RENDERERS, load_all
from terjemahin_worker.render.balloon import find_balloon, split_shared
from terjemahin_worker.render.hyphen import split_points, syllables
from terjemahin_worker.render.layout import FontMeasure, Shape, fit
from terjemahin_worker.render.vertical import NO_START, fit_vertical

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "fonts" / "ComicShanns2.ttf"
PAGE = ROOT / "eval" / "fixtures" / "page01.png"
GT = ROOT / "eval" / "fixtures" / "page01.gt.json"
CTD = ROOT / "weights" / "koharu-comictextdetector.onnx"


@pytest.fixture(scope="module")
def measure() -> FontMeasure:
    return FontMeasure(FONT if FONT.exists() else None)


def ellipse_mask(w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (w // 2 - 1, h // 2 - 1), 0, 0, 360, 255, -1)
    return m


def assert_inside(lay, mask: np.ndarray, m: FontMeasure) -> None:
    """Every line's ink box (all four corners) lies inside the mask."""
    top, bottom = m.ink(lay.size)
    h, w = mask.shape
    for ln in lay.lines:
        for x in (ln.x, ln.x + ln.width - 1):
            for y in (ln.y + top, ln.y + bottom - 1):
                xi, yi = round(x), round(y)
                assert 0 <= xi < w and 0 <= yi < h and mask[yi, xi], (ln.text, x, y)


# -- hyphenation --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "word,expected",
    [
        ("menyenangkan", ["me", "nye", "nang", "kan"]),
        ("instrumen", ["in", "stru", "men"]),
        ("perempuan", ["pe", "rem", "pu", "an"]),
        ("pantainya", ["pan", "tai", "nya"]),
        ("masyarakat", ["ma", "sya", "ra", "kat"]),
        ("mengkhawatirkan", ["meng", "kha", "wa", "tir", "kan"]),
    ],
)
def test_indonesian_syllables(word, expected):
    assert syllables(word) == expected


def test_split_points_keep_punctuation_and_min_parts():
    assert split_points("sendiri,") == ["sen", "diri,"]
    assert split_points("anak") == ["anak"]  # too short to split
    assert split_points("gimana...")[-1].endswith("...")
    for w in ["menyenangkan", "rambut,", "langsung", "Sakamoto", "perempuan?"]:
        parts = split_points(w)
        assert "".join(parts) == w
        assert all(len(p.strip(",.?!")) >= 3 for p in parts)


# -- layout engine ------------------------------------------------------------------------


def test_fit_rect_is_largest_size_and_inside(measure):
    shape = Shape.rect(200, 120)
    lay = fit("Aku tidak akan menyerah begitu saja", shape, measure, min_size=8, max_size=80)
    assert lay.fits and lay.lines
    assert_inside(lay, np.full((120, 200), 255, np.uint8), measure)
    bigger = fit(
        "Aku tidak akan menyerah begitu saja",
        shape,
        measure,
        min_size=lay.size + 1,
        max_size=lay.size + 1,
    )
    assert not bigger.fits or bigger.size == lay.size + 1 and not bigger.fits


def test_fit_follows_ellipse_not_bounding_box(measure):
    mask = ellipse_mask(220, 300)
    text = "Ah, terus kemarin itu, waktu aku ganti gaya rambut, dia langsung sadar..."
    lay = fit(text, Shape.from_mask(mask), measure, min_size=10, max_size=60)
    assert lay.fits
    assert_inside(lay, mask, measure)
    widths = [ln.width for ln in lay.lines]
    mid = len(widths) // 2
    # oval: the middle lines are wider than the first and last ones
    assert widths[mid] > widths[0] and widths[mid] > widths[-1]
    # and the block is centred vertically
    top, bottom = measure.ink(lay.size)
    y0, y1 = lay.lines[0].y + top, lay.lines[-1].y + bottom
    assert abs((y0 + y1) / 2 - 150) < lay.line_height


def test_no_text_lost_and_order_kept(measure):
    texts = [
        "Dia sopan pakai bahasa halus bahkan ke pelayan toko, keren gitu",
        "A-aku sih... nggak gimana-gimana...",
        "Eh?",
        "Supercalifragilisticexpialidocious " * 3,
    ]
    for t in texts:
        norm = measure.normalize(t)
        for shape in (Shape.from_mask(ellipse_mask(150, 260)), Shape.rect(60, 400)):
            lay = fit(norm, shape, measure, min_size=12, max_size=48)
            assert lay.text == norm, (lay.text, norm)


def test_hyphenates_only_when_it_pays(measure):
    narrow = Shape.from_mask(ellipse_mask(120, 400))
    lay = fit("Reaksinya selalu lucu", narrow, measure, min_size=8, max_size=40)
    assert lay.text == "Reaksinya selalu lucu"
    wide = fit("Reaksinya selalu lucu", Shape.rect(600, 200), measure, min_size=8, max_size=40)
    assert not wide.hyphenated
    off = fit(
        "menyenangkan sekali",
        Shape.rect(70, 400),
        measure,
        min_size=8,
        max_size=40,
        hyphenate=False,
    )
    assert not off.hyphenated and off.text == "menyenangkan sekali"


def test_names_and_compounds_are_not_soft_hyphenated(measure):
    lay = fit("Hei Sakamoto-kun", Shape.rect(120, 300), measure, min_size=8, max_size=60)
    assert "Saka-" not in [ln.text for ln in lay.lines]


def test_avoids_orphan_last_word(measure):
    text = "Dia sopan pakai bahasa halus bahkan ke pelayan toko keren gitu"
    lay = fit(text, Shape.rect(260, 400), measure, min_size=8, max_size=60)
    assert len(lay.lines) > 1
    assert " " in lay.lines[-1].text or lay.lines[-1].width >= 0.4 * lay.lines[-2].width


def test_overflow_keeps_every_word(measure):
    text = "Kalimat yang sangat panjang sekali tidak mungkin muat di balon kecil ini"
    lay = fit(text, Shape.rect(40, 30), measure, min_size=14, max_size=20)
    assert not lay.fits
    assert lay.size == 14
    assert lay.text.replace(" ", "") == text.replace(" ", "")


def test_alignment_left_shares_an_edge(measure):
    lay = fit(
        "satu dua tiga empat lima enam",
        Shape.rect(200, 200),
        measure,
        min_size=8,
        max_size=40,
        align="left",
    )
    assert len({round(ln.x, 1) for ln in lay.lines}) == 1


def test_shape_centre_ignores_crop_edge_and_lobes():
    m = np.zeros((100, 200), np.uint8)
    m[:, 100:] = 255  # interior that touches the crop's right edge (balloon cut by the page)
    s = Shape.from_mask(m)
    assert 120 < s.axis() < 180
    lobes = np.zeros((200, 300), np.uint8)
    cv2.circle(lobes, (80, 120), 75, 255, -1)
    cv2.circle(lobes, (230, 60), 50, 255, -1)
    cv2.rectangle(lobes, (80, 50), (230, 110), 255, -1)
    s = Shape.from_mask(lobes)
    assert abs(s.axis() - 80) < 30  # deepest lobe wins


def test_normalize_text(measure):
    assert measure.normalize("ＡＢＣ！？  dan…") == "ABC!? dan…"
    assert "…" in measure.normalize("sih...")


# -- balloon extraction -------------------------------------------------------------------


def _scene():
    rng = np.random.default_rng(0)
    img = np.full((600, 800, 3), 230, np.uint8)
    img[:, 420:] = (rng.random((600, 380, 3)) * 90 + 60).astype(np.uint8)
    cv2.ellipse(img, (210, 170), (170, 120), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (210, 170), (170, 120), 0, 0, 360, (0, 0, 0), 4)
    cv2.ellipse(img, (210, 450), (150, 100), 0, 0, 360, (25, 25, 25), -1)
    cv2.ellipse(img, (210, 450), (150, 100), 0, 0, 360, (250, 250, 250), 4)
    # page-edge balloon: centre off the right edge
    cv2.ellipse(img, (800, 90), (130, 80), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (800, 90), (130, 80), 0, 0, 360, (0, 0, 0), 4)
    boxes = [
        BBox(x1=150, y1=110, x2=270, y2=230),
        BBox(x1=160, y1=400, x2=260, y2=500),
        BBox(x1=580, y1=250, x2=640, y2=520),
        BBox(x1=720, y1=50, x2=780, y2=130),
    ]
    return img, boxes


def test_find_balloon_closed_and_open():
    img, boxes = _scene()
    white = find_balloon(img, img, boxes[0])
    assert white.closed
    bb = white.bbox
    assert abs(bb.x1 - 44) < 8 and abs(bb.x2 - 377) < 8 and abs(bb.y1 - 53) < 8
    assert white.color[0] > 240
    dark = find_balloon(img, img, boxes[1])
    assert dark.closed and dark.color[0] < 60
    assert not find_balloon(img, img, boxes[2]).closed  # over noisy art
    assert find_balloon(img, img, boxes[3]).closed  # cut by the page edge is still a balloon


def test_split_shared_balloon():
    img = np.full((300, 400, 3), 200, np.uint8)
    cv2.ellipse(img, (200, 150), (180, 120), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (200, 150), (180, 120), 0, 0, 360, (0, 0, 0), 3)
    boxes = [BBox(x1=70, y1=110, x2=150, y2=190), BBox(x1=250, y1=110, x2=330, y2=190)]
    balloons = [find_balloon(img, img, b) for b in boxes]
    assert all(b.closed for b in balloons)
    split_shared(balloons, boxes)
    assert balloons[0].bbox.x2 <= 205 and balloons[1].bbox.x1 >= 195


# -- renderer -----------------------------------------------------------------------------


def _bubbles(boxes, texts, **kw):
    return [
        Bubble(
            region=TextRegion(id=i, bbox=b, direction="h", font_size=40, **kw),
            source_text="x",
            target_text=t,
        )
        for i, (b, t) in enumerate(zip(boxes, texts))
    ]


def test_renderers_registered():
    load_all()
    assert {"default", "vertical", "basic"} <= set(RENDERERS.names())
    with pytest.raises(ValueError):
        RENDERERS.build("default", Settings(), not_an_option=1)


def test_default_renderer_colours_and_containment():
    load_all()
    img, boxes = _scene()
    texts = ["Aku tidak akan menyerah!", "Malam ini gelap sekali.", "DUAR!! Rasakan ini!", "Eh?"]
    r = RENDERERS.build("default", Settings())
    out = r.render(img, _bubbles(boxes, texts), img)
    assert out.shape == img.shape
    changed = np.abs(out.astype(int) - img.astype(int)).sum(axis=2) > 60
    plans = r.plans
    assert [p.closed for p in plans] == [True, True, False, True]
    assert all(p.layout.fits for p in plans)
    for p, t in zip(plans, texts):
        assert p.layout.text == r.measure.normalize(t)
    white = find_balloon(img, img, boxes[0])
    ys, xs = white.page_slice()
    region = changed[ys, xs]
    inner = cv2.erode(white.mask, np.ones((7, 7), np.uint8))
    assert region.any() and not (region & (inner == 0) & (white.mask > 0)).any()
    # dark balloon -> light text, over art -> outlined
    fg, _, sw = r.colors(plans[1], plans[1].layout.size)
    assert fg == (255, 255, 255) and sw == 0
    _, _, sw_art = r.colors(plans[2], plans[2].layout.size)
    assert sw_art >= 2
    # OCR colours win when present
    custom = _bubbles(boxes[:1], texts[:1], fg_color=(200, 0, 0), bg_color=(255, 255, 255))
    out2 = r.render(img, custom, img)
    reds = (out2[..., 2] > 150) & (out2[..., 1] < 80) & (out2[..., 0] < 80)
    assert reds.sum() > 50


def test_options_uppercase_offset_stroke():
    load_all()
    img, boxes = _scene()
    base = RENDERERS.build("default", Settings())
    base.render(img, _bubbles(boxes[:1], ["halo dunia"]), img)
    r = RENDERERS.build(
        "default", Settings(), uppercase=True, font_size_offset=-6, stroke_width=0.2
    )
    r.render(img, _bubbles(boxes[:1], ["halo dunia"]), img)
    p = r.plans[0]
    assert p.layout.text == "HALO DUNIA"
    assert p.layout.size <= base.plans[0].layout.size - 6 + 1
    assert r.colors(p, 30)[2] == 6


def test_tilted_text_is_drawn_rotated():
    load_all()
    img = np.full((400, 400, 3), 220, np.uint8)
    cv2.ellipse(img, (200, 200), (170, 90), -20, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (200, 200), (170, 90), -20, 0, 360, (0, 0, 0), 3)
    b = [
        Bubble(
            region=TextRegion(
                id=0,
                bbox=BBox(x1=140, y1=170, x2=260, y2=230),
                direction="h",
                angle=20.0,
                font_size=36,
            ),
            source_text="x",
            target_text="miring sedikit ke atas",
        )
    ]
    r = RENDERERS.build("default", Settings())
    out = r.render(img, b, img)
    ink = np.argwhere(np.abs(out.astype(int) - img.astype(int)).sum(axis=2) > 200)
    # rising to the right: ink on the right sits higher than ink on the left
    left = ink[ink[:, 1] < np.median(ink[:, 1])][:, 0].mean()
    right = ink[ink[:, 1] >= np.median(ink[:, 1])][:, 0].mean()
    assert right < left - 5
    assert r.plans[0].angle == 20.0


# -- vertical -----------------------------------------------------------------------------


def test_vertical_columns_keep_text_and_kinsoku(measure):
    shape = Shape.from_mask(ellipse_mask(220, 160).T)  # transposed: rows are page columns
    text = "あっ、あとこないだね、髪型変えたらすぐに気付いてくれてっ・・"
    lay = fit_vertical(text, shape, measure, min_size=10, max_size=60)
    assert lay.fits and len(lay.columns) > 1
    assert lay.text.replace("︑", "、") == text
    assert all(col[0] not in NO_START for col in lay.columns)
    xs = [c.x for c in lay.cells]
    assert xs[0] > xs[-1]  # first column on the right


def test_vertical_renderer_draws_cjk():
    load_all()
    img, boxes = _scene()
    r = RENDERERS.build("vertical", Settings())
    bub = _bubbles(boxes[:1], ["ねえねえ、めいは坂本のことどう思う？"])
    out = r.render(img, bub, img)
    assert r.plans[0].vertical
    assert (np.abs(out.astype(int) - img.astype(int)).sum(axis=2) > 60).sum() > 200


# -- real page ----------------------------------------------------------------------------


@pytest.mark.skipif(
    not (PAGE.exists() and GT.exists() and CTD.exists()), reason="needs page01 + ctd"
)
def test_page01_reference_translations_fit_inside_balloons(tmp_path):
    load_all()
    from terjemahin_worker.detect import sort_reading_order
    from terjemahin_worker.registry import DETECTORS, INPAINTERS, MASK_REFINERS

    settings = Settings()
    page = cv2.imread(str(PAGE))
    det = DETECTORS.build("ctd", settings).detect(page)
    regions = sort_reading_order(det.regions, image=page)
    gt = json.loads(GT.read_text())["bubbles"]
    assert len(regions) >= len(gt)
    bubbles = [
        Bubble(region=r, source_text=g["ja"], target_text=g["id_ref"]) for r, g in zip(regions, gt)
    ]
    mask = MASK_REFINERS.build("bbox", settings).refine(page, [b.region for b in bubbles], None)
    clean = INPAINTERS.build("fill", settings).inpaint(page, mask)
    r = RENDERERS.build("default", settings)
    out = r.render(clean, bubbles, page)
    cv2.imwrite(str(tmp_path / "page.png"), out)

    H, W = page.shape[:2]
    assert len(r.plans) == 13
    assert all(p.closed for p in r.plans), [p.bubble.region.id for p in r.plans if not p.closed]
    sizes = [p.layout.size for p in r.plans]
    assert min(sizes) >= round((W + H) / 200)
    assert max(sizes) <= 1.25 * float(np.median(sizes))
    for p, b in zip(r.plans, bubbles):
        assert p.layout.fits
        assert p.layout.text == r.measure.normalize(b.target_text)
        bal = find_balloon(clean, page, b.region.bbox)
        full = np.zeros((H, W), np.uint8)
        ys, xs = bal.page_slice()
        full[ys, xs] = bal.mask
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        r._draw(canvas, p)
        ink = np.asarray(canvas).astype(int).sum(axis=2) < 600
        gap = cv2.distanceTransform(full, cv2.DIST_L2, 3)[ink]
        assert gap.min() >= 6, (p.text, gap.min())
