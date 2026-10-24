import itertools
import json
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import latency_budget

from terjemahin_worker.config import Settings
from terjemahin_worker.models import BBox, TextRegion
from terjemahin_worker.ocr.mit48px import (
    PIECE_WIDTH,
    SPLIT_WIDTH,
    TEXT_HEIGHT,
    as_bgr8,
    join_lines,
    line_direction,
    line_pieces,
    order_quad,
    reading_order,
    split_lines,
    vertical_forms,
    warp_line,
)

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = Settings().weights_dir
CKPT, ALPHABET = WEIGHTS / "ocr_ar_48px.ckpt", WEIGHTS / "alphabet-all-v7.txt"
PAGE = ROOT / "eval" / "fixtures" / "page01.png"
GT = ROOT / "eval" / "fixtures" / "page01.gt.json"
needs_model = pytest.mark.skipif(
    not (CKPT.exists() and ALPHABET.exists() and PAGE.exists()), reason="48px weights missing"
)


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def test_order_quad_any_start():
    q = rect(10, 20, 40, 120)
    for shift in range(4):
        np.testing.assert_array_equal(order_quad(np.roll(q, shift, 0)), q)
    np.testing.assert_array_equal(order_quad(q[::-1]), q)


def test_line_direction():
    assert line_direction(rect(0, 0, 30, 200), "h") == "v"
    assert line_direction(rect(0, 0, 200, 30), "v") == "h"
    assert line_direction(rect(0, 0, 40, 44), "v") == "v"
    assert line_direction(rect(0, 0, 40, 44), "h") == "h"


def test_warp_vertical_rotates_top_to_left():
    img = np.full((300, 100, 3), 255, np.uint8)
    img[20:60, 30:70] = (255, 0, 0)  # first glyph (top) red
    img[220:260, 30:70] = (0, 0, 255)  # last glyph (bottom) blue
    out = warp_line(img, rect(25, 10, 75, 270), "v")
    assert out.shape[0] == TEXT_HEIGHT
    assert out.shape[1] == round(TEXT_HEIGHT * 260 / 50)
    left, right = out[:, : out.shape[1] // 4], out[:, -out.shape[1] // 4 :]
    assert left[..., 0].mean() > left[..., 2].mean()  # red on the left
    assert right[..., 2].mean() > right[..., 0].mean()


def test_warp_horizontal_and_margin():
    img = np.random.default_rng(0).integers(0, 255, (100, 400, 3), np.uint8)
    out = warp_line(img, rect(10, 10, 310, 40), "h")
    assert out.shape == (TEXT_HEIGHT, 480, 3)
    wider = warp_line(img, rect(10, 10, 310, 40), "h", pad=0.1)
    assert wider.shape[0] == TEXT_HEIGHT and wider.shape[1] < out.shape[1]  # thicker -> shorter


def test_split_lines_columns_and_rows():
    gray = np.full((200, 200), 255, np.uint8)
    for x in (30, 80, 130):  # three columns of ink
        gray[20:180, x : x + 25] = 0
    r = TextRegion(id=0, bbox=BBox(x1=10, y1=10, x2=190, y2=190), direction="v")
    quads = split_lines(gray, r)
    assert len(quads) == 3
    order = reading_order(quads, "v")
    xs = [order_quad(quads[i])[0, 0] for i in order]
    assert xs == sorted(xs, reverse=True)  # right to left
    assert all(abs(order_quad(q)[0, 1] - 20) <= 1 for q in quads)  # trimmed to the ink

    gray = np.full((200, 300), 255, np.uint8)
    for y in (30, 90):
        gray[y : y + 30, 20:280] = 0
    r = TextRegion(id=0, bbox=BBox(x1=0, y1=0, x2=300, y2=200), direction="h")
    quads = split_lines(gray, r)
    assert len(quads) == 2
    assert [round(order_quad(quads[i])[0, 1]) for i in reading_order(quads, "h")] == [30, 90]


def test_split_lines_glues_punctuation():
    gray = np.full((200, 200), 255, np.uint8)
    gray[20:180, 60:90] = 0
    gray[170:180, 93:98] = 0  # a small comma just right of the column
    r = TextRegion(id=0, bbox=BBox(x1=40, y1=10, x2=120, y2=190), direction="v")
    assert len(split_lines(gray, r)) == 1


def test_split_lines_keeps_dashes_spaces_and_touching_glyphs():
    """Glyph filtering is by thickness across the line only, and a line runs over word or
    full-width spaces: a dash run, a fused bold blob and the text after a space all stay."""
    gray = np.full((400, 120), 255, np.uint8)
    gray[20:56, 40:76] = 0  # glyph
    gray[60:140, 56:60] = 0  # ―― : thin and 2+ em long along the column
    gray[190:226, 40:76] = 0  # after a full-width space (50px gap)
    gray[230:340, 38:78] = 0  # three touching bold glyphs, one 110px blob
    r = TextRegion(id=0, bbox=BBox(x1=0, y1=0, x2=120, y2=400), direction="v")
    (q,) = split_lines(gray, r)
    q = order_quad(q)
    assert q[0, 1] <= 20 and q[2, 1] >= 339
    gray = np.full((140, 600), 255, np.uint8)
    gray[30:60, 20:200] = 0  # touching Latin letters fused into one wide component
    gray[55:68, 206:210] = 0  # a comma hanging below the baseline
    gray[30:60, 260:330] = 0  # next word after ", "
    gray[90:120, 20:120] = 0  # second row
    r = TextRegion(id=0, bbox=BBox(x1=0, y1=0, x2=600, y2=140), direction="h")
    q1, q2 = (order_quad(split_lines(gray, r)[i]) for i in reading_order(split_lines(gray, r), "h"))
    assert q1[0, 0] <= 20 and q1[1, 0] >= 329 and q1[2, 1] >= 67  # comma tail inside
    assert q2[0, 1] >= 75  # rows do not bleed into each other


def test_split_lines_leaves_out_stray_strokes_past_the_text():
    gray = np.full((400, 100), 255, np.uint8)
    cv2.line(gray, (30, 2), (36, 30), 0, 3)  # speed line clipped by the box, 2+ em above
    gray[110:146, 30:66] = 0  # は
    gray[150:186, 30:66] = 0  # い
    gray[190:198, 30:38] = 0  # 、 small, then a full-width space
    gray[250:286, 30:66] = 0  # そ
    r = TextRegion(id=0, bbox=BBox(x1=0, y1=0, x2=100, y2=400), direction="v")
    (q,) = split_lines(gray, r)
    q = order_quad(q)
    assert 100 <= q[0, 1] <= 110 and q[2, 1] >= 285  # stroke out, comma and space bridged


def test_as_bgr8_normalises_layouts_and_dtypes():
    ref = np.random.default_rng(0).integers(0, 256, (8, 9), np.uint8)
    want = np.repeat(ref[..., None], 3, 2)
    for img in (
        ref,
        ref[..., None],
        ref.astype(np.float32),
        ref / 255.0,
        ref.astype(np.uint16) << 8,
    ):
        out = as_bgr8(img)
        assert out.dtype == np.uint8 and out.shape == (8, 9, 3)
        assert np.abs(out.astype(int) - want).max() <= 1


def test_text_helpers():
    assert join_lines(["ねえねえ、", "めいは"]) == "ねえねえ、めいは"
    assert join_lines(["HELLO", "WORLD"]) == "HELLO WORLD"
    assert join_lines(["HELLO THERE,", "HOW", "ARE YOU?", '"FINE."', "OK"]) == (
        'HELLO THERE, HOW ARE YOU? "FINE." OK'
    )
    assert join_lines(["WAIT...", "WHAT"]) == "WAIT... WHAT"
    assert join_lines(["WON-", "DERFUL"]) == "WON-DERFUL"
    assert join_lines(["どう思う?", "じゃあ"]) == "どう思う?じゃあ"
    assert join_lines(["えー!", "10円"]) == "えー!10円"
    assert vertical_forms("そうだなあ：") == "そうだなあ‥"
    assert vertical_forms("：あ:") == "‥あ‥"
    assert vertical_forms("10:30に") == "10:30に"  # a real colon between digits stays


def test_line_pieces_cut_at_blank_columns():
    short = np.full((TEXT_HEIGHT, SPLIT_WIDTH, 3), 255, np.uint8)
    assert line_pieces(short) == [(0, SPLIT_WIDTH, False)]
    w, pitch = 5000, 44
    crop = np.full((TEXT_HEIGHT, w, 3), 255, np.uint8)
    for x in range(0, w - pitch, pitch):  # 40px glyphs with 4px gaps
        crop[4:44, x : x + 40] = 0
    pieces = line_pieces(crop)
    assert pieces[0][0] == 0 and pieces[-1][1] == w
    assert all(a == pb for (_, pb, _), (a, _, _) in itertools.pairwise(pieces))  # contiguous
    assert all(b - a <= PIECE_WIDTH for a, b, _ in pieces)
    for _, b, _ in pieces[:-1]:
        assert crop[:, b].min() == 255  # never through a glyph


def test_xpos_cache_matches_reference_formula():
    torch = pytest.importorskip("torch")
    from terjemahin_worker.ocr._mit48px_model import XPOS, centred

    xp = XPOS()
    x = torch.randn(2, 7, 80)

    def reference(x, offset=0, downscale=False):  # manga-image-translator's XPOS.forward
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        pos = torch.arange(min_pos, max_pos, 1).to(xp.scale)
        scale = xp.scale ** pos.div(xp.scale_base)[:, None]
        n, d = scale.shape
        inv = 1.0 / (10000 ** (torch.arange(0, d) / d))
        ang = torch.einsum("i,j->ij", torch.arange(0, n, dtype=torch.float), inv)
        sin, cos = ang.sin()[-length:], ang.cos()[-length:]
        scale = scale[-length:]
        if downscale:
            scale = 1 / scale
        sin, cos = ((t * scale).repeat_interleave(2, -1) for t in (sin, cos))
        rot = torch.stack((-x[..., 1::2], x[..., ::2]), -1).flatten(-2)
        return x * cos + rot * sin

    idx = torch.arange(7)
    for down in (False, True):
        got = xp(x, idx, centred(idx, 7), downscale=down)
        torch.testing.assert_close(got, reference(x, downscale=down))
    q = x[:, :1]  # one decoder query at step 6, as in the reference's cached decoding
    got = xp(q, torch.tensor([6]), centred(torch.tensor([6]), 7))
    torch.testing.assert_close(got, reference(q, offset=6))


def test_beam_search_shapes_on_random_weights():
    torch = pytest.importorskip("torch")
    from terjemahin_worker.ocr._mit48px_model import OCR48px

    torch.manual_seed(0)
    m = OCR48px(50).eval()
    widths = [40, 96, 70]
    x = torch.rand(3, 3, 48, 103) * 2 - 1
    for beams in (1, 3):
        out = m.decode(x, widths, beams=beams, max_len=6)
        assert len(out) == 3
        for d in out:
            assert len(d.tokens) <= 6 and d.colors.shape == (len(d.tokens), 10)
            assert d.logprob <= 0


@pytest.fixture(scope="module")
def ocr():
    from terjemahin_worker.registry import OCRS, load_all

    load_all()
    return OCRS.build("48px", Settings())


def page_lines() -> list[TextRegion]:
    """Column boxes of six page01 bubbles (keyed by ground-truth index), frozen from a ctd run
    so this test does not depend on the detector."""
    cols = {
        0: [(1415, 11, 1461, 173), (1363, 11, 1407, 311), (1313, 11, 1351, 201)],
        1: [(1175, 219, 1225, 305), (1145, 219, 1185, 333)],
        5: [(621, 931, 677, 1039)],
        7: [(1442, 1437, 1481, 1675), (1406, 1434, 1440, 1674)],
        8: [(1092, 1472, 1120, 1600)],
        11: [
            (423, 1431, 461, 1581),
            (384, 1430, 418, 1604),
            (301, 1537, 339, 1745),
            (252, 1538, 294, 1778),
            (210, 1538, 244, 1708),
        ],
    }
    return [
        TextRegion(
            id=i,
            bbox=BBox(
                x1=min(c[0] for c in cs),
                y1=min(c[1] for c in cs),
                x2=max(c[2] for c in cs),
                y2=max(c[3] for c in cs),
            ),
            lines=[[(x0, y0), (x1, y0), (x1, y1), (x0, y1)] for x0, y0, x1, y1 in cs[::-1]],
            direction="v",
        )
        for i, cs in cols.items()
    ]


@needs_model
def test_reads_page01_lines(ocr):
    img = cv2.imread(str(PAGE))
    regions = page_lines()
    res = ocr.recognize(img, regions)
    gt = json.loads(GT.read_text(encoding="utf-8"))["bubbles"]
    for r, o in zip(regions, res):
        want = gt[r.id]["ja"]
        assert cer(o.text, want) <= 0.1, (o.text, want)
        assert o.confidence > 0.8
        assert o.fg_color is not None and max(o.fg_color) < 90  # black lettering
    # the same blocks without line quads go through the projection-profile split
    bare = ocr.recognize(img, [r.model_copy(update={"lines": []}) for r in regions])
    assert sum(cer(o.text, gt[r.id]["ja"]) for r, o in zip(regions, bare)) / len(regions) < 0.15


@needs_model
def test_predicts_glyph_and_outline_colours(ocr):
    img = cv2.imread(str(PAGE))
    regions = page_lines()[:2]
    dark = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 128
    red = img.copy()
    red[dark] = (40, 40, 200)  # BGR: red lettering
    (o,) = ocr.recognize(red, regions[:1])
    assert o.fg_color[0] > 150 and max(o.fg_color[1:]) < 90 and o.bg_color is None
    stroke = cv2.dilate(dark.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    outlined = np.full_like(img, 150)
    outlined[stroke] = 0
    outlined[dark] = 255  # white glyphs, black outline, grey ground
    for o, r in zip(ocr.recognize(outlined, regions), regions):
        assert cer(o.text, json.loads(GT.read_text(encoding="utf-8"))["bubbles"][r.id]["ja"]) < 0.1
        assert min(o.fg_color) > 180 and o.bg_color is not None and max(o.bg_color) < 60


@needs_model
def test_page01_cer_with_ctd(ocr):
    """End to end on the sample page with the ctd detector's line quads."""
    from terjemahin_worker.detect import sort_reading_order
    from terjemahin_worker.registry import DETECTORS

    det = DETECTORS.methods.get("ctd")
    if det is None or not (WEIGHTS / "koharu-comictextdetector.onnx").exists():
        pytest.skip("ctd detector unavailable")
    img = cv2.imread(str(PAGE))
    regions = sort_reading_order(DETECTORS.build("ctd", Settings()).detect(img).regions, image=img)
    t = time.perf_counter()
    res = ocr.recognize(img, regions)
    elapsed = time.perf_counter() - t
    texts = [o.text for o in res]
    gt = [b["ja"] for b in json.loads(GT.read_text(encoding="utf-8"))["bubbles"]]
    err = sum(min(lev(norm(t), norm(g)) for t in texts) for g in gt)
    assert err / sum(len(g) for g in gt) < 0.06
    assert elapsed < latency_budget(30)


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    for a in "‥…・:.。":
        t = t.replace(a, "・")
    return "".join(t.split())


def lev(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def cer(got: str, want: str) -> float:
    return lev(norm(got), norm(want)) / len(norm(want))


@needs_model
def test_degenerate_inputs(ocr):
    img = cv2.imread(str(PAGE))
    assert ocr.recognize(img, []) == []
    blank = np.full((200, 200, 3), 255, np.uint8)
    regions = [
        TextRegion(id=0, bbox=BBox(x1=10, y1=10, x2=60, y2=150)),  # empty white box
        TextRegion(id=1, bbox=BBox(x1=199, y1=199, x2=200, y2=200)),  # 1px
        TextRegion(
            id=2, bbox=BBox(x1=0, y1=0, x2=50, y2=50), lines=[[(5, 5), (6, 5), (6, 6), (5, 6)]]
        ),
    ]
    res = ocr.recognize(blank, regions)
    assert len(res) == 3 and all(r.text == "" or r.confidence < 0.5 for r in res)


@needs_model
def test_odd_images_and_off_page_regions(ocr):
    img = cv2.imread(str(PAGE))
    regions = page_lines()[:1]
    (want,) = ocr.recognize(img, regions)
    (gray,) = ocr.recognize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), regions)
    (bgra,) = ocr.recognize(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA), regions)
    assert gray.text == want.text == bgra.text
    h, w = img.shape[:2]
    off = [
        TextRegion(id=0, bbox=BBox(x1=w + 10, y1=h + 10, x2=w + 50, y2=h + 50), direction="h"),
        TextRegion(id=1, bbox=BBox(x1=-80, y1=-80, x2=-10, y2=-10), direction="v"),
        regions[0].model_copy(
            update={"lines": [[(w + 5, 0), (w + 45, 0), (w + 45, 90), (w + 5, 90)]]}
        ),
    ]
    res = ocr.recognize(img, off)
    assert [r.text for r in res[:2]] == ["", ""]
    assert cer(res[2].text, want.text) < 0.1  # off-page quad ignored, falls back to the box


@needs_model
def test_long_line_is_split_not_collapsed(ocr):
    """The model collapses on lines over ~2100px at 48px high; long lines are read in pieces."""
    from PIL import Image, ImageDraw, ImageFont

    text = (
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG AND THEN RUNS AWAY "
        "INTO THE DARK FOREST NEAR THE OLD RIVER BANK"
    )
    font = ImageFont.truetype(str(ROOT / "fonts" / "ComicShanns2.ttf"), 40)
    im = Image.new("RGB", (int(font.getlength(text)) + 40, 70), "white")
    ImageDraw.Draw(im).text((20, 10), text, font=font, fill="black")
    img = np.ascontiguousarray(np.array(im)[..., ::-1])
    assert img.shape[1] * TEXT_HEIGHT / 50 > SPLIT_WIDTH
    region = TextRegion(id=0, bbox=BBox(x1=0, y1=0, x2=img.shape[1], y2=70), direction="h")
    (o,) = ocr.recognize(img, [region])
    assert cer(o.text, text) < 0.03, o.text
    assert " " in o.text and o.fg_color is not None


def _font(*paths):
    return next((p for p in map(Path, paths) if p.exists()), None)


JA_FONT = _font(
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/home/user/zyddnys/manga-image-translator/fonts/msgothic.ttc",
)


@needs_model
def test_box_only_fallback_keeps_latin_words_and_punctuation(ocr):
    """Regions without line quads (classical, ctd_blk, yolo) go through split_lines; nothing
    after a comma-space, a touching pair or a line-final comma may be lost."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(ROOT / "fonts" / "ComicShanns2.ttf"), 40)
    for lines in (
        ["HELLO THERE, HOW", "ARE YOU DOING", "TODAY, FRIEND?"],
        ["I'M NOT GOING", "ANYWHERE, OKAY?"],
    ):
        w = max(int(font.getlength(t)) for t in lines) + 60
        im = Image.new("RGB", (w, len(lines) * 52 + 60), "white")
        for i, t in enumerate(lines):
            ImageDraw.Draw(im).text((30, 30 + i * 52), t, font=font, fill="black")
        img = np.ascontiguousarray(np.array(im)[..., ::-1])
        region = TextRegion(id=0, bbox=BBox(x1=0, y1=0, x2=w, y2=im.height), direction="h")
        (o,) = ocr.recognize(img, [region])
        assert o.text == " ".join(lines), o.text


def _raqm() -> bool:
    from PIL import features

    return bool(features.check("raqm"))


@needs_model
@pytest.mark.skipif(JA_FONT is None or not _raqm(), reason="no Japanese font / vertical layout")
def test_box_only_fallback_keeps_dashes_and_fullwidth_spaces(ocr):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(JA_FONT), 40)
    for cols in (
        ["待って――！！"],
        ["まさか――", "嘘だろ？"],
        ["はい、　そうです"],
        ["ちょっと　まって", "まだ　だめ"],
    ):
        w, h = len(cols) * 56 + 60, max(map(len, cols)) * 42 + 80
        im = Image.new("RGB", (w, h), "white")
        d = ImageDraw.Draw(im)
        for j, col in enumerate(cols):
            x = w - 30 - (j + 1) * 56
            d.text((x, 30), col, font=font, fill="black", direction="ttb", features=["vert"])
        img = np.ascontiguousarray(np.array(im)[..., ::-1])
        region = TextRegion(id=0, bbox=BBox(x1=0, y1=0, x2=w, y2=h), direction="v")
        (o,) = ocr.recognize(img, [region])
        dash = str.maketrans("―ー一", "---")  # a lone vertical bar is any of these
        got, want = o.text.translate(dash), "".join(cols).translate(dash)
        assert cer(got, want) <= 0.1 and len(norm(got)) >= len(norm(want)), o.text
