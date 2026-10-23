"""Synthetic manga pages: one run emits detector, OCR and inpainting training data.

Each page is a background (a crop of a text-free page from --backgrounds, or procedural paper
with panels, screentone, speed lines and doodles as hard negatives) with balloons, narration
boxes and free stroked text. Japanese is set vertically (top-to-bottom columns, right-to-left)
through raqm's `ttb` direction, which applies the fonts' `vert` forms (ー、「」… rotate or move
the way they do in print), or horizontally. Every pixel of text is known, so the outputs are
exact:

    out/det/{images,labels}/{train,val}/NNNNNN.{jpg,txt}   YOLO, one class "text" (block boxes)
    out/det/dataset.yaml                                   -> training/train.py --data
    out/ocr/crops/NNNNNN_KK.png + out/ocr/{train,val}.jsonl -> training/train_ocr.py --data
    out/inpaint/{clean,mask,texted}/NNNNNN.png             inpainter fine-tuning triplets
    out/pages.jsonl                                        every block: box, text, font, layout
    out/preview.jpg, out/ocr_preview.jpg                   eyeball these before training

The same --seed gives byte-identical outputs; page i depends only on (seed, i).

Usage:
    python training/synth.py --out data/synth --n 2000 \
        --fonts /path/to/fonts --backgrounds data/clean-pages --corpus lines.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_EXT = {".ttf", ".otf", ".ttc"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
FONT_DIRS = [
    ROOT / "fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path.home() / ".local/share/fonts",
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
    Path("C:/Windows/Fonts"),
]
DEFAULT_VOCAB = ROOT / "weights" / "koharu-vocab.txt"
# never start a column/line with these (simple kinsoku shori)
ELLIPSES = ["…", "…", "・・", "・・・", "……", "．．．"]
NO_START = set("、。，．！？!?ーっゃゅょッャュョ」』）)…・〜～")

CORPUS = """\
ねえ、ちょっと待ってよ！
なんでそんなこと言うの？
お前には関係ないだろ
今日は本当にありがとう
大丈夫？ケガはない？
もう行かなきゃ…
信じられない…！
あいつ、また遅刻か
俺が守ってやる
そんなの無理に決まってるじゃん
え？今なんて言った？
ちょっと、聞いてる？
やっぱりここにいたんだ
早く逃げろ！！
これで終わりだと思うなよ
うるさいなぁ
わかった、行くよ
あの時の約束、覚えてる？
ずっと前から好きでした
冗談だよ、冗談
何してるの、こんなところで
いいから黙ってついて来い
まさか…お前が犯人なのか！？
ごめん、遅くなった
お腹すいたー
明日、学校休みだって
先生に怒られちゃった
うそでしょ！？
ふざけるな！
静かにしろ、誰か来る
その手を離せ！
へえ、意外とやるじゃん
もう少しだけ、ここにいさせて
ほら、見てごらん
この店のラーメン、めちゃくちゃうまいぞ
次の試合、絶対に勝つ！
あれ？鍵がない…
おはようございます
いってきまーす！
ただいまー
おかえりなさい
いただきます
お疲れさまでした
はじめまして、よろしくお願いします
どうしてわかってくれないの
仕方ないなぁ、今回だけだぞ
へっ、面白くなってきやがった
しっかりしろ！目を開けろ！
そこをどけ
助けて…誰か…
あ、あの…これ、受け取ってください
気にしないで
本気で言ってるのか？
私だって頑張ってるんだから！
ちょっと休憩しようか
なにこれ、かわいい！
すごい…！
どこに行ってたんだよ
いい加減にしてよね
きっと大丈夫だよ
お前、何者だ？
ここから先は通さない
あはは、そうかもね
じゃあ、また明日
それって、どういう意味？
覚悟しろよ
やめてくれ！
待ってたよ
ねえねえ、聞いて聞いて！
ひどいよ、そんなの
やれやれ…
もう二度と会えないかもしれない
バカ！心配したんだから！
本当にいいの？
僕にも手伝わせてください
すみません、道に迷ってしまって
この学校の七不思議って知ってる？
世界を救うのはお前だ
魔王を倒すまで帰らない
くそっ、間に合わない！
ほんとに？やったー！
しーっ、声が大きい
ふーん、そういうこと
ちがう、そうじゃない
何度言ったらわかるんだ
今度の日曜日、空いてる？
映画でも見に行かない？
えっと、その…
とにかく急げ！
へへっ、どんなもんだ
油断するな、まだ終わってない
これが俺の本当の力だ
おい、起きろ！朝だぞ！
もう眠い…
宿題、終わった？
全然ダメだ…
よし、行くぞ！
誰だ！そこにいるのは！
一緒に帰ろう
わたし、負けないから
あんたなんか大嫌い！
ありがとう…本当に
なんだと！？
ついに見つけたぞ
この勝負、もらった！
ちょっとだけだよ？
三年前のあの日から、すべてが変わった
父さん、母さん…
お前の相手は俺だ
今日の晩ごはん、カレーだって
雨、やまないね
ここが噂の神社か
何か隠してるでしょ
逃がすか！
いくぞ、みんな！
はぁ…はぁ…
きゃあああっ！
うおおおおっ！
ドキドキする…
お兄ちゃん、待って！
部長、大変です！
会議は三時からです
この件は私に任せてください
社長がお呼びです
電車に乗り遅れた！
ケーキ、半分こしよう
秘密だよ
ずるい！
約束だからね
お前のせいだ
負けてたまるか
ちゃんと話を聞いて
今すぐここを出るんだ
もう、知らない！
""".splitlines()


@dataclass(frozen=True)
class FontFace:
    path: str
    index: int
    vertical: bool


@dataclass
class Block:
    text: str
    bbox: list[int]  # x1 y1 x2 y2, ink incl. stroke
    vertical: bool
    font: str
    font_size: int
    container: str
    lines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------- fonts


@lru_cache(maxsize=256)
def load_font(path: str, index: int, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=index, layout_engine=ImageFont.Layout.RAQM)


def _render(font: ImageFont.FreeTypeFont, text: str, direction: str = "ltr") -> np.ndarray:
    s = int(font.size * (len(text) + 2))
    im = Image.new("L", (s, s) if direction == "ttb" else (s, font.size * 3))
    ImageDraw.Draw(im).text((font.size, font.size), text, font=font, fill=255, direction=direction)
    return np.asarray(im)


@lru_cache(maxsize=64)
def _missing_glyph(path: str, index: int) -> bytes:
    return _render(load_font(path, index, 32), "\U000f0000").tobytes()


@lru_cache(maxsize=200_000)
def has_glyph(path: str, index: int, ch: str) -> bool:
    if ch.isspace():
        return True
    a = _render(load_font(path, index, 32), ch)
    return a.any() and a.tobytes() != _missing_glyph(path, index)


def probe_font(path: str, index: int) -> FontFace | None:
    """A face that draws kana and kanji; `vertical` when its ttb layout stacks glyphs properly."""
    try:
        font = load_font(path, index, 32)
    except OSError:
        return None
    if not all(has_glyph(path, index, c) for c in "あカ漢字。"):
        return None
    ys, xs = np.nonzero(_render(font, "あいう国", "ttb"))
    stacked = len(ys) > 0 and np.ptp(ys) > 3.3 * 32 and np.ptp(xs) < 1.4 * 32
    ys, xs = np.nonzero(_render(font, "ー", "ttb"))  # `vert` turns the long vowel mark upright
    vertical = stacked and len(ys) > 0 and np.ptp(ys) > 2 * np.ptp(xs)
    return FontFace(path, index, bool(vertical))


def find_fonts(sources: list[Path]) -> list[FontFace]:
    files: list[Path] = []
    for s in sources:
        if s.is_file():
            files.append(s)
        elif s.is_dir():
            files += [p for p in s.rglob("*") if p.suffix.lower() in FONT_EXT]
    faces = []
    for p in sorted(set(files)):
        for idx in range(4 if p.suffix.lower() == ".ttc" else 1):
            f = probe_font(str(p), idx)
            if f is None:
                break
            faces.append(f)
    return faces


# ---------------------------------------------------------------------------- text


def load_corpus(path: Path | None, vocab: set[str] | None) -> list[str]:
    raw = path.read_text(encoding="utf-8").splitlines() if path else CORPUS
    lines = [ln.strip() for ln in raw if ln.strip() and not ln.lstrip().startswith("#")]
    if vocab:
        lines = [ln for ln in lines if all(c in vocab for c in ocr_target(ln))]
    if not lines:
        raise SystemExit("corpus is empty after filtering by the OCR vocab")
    return lines


def ocr_target(text: str) -> str:
    """What manga-ocr's tokenizer sees: NFKC (！→!, …→...), the form its vocab is built on."""
    return unicodedata.normalize("NFKC", text)


def load_vocab(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    return {t for t in path.read_text(encoding="utf-8").split("\n") if t and not t.startswith("[")}


def wrap(text: str, width: int) -> list[str]:
    """Greedy wrap at `width` chars that never starts a line with closing punctuation."""
    out: list[str] = []
    for chunk in text.split("\n"):
        cur = ""
        for ch in chunk:
            if len(cur) >= width and ch not in NO_START:
                out.append(cur)
                cur = ""
            cur += ch
        if cur:
            out.append(cur)
    return out


def split_corpus(lines: list[str], ratio: float, seed: int) -> tuple[list[str], list[str]]:
    """Disjoint (train, val) lines so validation text is unseen; tiny corpora are shared."""
    k = round(len(lines) * ratio)
    if k < 1 or len(lines) - k < 1:
        return lines, lines
    order = np.random.default_rng(seed).permutation(len(lines))
    return [lines[i] for i in sorted(order[k:])], [lines[i] for i in sorted(order[:k])]


def sample_text(rng: np.random.Generator, corpus: list[str]) -> str:
    n = 1 if rng.random() < 0.65 else int(rng.integers(2, 4))
    picks = [corpus[int(i)] for i in rng.integers(0, len(corpus), n)]
    text = ("\n" if rng.random() < 0.5 else "").join(picks)
    # letterers write trailing-off dots many ways; the corpus only has …
    return "".join(str(rng.choice(ELLIPSES)) if c == "…" else c for c in text)


# ---------------------------------------------------------------------------- rendering


def render_block(
    lines: list[str],
    face: FontFace,
    size: int,
    vertical: bool,
    stroke: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """(fill alpha, fill+stroke alpha) of a text block, cropped to the stroked ink."""
    font = load_font(face.path, face.index, size)
    pitch = size * (1.0 + rng.uniform(0.05, 0.45))
    pad = size + stroke * 2
    centered = rng.random() < 0.5
    if vertical:
        lengths = [font.getlength(c, direction="ttb") for c in lines]
        H = int(max(lengths)) + 2 * pad
        W = int(pitch * len(lines)) + 2 * pad
    else:
        lengths = [font.getlength(c) for c in lines]
        W = int(max(lengths)) + 2 * pad
        H = int(pitch * len(lines)) + 2 * pad
    fill_im, outer_im = Image.new("L", (W, H)), Image.new("L", (W, H))
    df, do = ImageDraw.Draw(fill_im), ImageDraw.Draw(outer_im)
    for i, (text, length) in enumerate(zip(lines, lengths)):
        if vertical:
            cx = W - pad - pitch * (i + 0.5)  # columns run right to left
            y = pad + ((max(lengths) - length) / 2 if centered else 0)
            # ttb anchors on ink, so place the column by where its ink sits next to a full-em glyph
            x = cx - _guo_width(face, size) / 2 + _col_shift(face, size, text)
            kw = {"direction": "ttb", "anchor": "lt"}
            pos = (x, y)
        else:
            x = pad + ((max(lengths) - length) / 2 if centered else 0)
            pos = (x, pad + pitch * i)
            kw = {"anchor": "la"}
        df.text(pos, text, font=font, fill=255, **kw)
        do.text(pos, text, font=font, fill=255, stroke_width=stroke, stroke_fill=255, **kw)
    fill, outer = np.asarray(fill_im), np.asarray(outer_im)
    ys, xs = np.nonzero(outer > 32)
    if not len(ys):
        return fill[:1, :1], outer[:1, :1]
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return fill[y1:y2, x1:x2], outer[y1:y2, x1:x2]


@lru_cache(maxsize=4096)
def _guo_width(face: FontFace, size: int) -> int:
    x1, _, x2, _ = load_font(face.path, face.index, size).getbbox("国", direction="ttb")
    return x2 - x1


def _col_shift(face: FontFace, size: int, text: str) -> float:
    """Ink-left of a ttb column relative to the ink-left of a full-width 国 stacked above it.

    PIL anchors vertical text on its ink, so a column of narrow glyphs (ー、。) would drift
    sideways; measuring against 国 puts every column's em box on the same centre line.
    """
    font = load_font(face.path, face.index, size)
    im = Image.new("L", (size * 4, int(size * (len(text) + 4))))
    ImageDraw.Draw(im).text((size, size), "国" + text, font=font, fill=255, direction="ttb",
                            anchor="lt")
    a = np.asarray(im)
    split = size + int(font.getlength("国", direction="ttb"))
    hx, tx = np.nonzero(a[:split].any(0))[0], np.nonzero(a[split:].any(0))[0]
    return float(tx.min() - hx.min()) if len(hx) and len(tx) else 0.0


def composite(page: np.ndarray, alpha: np.ndarray, color, x: int, y: int) -> None:
    h, w = alpha.shape
    roi = page[y : y + h, x : x + w].astype(np.float32)
    a = (alpha.astype(np.float32) / 255)[..., None]
    page[y : y + h, x : x + w] = (roi * (1 - a) + np.float32(color) * a + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------- backgrounds


def halftone(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Dot screentone over a linear gradient, as 0..1 ink coverage."""
    period = rng.uniform(3.5, 8.0)
    ang = math.radians(rng.choice([45.0, 30.0, 0.0, 15.0]))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx * math.cos(ang) + yy * math.sin(ang)
    v = -xx * math.sin(ang) + yy * math.cos(ang)
    pattern = (np.cos(2 * np.pi * u / period) + np.cos(2 * np.pi * v / period)) / 4 + 0.5
    g_ang = rng.uniform(0, 2 * np.pi)
    grad = xx * math.cos(g_ang) + yy * math.sin(g_ang)
    grad = (grad - grad.min()) / max(1e-6, float(np.ptp(grad)))
    lo, hi = sorted(rng.uniform(0.05, 0.7, 2))
    level = lo + (hi - lo) * grad
    return (pattern < level).astype(np.float32)


def doodle(rng: np.random.Generator, page: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """Smooth random strokes and shapes standing in for line art (detector hard negatives)."""
    for _ in range(int(rng.integers(2, 14))):
        n = int(rng.integers(3, 7))
        pts = np.stack([rng.uniform(x1, x2, n), rng.uniform(y1, y2, n)], 1)
        t = np.linspace(0, 1, 40)[:, None]
        curve = pts[0] * (1 - t) ** 2 + pts[1] * 2 * t * (1 - t) + pts[-1] * t**2
        color = int(rng.choice([0, 0, 40, 90]))
        thick = int(rng.integers(1, 5))
        if rng.random() < 0.3:
            c = (int(rng.uniform(x1, x2)), int(rng.uniform(y1, y2)))
            r = int(rng.uniform(5, max(6, (x2 - x1) / 4)))
            cv2.circle(page, c, r, (color,) * 3, thick, cv2.LINE_AA)
        else:
            cv2.polylines(page, [curve.astype(np.int32)], False, (color,) * 3, thick, cv2.LINE_AA)


def speed_lines(rng: np.random.Generator, page: np.ndarray, x1, y1, x2, y2) -> None:
    cx, cy = rng.uniform(x1, x2), rng.uniform(y1, y2)
    radius = math.hypot(x2 - x1, y2 - y1)
    for a in rng.uniform(0, 2 * np.pi, int(rng.integers(30, 120))):
        r0 = rng.uniform(0.25, 0.7) * radius
        p0 = (int(cx + r0 * math.cos(a)), int(cy + r0 * math.sin(a)))
        p1 = (int(cx + radius * math.cos(a)), int(cy + radius * math.sin(a)))
        cv2.line(page, p0, p1, (0, 0, 0), int(rng.integers(1, 3)), cv2.LINE_AA)


def split_panels(rng, x1, y1, x2, y2, depth=0) -> list[tuple[int, int, int, int]]:
    w, h = x2 - x1, y2 - y1
    if depth >= 3 or min(w, h) < 260 or (depth > 0 and rng.random() < 0.3):
        return [(x1, y1, x2, y2)]
    gutter = int(rng.integers(10, 26))
    if h >= w or rng.random() < 0.3:
        cut = int(y1 + h * rng.uniform(0.3, 0.7))
        return split_panels(rng, x1, y1, x2, cut - gutter // 2, depth + 1) + split_panels(
            rng, x1, cut + gutter // 2, x2, y2, depth + 1
        )
    cut = int(x1 + w * rng.uniform(0.3, 0.7))
    return split_panels(rng, x1, y1, cut - gutter // 2, y2, depth + 1) + split_panels(
        rng, cut + gutter // 2, y1, x2, y2, depth + 1
    )


def procedural_background(rng: np.random.Generator, W: int, H: int) -> np.ndarray:
    paper = rng.uniform(232, 255)
    page = np.clip(paper + rng.normal(0, rng.uniform(0.5, 4), (H, W)), 0, 255).astype(np.uint8)
    page = np.repeat(page[..., None], 3, 2)
    margin = int(rng.integers(20, 60))
    for x1, y1, x2, y2 in split_panels(rng, margin, margin, W - margin, H - margin):
        kind = rng.choice(["blank", "tone", "tone", "gradient", "speed", "hatch"], p=None)
        roi = page[y1:y2, x1:x2]
        if kind == "tone":
            cov = halftone(rng, y2 - y1, x2 - x1)
            roi[:] = (roi * (1 - cov[..., None] * rng.uniform(0.6, 1.0))).astype(np.uint8)
        elif kind == "gradient":
            g = np.linspace(rng.uniform(120, 255), rng.uniform(120, 255), y2 - y1)
            roi[:] = np.minimum(roi, g[:, None, None]).astype(np.uint8)
        elif kind == "speed":
            speed_lines(rng, roi, 0, 0, x2 - x1, y2 - y1)
        elif kind == "hatch":
            step = int(rng.integers(4, 12))
            for k in range(-(y2 - y1), x2 - x1, step):
                cv2.line(roi, (k, 0), (k + (y2 - y1), y2 - y1), (60, 60, 60), 1, cv2.LINE_AA)
        if rng.random() < 0.8:
            doodle(rng, roi, 0, 0, x2 - x1, y2 - y1)
        border = int(rng.integers(2, 6))
        cv2.rectangle(page, (x1, y1), (x2, y2), (0, 0, 0), border, cv2.LINE_AA)
    return page


def image_background(rng: np.random.Generator, path: Path, W: int, H: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return procedural_background(rng, W, H)
    h, w = img.shape[:2]
    s = rng.uniform(0.45, 1.0)
    ch, cw = int(h * s), int(min(w, h * s * W / H))
    y, x = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    crop = cv2.resize(img[y : y + ch, x : x + cw], (W, H), interpolation=cv2.INTER_AREA)
    crop = crop[..., ::-1]
    if rng.random() < 0.7:
        crop = np.repeat(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)[..., None], 3, 2)
    return np.ascontiguousarray(crop)


# ---------------------------------------------------------------------------- containers


def shape_mask(rng, kind: str, w: int, h: int) -> tuple[np.ndarray, int, int]:
    """Balloon mask around a w x h text box, cropped; plus the box's top-left inside it."""
    m0 = int(0.6 * max(w, h)) + 40
    m = np.zeros((h + 2 * m0, w + 2 * m0), np.uint8)
    cx, cy = m0 + w / 2, m0 + h / 2
    if kind == "rect":
        mx, my = rng.uniform(0.15, 0.4, 2) * min(w, h) + 6
        cv2.rectangle(m, (int(m0 - mx), int(m0 - my)), (int(m0 + w + mx), int(m0 + h + my)), 1, -1)
    else:
        k = rng.uniform(1.18, 1.38)
        ax, ay = w / 2 * k + 8, h / 2 * k + 8
        if kind == "spiky":
            n = int(rng.integers(14, 28))
            ang = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
            r = np.where(np.arange(2 * n) % 2 == 0, 1.35, 1.0) * rng.uniform(0.95, 1.1, 2 * n)
            pts = np.stack([cx + ax * r * np.cos(ang), cy + ay * r * np.sin(ang)], 1)
            cv2.fillPoly(m, [pts.astype(np.int32)], 1, cv2.LINE_AA)
        else:
            cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 1, -1, cv2.LINE_AA)
            if kind == "cloud":
                for a in np.linspace(0, 2 * np.pi, int(rng.integers(8, 14)), endpoint=False):
                    r = int(min(ax, ay) * rng.uniform(0.25, 0.4))
                    c = (int(cx + ax * math.cos(a)), int(cy + ay * math.sin(a)))
                    cv2.circle(m, c, r, 1, -1)
            if rng.random() < 0.6:  # tail
                a = rng.uniform(0, 2 * np.pi)
                base = np.array([cx + ax * 0.8 * math.cos(a), cy + ay * 0.8 * math.sin(a)])
                tip = np.array([cx + ax * 1.5 * math.cos(a), cy + ay * 1.5 * math.sin(a)])
                nrm = np.array([-math.sin(a), math.cos(a)]) * min(ax, ay) * 0.25
                cv2.fillPoly(m, [np.array([base + nrm, base - nrm, tip]).astype(np.int32)], 1)
    ys, xs = np.nonzero(m)
    y1, x1 = ys.min(), xs.min()
    return m[y1 : ys.max() + 1, x1 : xs.max() + 1], int(m0 - x1), int(m0 - y1)


def draw_container(page: np.ndarray, mask: np.ndarray, x: int, y: int, border: int, fill: int):
    """Outline + fill of a local balloon mask placed with its top-left at (x, y)."""
    pad = border + 2
    m = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    H, W = page.shape[:2]
    x0, y0 = x - pad, y - pad
    sx, sy = max(0, -x0), max(0, -y0)
    ex, ey = min(m.shape[1], W - x0), min(m.shape[0], H - y0)
    m = m[sy:ey, sx:ex]
    roi = page[y0 + sy : y0 + ey, x0 + sx : x0 + ex].astype(np.float32)
    ring = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1,) * 2))
    soft = cv2.GaussianBlur(ring.astype(np.float32), (3, 3), 0)[..., None]
    inner = cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0)[..., None]
    roi = roi * (1 - soft)
    roi = roi * (1 - inner) + fill * inner
    page[y0 + sy : y0 + ey, x0 + sx : x0 + ex] = (roi + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------- page


@dataclass
class Config:
    width: tuple[int, int] = (800, 1200)
    blocks: tuple[int, int] = (2, 9)
    font_px: tuple[float, float] = (0.016, 0.042)  # font size as a fraction of page width
    vertical: float = 0.75


def make_page(
    i: int,
    seed: int,
    fonts: list[FontFace],
    corpus: list[str],
    backgrounds: list[Path],
    cfg: Config,
):
    rng = np.random.default_rng([seed, i])
    W = int(rng.integers(*cfg.width))
    H = int(W * rng.uniform(1.35, 1.5))
    if backgrounds and rng.random() < 0.6:
        page = image_background(rng, backgrounds[int(rng.integers(len(backgrounds)))], W, H)
    else:
        page = procedural_background(rng, W, H)
    occupied: list[tuple[int, int, int, int]] = []
    placed: list[tuple] = []
    for _ in range(int(rng.integers(*cfg.blocks))):
        vfonts = [f for f in fonts if f.vertical]
        vertical = bool(vfonts) and rng.random() < cfg.vertical
        pool = vfonts if vertical else fonts
        face = pool[int(rng.integers(len(pool)))]
        text = ""
        for _ in range(8):
            text = sample_text(rng, corpus)
            if all(has_glyph(face.path, face.index, c) for c in text if c != "\n"):
                break
        else:
            continue
        size = max(12, int(W * rng.uniform(*cfg.font_px)))
        width = int(rng.integers(3, 11)) if vertical else int(rng.integers(5, 15))
        lines = wrap(text, width)
        if len(lines) > 6:
            lines = wrap(text, math.ceil(len(text) / 6) + 1)[:6]
        container = str(rng.choice(["ellipse", "ellipse", "ellipse", "cloud", "spiky", "rect",
                                    "none"], p=[0.25, 0.2, 0.15, 0.08, 0.07, 0.12, 0.13]))
        if container == "none":
            stroke = int(rng.integers(2, max(3, size // 6) + 1))
            dark = rng.random() < 0.6
            fg, sc = ((0, 0, 0), (255, 255, 255)) if dark else ((255, 255, 255), (0, 0, 0))
        else:
            stroke = 0 if rng.random() < 0.85 else 2
            fg, sc = (int(rng.integers(0, 50)),) * 3, (255, 255, 255)
        fill, outer = render_block(lines, face, size, vertical, stroke, rng)
        h, w = outer.shape
        if w < 4 or h < 4 or w > W * 0.8 or h > H * 0.6:
            continue
        if container == "none":
            cmask, ox, oy = None, 4, 4
            bw, bh = w + 8, h + 8
        else:
            cmask, ox, oy = shape_mask(rng, container, w, h)
            bh, bw = cmask.shape
        if bw >= W - 8 or bh >= H - 8:
            continue
        for _ in range(40):
            bx, by = int(rng.integers(4, W - bw - 3)), int(rng.integers(4, H - bh - 3))
            box = (bx - 6, by - 6, bx + bw + 6, by + bh + 6)
            if not any(_overlap(box, o) for o in occupied):
                break
        else:
            continue
        occupied.append(box)
        x, y = bx + ox, by + oy
        if cmask is not None:
            draw_container(page, cmask, bx, by, int(rng.integers(2, 5)), int(rng.integers(240, 256)))
        block = Block(
            text="".join(lines), bbox=[x, y, x + w, y + h], vertical=vertical,
            font=f"{Path(face.path).name}#{face.index}", font_size=size, container=container,
            lines=lines,
        )
        placed.append((block, fill, outer if stroke else None, fg, sc, x, y))

    clean = page.copy()
    mask = np.zeros((H, W), np.uint8)
    for _, fill, outer, fg, sc, x, y in placed:
        if outer is not None:
            composite(page, outer, sc, x, y)
        composite(page, fill, fg, x, y)
        ink = fill if outer is None else outer
        h, w = ink.shape
        np.maximum(mask[y : y + h, x : x + w], (ink > 32).astype(np.uint8) * 255,
                   out=mask[y : y + h, x : x + w])
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return clean, page, mask, [b for b, *_ in placed], rng


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def degrade(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    """Scan/compression look for detector and OCR inputs (triplets stay pixel exact)."""
    out = img
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.3, 1.0))
    if rng.random() < 0.4:
        out = np.clip(out + rng.normal(0, rng.uniform(1, 6), out.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        q = int(rng.integers(55, 95))
        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


# ---------------------------------------------------------------------------- driver


@dataclass
class Job:
    out: Path
    seed: int
    fonts: list[FontFace]
    corpus: list[str]
    val_corpus: list[str]
    backgrounds: list[Path]
    cfg: Config
    val: set[int]
    preview: int
    emit: frozenset[str] = frozenset({"det", "ocr", "inpaint"})


def write_page(job: Job, i: int) -> tuple[dict, list[tuple[str, dict]], np.ndarray | None, list]:
    """Render page i and write its files; returns (page record, ocr rows, thumb, preview crops)."""
    out = job.out
    split = "val" if i in job.val else "train"
    corpus = job.val_corpus if split == "val" else job.corpus
    clean, texted, mask, blocks, rng = make_page(
        i, job.seed, job.fonts, corpus, job.backgrounds, job.cfg
    )
    stem = f"{i:06d}"
    H, W = mask.shape
    bgr_texted = np.ascontiguousarray(texted[..., ::-1])
    if "inpaint" in job.emit:
        cv2.imwrite(str(out / "inpaint" / "clean" / f"{stem}.png"), clean[..., ::-1])
        cv2.imwrite(str(out / "inpaint" / "texted" / f"{stem}.png"), bgr_texted)
        cv2.imwrite(str(out / "inpaint" / "mask" / f"{stem}.png"), mask)
    seen = degrade(rng, bgr_texted)
    if "det" in job.emit:
        cv2.imwrite(str(out / "det" / "images" / split / f"{stem}.jpg"), seen,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    labels, rows, crops = [], [], []
    for k, b in enumerate(blocks):
        x1, y1, x2, y2 = b.bbox
        labels.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                      f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        # same margin family as ocr.common.padded_box (0.12 x glyph size), jittered per side
        p = rng.uniform(0.04, 0.35, 4) * b.font_size + 2
        cx1, cy1 = max(0, int(x1 - p[0])), max(0, int(y1 - p[1]))
        cx2, cy2 = min(W, int(x2 + p[2])), min(H, int(y2 + p[3]))
        name = f"{stem}_{k:02d}.png"
        crop = seen[cy1:cy2, cx1:cx2]
        if "ocr" in job.emit:
            cv2.imwrite(str(out / "ocr" / "crops" / name), crop)
            rows.append((split, {"image": f"crops/{name}", "text": b.text,
                                 "vertical": b.vertical, "font": b.font,
                                 "font_size": b.font_size}))
        if i < job.preview:
            crops.append((crop, b.text))
    if "det" in job.emit:
        (out / "det" / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(labels) + ("\n" if labels else "")
        )
    thumb = None
    if i < job.preview:
        vis = seen.copy()
        for b in blocks:
            cv2.rectangle(vis, tuple(b.bbox[:2]), tuple(b.bbox[2:]), (0, 0, 255), 3)
        thumb = cv2.resize(vis, (int(W * 360 / H), 360), interpolation=cv2.INTER_AREA)
    record = {"image": stem, "split": split, "width": W, "height": H,
              "blocks": [asdict(b) for b in blocks]}
    return record, rows, thumb, crops


_JOB: Job | None = None


def _init_worker(job: Job) -> None:
    global _JOB
    _JOB = job
    cv2.setNumThreads(1)


def _work(i: int):
    assert _JOB is not None
    return write_page(_JOB, i)


def generate(
    out: Path,
    n: int,
    seed: int,
    fonts: list[FontFace],
    corpus: list[str],
    backgrounds: list[Path] | None = None,
    val_ratio: float = 0.1,
    cfg: Config | None = None,
    preview: int = 12,
    workers: int = 1,
    val_corpus_ratio: float = 0.15,
    emit: frozenset[str] = frozenset({"det", "ocr", "inpaint"}),
) -> dict:
    if not fonts:
        raise SystemExit("no Japanese-capable fonts found; pass --fonts DIR_OR_FILE")
    dirs = [out / "det" / k / s for k in ("images", "labels") for s in ("train", "val")]
    dirs += [out / "ocr" / "crops"] + [out / "inpaint" / k for k in ("clean", "mask", "texted")]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    n_val = max(1, round(n * val_ratio)) if n > 1 and val_ratio > 0 else 0
    val = set(np.random.default_rng(seed).permutation(n)[:n_val].tolist())
    train_lines, val_lines = split_corpus(corpus, val_corpus_ratio if n_val else 0.0, seed)
    job = Job(out, seed, fonts, train_lines, val_lines, backgrounds or [], cfg or Config(), val,
              preview, frozenset(emit))

    if workers > 1:
        import multiprocessing as mp

        # spawn, not fork: a fork after OpenCV has started its thread pool can deadlock
        pool = mp.get_context("spawn").Pool(workers, _init_worker, (job,))
        results = pool.imap(_work, range(n), chunksize=4)
    else:
        pool, results = None, (write_page(job, i) for i in range(n))

    thumbs, crops_preview = [], []
    n_blocks = n_vertical = 0
    jsonl = {k: out / "ocr" / f"{k}.jsonl" for k in ("train", "val")}
    try:
        with (
            open(jsonl["train"], "w", encoding="utf-8") as tr,
            open(jsonl["val"], "w", encoding="utf-8") as va,
            open(out / "pages.jsonl", "w", encoding="utf-8") as pages_f,
        ):
            ocr = {"train": tr, "val": va}
            for record, rows, thumb, crops in results:
                pages_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                for split, row in rows:
                    ocr[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                n_blocks += len(record["blocks"])
                n_vertical += sum(b["vertical"] for b in record["blocks"])
                if thumb is not None:
                    thumbs.append(thumb)
                crops_preview += crops[: max(0, 24 - len(crops_preview))]
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    (out / "det" / "dataset.yaml").write_text(
        f"path: {(out / 'det').resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: text\n"
    )
    if thumbs:
        cv2.imwrite(str(out / "preview.jpg"), grid(thumbs, 360), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if crops_preview:
        cv2.imwrite(str(out / "ocr_preview.jpg"), crop_sheet(crops_preview, fonts[0]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    meta = {"pages": n, "val_pages": n_val, "blocks": n_blocks, "vertical_blocks": n_vertical,
            "seed": seed, "fonts": [f"{f.path}#{f.index}" for f in fonts],
            "vertical_fonts": sum(f.vertical for f in fonts), "corpus_lines": len(corpus),
            "val_only_lines": 0 if val_lines is train_lines else len(val_lines),
            "backgrounds": len(job.backgrounds)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def grid(images: list[np.ndarray], height: int, cols: int = 6) -> np.ndarray:
    tiles = [cv2.resize(im, (int(im.shape[1] * height / im.shape[0]), height)) for im in images]
    tw = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 4, 4, 4, 4 + tw - t.shape[1], cv2.BORDER_CONSTANT,
                                value=(128, 128, 128)) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 128))
    rows = [np.hstack(tiles[r : r + cols]) for r in range(0, len(tiles), cols)]
    return np.vstack(rows)


def crop_sheet(crops: list[tuple[np.ndarray, str]], face: FontFace, h: int = 160) -> np.ndarray:
    """Crops with their labels underneath, to check text/crop alignment by eye."""
    font = load_font(face.path, face.index, 18)
    cells = []
    for crop, text in crops:
        c = cv2.resize(crop, (max(1, int(crop.shape[1] * h / crop.shape[0])), h))
        c = c[:, :300]
        cell = Image.new("RGB", (300, h + 50), (255, 255, 255))
        cell.paste(Image.fromarray(c[..., ::-1]), (0, 0))
        d = ImageDraw.Draw(cell)
        d.text((2, h + 2), text[:16], font=font, fill=(200, 0, 0))
        d.text((2, h + 26), text[16:32], font=font, fill=(200, 0, 0))
        cells.append(np.asarray(cell)[..., ::-1])
    return grid(cells, h + 50, cols=6)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/synth"))
    ap.add_argument("--n", type=int, default=200, help="pages to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fonts", type=Path, nargs="*", default=None,
                    help="font files or folders (default: common system font dirs)")
    ap.add_argument("--corpus", type=Path, default=None, help="UTF-8 file, one line per text")
    ap.add_argument("--backgrounds", type=Path, default=None,
                    help="folder of text-free page images (e.g. inpainted pages)")
    ap.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB,
                    help="OCR vocab; corpus lines with characters outside it are dropped")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--vertical", type=float, default=0.75, help="share of vertical blocks")
    ap.add_argument("--preview", type=int, default=12, help="pages shown in preview.jpg")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    ap.add_argument("--val-corpus-ratio", type=float, default=0.15,
                    help="share of corpus lines reserved for validation pages")
    ap.add_argument("--emit", default="det,ocr,inpaint", help="outputs to write")
    a = ap.parse_args(argv)

    fonts = find_fonts(a.fonts if a.fonts else FONT_DIRS)
    bgs = []
    if a.backgrounds:
        bgs = sorted(p for p in a.backgrounds.rglob("*") if p.suffix.lower() in IMG_EXT)
    corpus = load_corpus(a.corpus, load_vocab(a.vocab))
    cfg = Config(vertical=a.vertical)
    meta = generate(a.out, a.n, a.seed, fonts, corpus, bgs, a.val_ratio, cfg, a.preview,
                    a.workers, a.val_corpus_ratio, frozenset(a.emit.split(",")))
    print(json.dumps(meta, indent=2, ensure_ascii=False), file=sys.stderr)
    print(f"preview: {a.out / 'preview.jpg'}  ocr: {a.out / 'ocr_preview.jpg'}")
    return meta


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
