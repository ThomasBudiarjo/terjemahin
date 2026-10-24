"""Vertical (tategaki) typesetting for CJK targets: top-to-bottom columns, right to left.

Punctuation and brackets switch to their Unicode vertical presentation forms when the font has
them; the long-vowel mark and wave dashes have none and are drawn rotated 90 degrees. Kinsoku
shori keeps closing punctuation and small kana from starting a column (they hang at the end of
the previous one instead). The shape handed in is transposed (rows are page columns), so the
same balloon interior code serves both directions.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ..config import WORKER_ROOT
from ..registry import RENDERERS
from .layout import FontMeasure, Shape

H2V = {
    "、": "︑",
    "。": "︒",
    ",": "︐",
    "，": "︐",
    ".": "︒",
    ":": "︓",
    "：": "︓",
    ";": "︔",
    "；": "︔",
    "!": "︕",
    "！": "︕",
    "?": "︖",
    "？": "︖",
    "…": "︙",
    "‥": "︰",
    "—": "︱",
    "―": "︱",
    "–": "︲",
    "(": "︵",
    ")": "︶",
    "（": "︵",
    "）": "︶",
    "{": "︷",
    "}": "︸",
    "〔": "︹",
    "〕": "︺",
    "【": "︻",
    "】": "︼",
    "《": "︽",
    "》": "︾",
    "〈": "︿",
    "〉": "﹀",
    "「": "﹁",
    "」": "﹂",
    "『": "﹃",
    "』": "﹄",
    "[": "﹇",
    "]": "﹈",
    "＿": "︳",
}
ROTATE = set("ー〜～~-－=＝→←")
NO_START = set(
    "、。，．,.!?！？)）」』】〉》〕ーっゃゅょぁぃぅぇぉゎッャュョァィゥェォヮヵヶ…‥・：；"
) | {H2V[c] for c in "、。，.!?！？)）」』】〉》〕…‥：；" if c in H2V}
SMALL_KANA = set("っゃゅょぁぃぅぇぉゎッャュョァィゥェォヮヵヶ")
# horizontal-form commas/stops sit bottom-left of the em box; vertically they belong top-right
CORNER_PUNCT = set("、。，．,.")


def find_cjk_font() -> str | None:
    """A font with Japanese glyphs: worker fonts/ first, then fontconfig."""
    fonts = WORKER_ROOT / "fonts"
    for pat in ("*CJK*", "*Noto*JP*", "*gothic*", "*Gothic*", "*mincho*", "*.ttc"):
        for p in sorted(fonts.glob(pat)):
            return str(p)
    if shutil.which("fc-match"):
        try:
            out = subprocess.run(
                ["fc-match", "-f", "%{file}", ":lang=ja"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out = ""
        if out and Path(out).exists() and _covers_kana(out):
            return out
    return None


@lru_cache(maxsize=8)
def _covers_kana(path: str) -> bool:
    return FontMeasure(path).has_glyph("あ")


@dataclass
class Cell:
    ch: str
    x: float  # cell centre, shape-frame coords (x along the page, y down the column)
    y: float
    rotate: bool = False


@dataclass
class VLayout:
    cells: list[Cell] = field(default_factory=list)
    size: int = 0
    columns: list[str] = field(default_factory=list)
    fits: bool = True

    @property
    def text(self) -> str:
        return "".join(self.columns)


def prepare(text: str, m: FontMeasure) -> list[tuple[str, bool]]:
    """(glyph, rotate) per character, with vertical forms substituted where the font has them."""
    text = "".join(text.split()).replace("...", "…").replace("..", "‥")
    out = []
    for ch in text:
        v = H2V.get(ch)
        if v and m.has_glyph(v):
            out.append((v, False))
        else:
            out.append((ch, ch in ROTATE))
    return out


def _columns(chars: list[tuple[str, bool]], caps: list[int]) -> list[list[tuple[str, bool]]] | None:
    """Fill columns in proportion to their capacity (balanced, not greedy), with kinsoku."""
    cols: list[list[tuple[str, bool]]] = []
    i = 0
    for j, cap in enumerate(caps):
        if i >= len(chars):
            break
        if cap <= 0:
            return None
        rest = len(chars) - i
        share = math.ceil(rest * cap / max(1, sum(caps[j:])))
        take = min(cap, rest, max(1, share))
        col = chars[i : i + take]
        i += take
        while i < len(chars) and chars[i][0] in NO_START:  # hang closing punctuation
            col.append(chars[i])
            i += 1
        cols.append(col)
    return cols if i >= len(chars) else None


def _try(chars, shape: Shape, size: int, spacing: float, allow_hang: int = 1):
    pitch = size * (1 + spacing)
    longest = max(shape.right - shape.left)
    if longest < size:
        return None
    n_min = max(1, math.ceil(len(chars) / max(1, int(longest // size))))
    for n in range(n_min, n_min + 4):
        block = n * size + (n - 1) * (pitch - size)
        if block > shape.height:
            return None
        cx = shape.vertical_center()
        right = cx + block / 2
        centres = [right - size / 2 - i * pitch for i in range(n)]
        spans = [shape.span(c - size / 2, c + size / 2) for c in centres]
        caps = [int((hi - lo) // size) for lo, hi in spans]
        cols = _columns(chars, caps)
        if cols is None or len(cols) < n and n > n_min:
            continue
        if any(len(c) > cap + allow_hang for c, cap in zip(cols, caps)):
            continue
        return cols, centres, spans
    return None


def fit_vertical(
    text: str,
    shape: Shape,
    measure: FontMeasure,
    *,
    min_size: int = 10,
    max_size: int = 64,
    column_spacing: float = 0.25,
) -> VLayout:
    """Largest size whose columns fit `shape` (transposed: rows = page x, runs = y ranges)."""
    chars = prepare(text, measure)
    if not chars or shape.height == 0:
        return VLayout(size=min_size, fits=not chars)
    lo, hi, best = min_size, max(min_size, max_size), None
    while lo <= hi:
        mid = (lo + hi) // 2
        r = _try(chars, shape, mid, column_spacing)
        if r:
            best, lo = (mid, r), mid + 1
        else:
            hi = mid - 1
    fits = best is not None
    if best is None:  # overflow: keep the minimum size and let the block outgrow the shape
        longest = float(max(shape.right - shape.left))
        for n in range(1, 10_000):
            wide = Shape.rect(longest + 2 * n * min_size, shape.height + 2 * n * min_size)
            r = _try(chars, wide, min_size, column_spacing, allow_hang=len(chars))
            if r:
                cols, centres, spans = r
                dx, dy = (
                    (shape.height - wide.height) / 2,
                    shape.axis() - (longest / 2 + n * min_size),
                )
                best = (
                    min_size,
                    (cols, [c + dx for c in centres], [(a + dy, b + dy) for a, b in spans]),
                )
                break
        assert best is not None
    size, (cols, centres, spans) = best
    cells: list[Cell] = []
    for col, cx, (top, bottom) in zip(cols, centres, spans):
        y0 = (top + bottom) / 2 - len(col) * size / 2
        for k, (ch, rot) in enumerate(col):
            cells.append(Cell(ch, cx, y0 + (k + 0.5) * size, rot))
    return VLayout(cells, size, ["".join(c for c, _ in col) for col in cols], fits)


def draw_vertical(
    pil: Image.Image,
    lay: VLayout,
    measure: FontMeasure,
    ox: float,
    oy: float,
    fill: tuple[int, int, int],
    stroke: tuple[int, int, int],
    stroke_width: int,
) -> None:
    font = measure.font(lay.size)
    draw = ImageDraw.Draw(pil)
    for c in lay.cells:
        x, y = ox + c.x, oy + c.y
        if c.ch in SMALL_KANA:
            x, y = x + lay.size * 0.08, y - lay.size * 0.08
        elif c.ch in CORNER_PUNCT:
            x, y = x + lay.size * 0.55, y - lay.size * 0.6
        if not c.rotate:
            draw.text(
                (x, y),
                c.ch,
                font=font,
                fill=fill,
                anchor="mm",
                stroke_width=stroke_width,
                stroke_fill=stroke,
            )
            continue
        s = lay.size * 2
        tile = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text(
            (s / 2, s / 2),
            c.ch,
            font=font,
            fill=(*fill, 255),
            anchor="mm",
            stroke_width=stroke_width,
            stroke_fill=(*stroke, 255),
        )
        tile = tile.rotate(-90, resample=Image.BICUBIC)
        pil.paste(tile, (round(x - s / 2), round(y - s / 2)), tile)


@RENDERERS.register(
    "vertical",
    description="Vertical right-to-left columns for CJK targets (ja/zh/ko), balloon-aware, "
    "vertical punctuation forms, kinsoku; needs a CJK font (font_path or fonts/*CJK*).",
)
def _build(settings, **opts: Any):
    from .default import build

    opts.setdefault("direction", "v")
    if opts.get("font_path") and not opts.get("cjk_font_path"):
        opts["cjk_font_path"] = opts["font_path"]
    return build(settings, **opts)
