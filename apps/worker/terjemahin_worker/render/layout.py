"""Shape-aware line breaking: fit a paragraph into an arbitrary balloon interior.

A `Shape` stores, for every pixel row, one horizontal run of the usable interior, so the width
available to a line is the intersection of the runs over the rows its glyphs cover. That makes
ovals, tails and slanted balloons work the same way as rectangles.

`fit()` binary-searches the font size; for each size it tries a few line counts and vertical
positions, and breaks lines with a small dynamic programme (minimum raggedness against each
line's own capacity, penalties for hyphens and single-word last lines) instead of greedy
filling, which keeps the text block balanced like hand lettering.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np
from PIL import ImageFont

from .hyphen import split_points

Align = Literal["center", "left", "right"]

HYPHEN_COST = 0.35
ORPHAN_COST = 0.6
OFFSET_COST = 0.15
SAFETY = 0.98  # fragment widths are summed, so leave room for kerning differences
FIT_TOL = 1.5  # px a re-measured line may exceed its band before the size is rejected

# Typographic characters a Latin comic font often lacks, with a safe ASCII spelling.
ASCII_FALLBACK = {
    "…": "...",
    "⋯": "...",
    "‥": "..",
    "“": '"',
    "”": '"',
    "„": '"',
    "‘": "'",
    "’": "'",
    "—": "-",
    "–": "-",
    "―": "-",
    "〜": "~",
    "～": "~",
    "「": '"',
    "」": '"',
    "『": '"',
    "』": '"',
    "ー": "-",
    "、": ",",
    "。": ".",
    "・": "-",
    "♪": "~",
    "♡": "~",
}


class Measure(Protocol):
    def width(self, text: str, size: int) -> float: ...

    def ink(self, size: int) -> tuple[float, float]:
        """(top, bottom) of a line's ink, relative to the line's top (ascender) anchor."""
        ...


class FontMeasure:
    """PIL FreeType metrics with per-instance caches; also the object renderers draw with."""

    def __init__(self, font_path: str | Path | None) -> None:
        p = Path(font_path) if font_path else None
        self.path = str(p) if p and p.exists() else None
        self._fonts: dict[int, ImageFont.FreeTypeFont] = {}
        self._widths: dict[tuple[str, int], float] = {}
        self._glyphs: dict[str, bool] = {}

    def font(self, size: int) -> ImageFont.FreeTypeFont:
        f = self._fonts.get(size)
        if f is None:
            if self.path:
                f = ImageFont.truetype(self.path, size)
            else:
                f = ImageFont.load_default(size=size)  # type: ignore[assignment]
            self._fonts[size] = f  # type: ignore[assignment]
        return f  # type: ignore[return-value]

    def width(self, text: str, size: int) -> float:
        key = (text, size)
        w = self._widths.get(key)
        if w is None:
            w = self._widths[key] = float(self.font(size).getlength(text))
        return w

    def ink(self, size: int) -> tuple[float, float]:
        _, top, _, bottom = self.font(size).getbbox("Hdgj")
        return float(top), float(bottom)

    def has_glyph(self, ch: str) -> bool:
        ok = self._glyphs.get(ch)
        if ok is None:
            f = self.font(32)
            ok = ch.isspace() or bytes(f.getmask(ch)) != bytes(f.getmask("\uffff"))
            self._glyphs[ch] = ok
        return ok

    def normalize(self, text: str) -> str:
        """NFKC (full-width -> ASCII) plus ASCII spellings for glyphs the font lacks.

        "..." becomes one ellipsis glyph when the font has it: in a monospaced comic font three
        dots cost three cells, which is often what limits the size in a narrow balloon.
        """
        text = unicodedata.normalize("NFKC", text)
        if self.has_glyph("\u2026"):
            text = re.sub(r"\.{3,}", "\u2026", text)
        out = []
        for ch in text:
            if not self.has_glyph(ch) and ch in ASCII_FALLBACK:
                ch = ASCII_FALLBACK[ch]
            out.append(ch)
        return " ".join("".join(out).split())


@dataclass
class Shape:
    """Usable interior: per-row [left, right) of one run; right <= left means empty.

    The run kept on each row is the one nearest the mask's deepest point (distance-transform
    maximum), so a two-lobed balloon gives one coherent column of text instead of lines that
    jump between lobes.
    """

    left: np.ndarray
    right: np.ndarray
    cx: float | None = None  # visual centre (deepest interior), None = middle of the runs
    cy: float | None = None

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> Shape:
        m = mask > 0
        h, w = m.shape
        left = np.zeros(h, np.float32)
        right = np.zeros(h, np.float32)
        padded = np.zeros((h, w + 2), np.int8)
        padded[:, 1:-1] = m
        d = np.diff(padded, axis=1)
        # pad: OpenCV treats outside-the-image as interior, which would pull the centre onto
        # a crop edge
        dist = cv2.distanceTransform(np.pad(m, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
        ridge_y, ridge_x = np.nonzero(dist >= 0.9 * dist.max())
        cx, cy = float(ridge_x.mean()) + 0.5, float(ridge_y.mean()) + 0.5
        for y in np.flatnonzero(m.any(axis=1)):
            starts = np.flatnonzero(d[y] == 1)
            ends = np.flatnonzero(d[y] == -1)
            off = np.maximum(np.maximum(starts - cx, cx - ends), 0)
            k = int(np.lexsort((-(ends - starts), off))[0])
            left[y], right[y] = starts[k], ends[k]
        return cls(left, right, cx, cy)

    @classmethod
    def rect(cls, w: float, h: int) -> Shape:
        return cls(np.zeros(h, np.float32), np.full(h, w, np.float32))

    @property
    def height(self) -> int:
        return len(self.left)

    @property
    def width(self) -> float:
        return float(self.right.max() - self.left.min()) if self.height else 0.0

    def span(self, y0: float, y1: float) -> tuple[float, float]:
        a, b = max(0, math.floor(y0)), min(self.height, math.ceil(y1))
        if y0 < 0 or y1 > self.height or b <= a:
            return 0.0, 0.0
        return float(self.left[a:b].max()), float(self.right[a:b].min())

    def vertical_center(self) -> float:
        """Deepest interior point, else the middle of rows >= 35% of the widest (no tail tip)."""
        if self.cy is not None:
            return self.cy
        widths = self.right - self.left
        rows = np.flatnonzero(widths >= 0.35 * widths.max()) if widths.max() > 0 else []
        return (rows[0] + rows[-1] + 1) / 2 if len(rows) else self.height / 2

    def axis(self) -> float:
        if self.cx is not None:
            return self.cx
        widths = self.right - self.left
        rows = widths >= 0.5 * widths.max()
        return float(np.median((self.left[rows] + self.right[rows]) / 2)) if rows.any() else 0.0

    def widen(self, factor: float) -> Shape:
        """Overflow fallback: a rectangle `factor` x the widest run and the height, same axis."""
        ax = self.axis()
        half = max(float((self.right - self.left).max()) * factor, 1.0) / 2
        h = max(1, round(self.height * factor))
        return Shape(np.full(h, ax - half, np.float32), np.full(h, ax + half, np.float32), ax)


@dataclass
class Line:
    text: str
    x: float
    y: float  # top (ascender) anchor in shape coordinates
    width: float
    join: Literal["space", "soft", "none", "end"] = "end"  # how it continues on the next line


@dataclass
class Layout:
    lines: list[Line] = field(default_factory=list)
    size: int = 0
    line_height: float = 0.0
    fits: bool = True
    hyphenated: bool = False

    @property
    def text(self) -> str:
        """Lines joined back into the paragraph (soft hyphens at breaks removed)."""
        out = ""
        for ln in self.lines:
            if ln.join == "soft":
                out += ln.text.removesuffix("-")
            elif ln.join == "space":
                out += ln.text + " "
            else:
                out += ln.text
        return out.strip()


@dataclass
class _Frag:
    text: str
    width: float
    join: Literal["space", "soft", "none", "end"]  # what follows: word gap, hyphen point, ...


def _fragments(text: str, size: int, m: Measure, hyphenate: int) -> list[_Frag]:
    """hyphenate: 0 never, 1 not names (capitalised mid-sentence), 2 any word."""
    frags: list[_Frag] = []
    sentence_start = True
    for word in text.split():
        name = word[:1].isupper() and not sentence_start
        ok = hyphenate >= 2 or (hyphenate == 1 and not name)
        hard = [p for p in word.replace("-", "-\x00").split("\x00") if p]
        ok = ok and (len(hard) == 1 or hyphenate >= 2)  # a compound already has its break
        for j, h in enumerate(hard):
            subs = split_points(h) if ok else [h]
            for k, sub in enumerate(subs):
                join = "soft" if k < len(subs) - 1 else ("none" if j < len(hard) - 1 else "space")
                frags.append(_Frag(sub, m.width(sub, size), join))  # type: ignore[arg-type]
        sentence_start = word.rstrip("\"')»").endswith((".", "!", "?", "…"))
    if frags:
        frags[-1].join = "end"
    return frags


def _char_fragments(frags: list[_Frag], size: int, m: Measure, limit: float) -> list[_Frag]:
    """Break fragments wider than `limit` into characters so nothing can ever be dropped."""
    out: list[_Frag] = []
    for f in frags:
        if f.width <= limit or len(f.text) == 1:
            out.append(f)
            continue
        for i, ch in enumerate(f.text):
            out.append(_Frag(ch, m.width(ch, size), "soft" if i < len(f.text) - 1 else f.join))
    return out


def _line_text(frags: list[_Frag], a: int, b: int) -> str:
    s = ""
    for f in frags[a:b]:
        s += f.text
        if f.join == "space":
            s += " "
    s = s.rstrip()
    if b < len(frags) and frags[b - 1].join == "soft" and not s.endswith("-"):
        s += "-"
    return s


def _break(
    frags: list[_Frag],
    caps: list[float],
    space_w: float,
    hyphen_w: float,
    ladder: bool = False,
) -> tuple[float, list[int]] | None:
    """Min-cost split of `frags` into exactly len(caps) non-empty lines, or None.

    Unless `ladder`, no line may be the middle of a word ("menye- / nang- / kan"): a word is
    split at most once.
    """
    n, m = len(caps), len(frags)
    if m < n:
        return None
    # prefix widths: text width of frags[a:b] = pw[b] - pw[a] - gap(b-1)
    pw = [0.0]
    for f in frags:
        pw.append(pw[-1] + f.width + (space_w if f.join == "space" else 0.0))

    def width(a: int, b: int) -> float:
        w = pw[b] - pw[a]
        last = frags[b - 1]
        if last.join == "space":
            w -= space_w
        if last.join == "soft":
            w += hyphen_w
        return w

    INF = float("inf")
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[-1] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n):
        cap = caps[i] * SAFETY
        if cap <= 0:
            return None
        last_line = i == n - 1
        for a in range(i, m):
            base = dp[i][a]
            if base == INF:
                continue
            for b in [m] if last_line else range(a + 1, m - (n - i - 1) + 1):
                w = width(a, b)
                if w > cap:
                    break
                fill = w / cap
                cost = (1.0 - fill) ** 2
                if frags[b - 1].join == "soft" and b < m:
                    if a > 0 and frags[a - 1].join == "soft":
                        if not ladder and all(f.join == "soft" for f in frags[a:b]):
                            continue
                        cost += HYPHEN_COST  # hyphen ladders read badly
                    cost += HYPHEN_COST
                if last_line and n > 1 and all(f.join != "space" for f in frags[a : b - 1]):
                    cost += ORPHAN_COST * (1.0 - fill)
                if base + cost < dp[i + 1][b]:
                    dp[i + 1][b] = base + cost
                    back[i + 1][b] = a
    if dp[n][m] == INF:
        return None
    cuts, b = [], m
    for i in range(n, 0, -1):
        cuts.append(b)
        b = back[i][b]
    return dp[n][m], cuts[::-1]


@dataclass
class _Try:
    cost: float
    frags: list[_Frag]
    cuts: list[int]
    top: float
    pitch: float


def _band(shape: Shape, y0: float, y1: float, pad: float) -> tuple[float, float]:
    """Span of rows [y0, y1] kept `pad` px clear of the interior's edge on every side."""
    lo, hi = shape.span(y0 - pad, y1 + pad)
    return lo + pad, hi - pad


def _try_size(
    text: str,
    shape: Shape,
    size: int,
    m: Measure,
    line_spacing: float,
    hyphenate: int,
    force_chars: bool = False,
    pad: float = 0.0,
) -> _Try | None:
    frags = _fragments(text, size, m, hyphenate)
    if not frags:
        return None
    if force_chars:
        frags = _char_fragments(frags, size, m, shape.width * SAFETY)
    space_w, hyphen_w = m.width(" ", size), m.width("-", size)
    ink_top, ink_bottom = m.ink(size)
    pitch = size * line_spacing
    total = sum(f.width for f in frags) + space_w * sum(f.join == "space" for f in frags)
    widest = max(shape.right - shape.left)
    if widest <= 0:
        return None
    n_min = max(1, math.ceil(total / widest))
    n_max = int((shape.height - (ink_bottom - ink_top)) // pitch) + 1
    cy = shape.vertical_center()
    best: _Try | None = None
    for n in range(n_min, min(n_max, n_min + 4, len(frags)) + 1):
        block = (n - 1) * pitch + (ink_bottom - ink_top)
        # slide the block away from the centre in half-line steps, nearest first; an offset
        # costs OFFSET_COST per line height, so stop once that alone exceeds the best cost
        reach = int(shape.height / pitch * 2) + 1
        for step in sorted(range(-reach, reach + 1), key=abs):
            k = step / 2
            if best is not None and OFFSET_COST * abs(k) >= best.cost:
                break
            top = cy - block / 2 + k * pitch - ink_top
            caps = []
            for i in range(n):
                y = top + i * pitch
                lo, hi = _band(shape, y + ink_top, y + ink_bottom, pad)
                caps.append(hi - lo)
            if min(caps) <= 0 or sum(caps) * SAFETY < total:
                continue
            res = _break(frags, caps, space_w, hyphen_w, ladder=force_chars or hyphenate >= 2)
            if res is None:
                continue
            cost = res[0] / n + OFFSET_COST * abs(k)
            if best is None or cost < best.cost:
                best = _Try(cost, frags, res[1], top, pitch)
    return best


def _place(
    t: _Try, shape: Shape, size: int, m: Measure, align: Align, pad: float = 0.0
) -> list[Line]:
    """Position lines on one common axis (or edge) when every band allows it, else clamp each
    line into its own band so it never leaves the shape."""
    ink_top, ink_bottom = m.ink(size)
    axis = shape.axis()
    rows, a = [], 0
    for i, b in enumerate(t.cuts):
        text = _line_text(t.frags, a, b)
        y = t.top + i * t.pitch
        w = m.width(text, size)
        lo, hi = _band(shape, y + ink_top, y + ink_bottom, pad)
        if hi <= lo:
            lo, hi = axis - w / 2, axis + w / 2
        rows.append((text, y, w, lo, hi, t.frags[b - 1].join))
        a = b
    # the anchor is the line centre / left edge / right edge, depending on alignment
    off = {"center": 0.5, "left": 0.0, "right": 1.0}[align]
    lo_a = max(lo + w * off for _, _, w, lo, _, _ in rows)
    hi_a = min(hi - w * (1 - off) for _, _, w, _, hi, _ in rows)
    want = {"center": axis, "left": lo_a, "right": hi_a}[align]
    common = min(max(want, lo_a), hi_a) if lo_a <= hi_a else None
    lines = []
    for text, y, w, lo, hi, join in rows:
        anchor = common if common is not None else min(max(want, lo + w * off), hi - w * (1 - off))
        x = min(max(anchor - w * off, lo), max(lo, hi - w))
        lines.append(Line(text, x, y, w, join))
    return lines


def _fits_measured(
    lines: list[Line], shape: Shape, size: int, m: Measure, pad: float = 0.0
) -> bool:
    ink_top, ink_bottom = m.ink(size)
    for ln in lines:
        lo, hi = _band(shape, ln.y + ink_top, ln.y + ink_bottom, pad)
        if ln.x < lo - FIT_TOL or ln.x + ln.width > hi + FIT_TOL:
            return False
    return True


def _orphan(lines: list[Line]) -> bool:
    return len(lines) > 1 and " " not in lines[-1].text and lines[-1].width < 0.4 * lines[-2].width


def _search(
    text: str,
    shape: Shape,
    m: Measure,
    lo: int,
    hi: int,
    line_spacing: float,
    hyphenate: int,
    padding: float,
) -> tuple[int, _Try] | None:
    best: tuple[int, _Try] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        pad = padding * mid
        t = _try_size(text, shape, mid, m, line_spacing, hyphenate, pad=pad)
        lines = _place(t, shape, mid, m, "center", pad) if t is not None else []
        if t is not None and not _fits_measured(lines, shape, mid, m, pad):
            t = None
        if t is not None:
            best, lo = (mid, t), mid + 1
        else:
            hi = mid - 1
    return best


def fit(
    text: str,
    shape: Shape,
    measure: Measure,
    *,
    min_size: int = 10,
    max_size: int = 64,
    line_spacing: float = 1.15,
    hyphenate: bool = True,
    hyphen_gain: float = 1.15,
    align: Align = "center",
    padding: float = 0.1,
) -> Layout:
    """Largest size (within [min_size, max_size]) at which `text` fits inside `shape`.

    Hyphenation is used only when it buys at least `hyphen_gain` x font size, or when the text
    would not fit at all otherwise; capitalised words (names) are only split in that last case.
    `padding` (x font size) keeps glyphs clear of the interior edge. If even `min_size`
    overflows, the shape is widened step by step (and over-long words broken by character) so
    the result always contains every word; `fits` is then False.
    """
    text = " ".join(text.split())
    if not text or shape.height == 0:
        return Layout(size=min_size, fits=not text)
    max_size = max(min_size, max_size)
    plain = _search(text, shape, measure, min_size, max_size, line_spacing, 0, padding)
    hyph, level = None, 0
    if hyphenate:
        floor = min_size if plain is None else plain[0] + 1
        hyph, level = _search(text, shape, measure, floor, max_size, line_spacing, 1, padding), 1
        if plain is None and hyph is None:
            hyph, level = (
                _search(text, shape, measure, min_size, max_size, line_spacing, 2, padding),
                2,
            )
    choice = plain
    if hyph is not None and (plain is None or hyph[0] >= plain[0] * hyphen_gain):
        choice = hyph
    else:
        level = 0
    if choice is not None:
        size, t = choice
        lines = _place(t, shape, size, measure, align, padding * size)
        if _orphan(lines):
            # one short word alone on the last line: accept up to ~8% smaller text to avoid it
            for s2 in range(size - 1, max(min_size, int(size * 0.92)) - 1, -1):
                pad = padding * s2
                t2 = _try_size(text, shape, s2, measure, line_spacing, level, pad=pad)
                if t2 is None:
                    continue
                l2 = _place(t2, shape, s2, measure, align, pad)
                if not _orphan(l2) and _fits_measured(l2, shape, s2, measure, pad):
                    size, t, lines = s2, t2, l2
                    break
        return Layout(lines, size, t.pitch, True, any(ln.join == "soft" for ln in lines))

    size, factor = min_size, 1.0
    for _ in range(64):
        factor *= 1.15
        grow = shape.widen(factor)
        t = _try_size(text, grow, size, measure, line_spacing, 2 * hyphenate, force_chars=True)
        if t is not None:
            lines = _place(t, grow, size, measure, align)
            for ln in lines:
                ln.y += (shape.height - grow.height) / 2
            return Layout(lines, size, t.pitch, False, any(ln.join == "soft" for ln in lines))
    raise RuntimeError(f"could not lay out {text!r}")  # unreachable: the box grows ~7500x
