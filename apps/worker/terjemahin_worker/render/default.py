"""Balloon-aware typesetter: the default renderer.

Per bubble: find the balloon in the page (balloon.py), erode it into a usable interior, fit the
translation into that shape (layout.py: font-size search, shape-constrained line widths,
Indonesian hyphenation, no orphans), then draw it centred with colours taken from OCR or the
balloon. Text without a closed balloon (over art) gets its detector box, widened for horizontal
text, and an outline so it stays legible. Font sizes are normalised across the page so a short
"Eh?" does not shout next to a long sentence.
"""

from __future__ import annotations

import inspect
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..models import RGB, BBox, Bubble
from ..opts import as_bool, as_color, as_float, as_int, optional
from ..registry import RENDERERS
from .balloon import Balloon, find_balloon, split_shared
from .layout import Align, FontMeasure, Layout, Shape, fit

log = logging.getLogger(__name__)

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯ｦ-ﾟ]")
LOW_CONTRAST = 90  # luminance difference below which the text needs an outline
AUTO_STROKE = 0.14  # outline width as a fraction of the font size, when one is needed
MAX_SCALE = 1.0  # max Latin font size relative to the source glyph size
MARGIN = 0.08  # interior inset as a fraction of the balloon's smaller side
PADDING = 0.1  # clearance between glyphs and the interior edge, x font size
OVERFLOW_SHRINK = 0.65  # how far below the page minimum a size may go before text overflows
PAGE_ASPECT = 1.6  # long/short side of a page; a taller strip (webtoon) is many pages


def page_min_size(W: int, H: int) -> int:
    """Smallest font size the page should use: about 1/200 of width + height, with the long
    side counted at most as a page's (an 800x12000 webtoon strip asked for 64 px, more than
    any of its balloons could hold, so every fit collapsed to one size and overflowed)."""
    short, long = min(W, H), max(W, H)
    return max(10, round((short + min(long, PAGE_ASPECT * short)) / 200))


def luminance(c: RGB) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def is_cjk(text: str) -> bool:
    letters = [c for c in text if not c.isspace()]
    return bool(letters) and sum(bool(CJK.match(c)) for c in letters) >= 0.5 * len(letters)


@dataclass
class Plan:
    bubble: Bubble
    shape: Shape
    ox: float  # page position of the shape frame
    oy: float
    closed: bool
    local_bg: RGB
    max_size: int
    vertical: bool
    text: str
    layout: Any = None  # Layout | VLayout
    measure: Any = None  # FontMeasure the text is set in (the CJK font for CJK text)
    angle: float = 0.0  # text tilt (CCW degrees); the shape frame is then the de-rotated interior
    frame: tuple[int, int] = (0, 0)  # (w, h) of the shape frame, for tilted drawing


class DefaultRenderer:
    def __init__(
        self,
        font_path: str | Path | None,
        *,
        cjk_font_path: str | Path | None = None,
        font_size_offset: int = 0,
        min_font_size: int | None = None,
        max_font_size: int | None = None,
        uppercase: bool = False,
        alignment: Align = "center",
        line_spacing: float = 1.15,
        stroke_width: float | None = None,
        hyphenate: bool = True,
        normalize: bool = True,
        normalize_tol: float = 0.15,
        direction: Literal["auto", "h", "v"] = "auto",
        fg_color: RGB | None = None,
        debug: bool = False,
    ) -> None:
        if alignment not in ("center", "left", "right"):
            raise ValueError(f"alignment must be center, left or right, not {alignment!r}")
        if direction not in ("auto", "h", "v"):
            raise ValueError(f"direction must be auto, h or v, not {direction!r}")
        for name, path in (("font_path", font_path), ("cjk_font_path", cjk_font_path)):
            if path and not Path(path).is_file():
                raise ValueError(f"renderer {name}: no such font file: {path}")
        self.measure = FontMeasure(font_path)
        self.cjk_font_path = cjk_font_path
        self._cjk: Any = None
        self.font_size_offset = as_int(font_size_offset, "renderer font_size_offset", -200, 200)
        self.min_font_size = optional(as_int, min_font_size, "renderer min_font_size", 1, 1000)
        self.max_font_size = optional(as_int, max_font_size, "renderer max_font_size", 1, 1000)
        if self.min_font_size and self.max_font_size and self.min_font_size > self.max_font_size:
            raise ValueError(
                f"renderer min_font_size ({self.min_font_size}) is larger than max_font_size "
                f"({self.max_font_size})"
            )
        self.uppercase = as_bool(uppercase, "renderer uppercase")
        self.alignment = alignment
        self.line_spacing = as_float(line_spacing, "renderer line_spacing", 0.5, 5.0)
        self.stroke_width = optional(as_float, stroke_width, "renderer stroke_width", 0.0, 1.0)
        self.hyphenate = as_bool(hyphenate, "renderer hyphenate")
        self.normalize = as_bool(normalize, "renderer normalize")
        self.normalize_tol = as_float(normalize_tol, "renderer normalize_tol", 0.0)
        self.direction = direction
        self.fg_color = optional(as_color, fg_color, "renderer fg_color") if fg_color else None
        self.debug = as_bool(debug, "renderer debug")
        self.plans: list[Plan] = []  # last page, for tests and debugging

    # -- planning ---------------------------------------------------------------------------

    def _cjk_measure(self) -> FontMeasure:
        if self._cjk is None:
            from .vertical import find_cjk_font

            path = self.cjk_font_path or find_cjk_font()
            if not path:
                log.warning(
                    "no CJK font found (set renderer font_path); vertical text will be tofu"
                )
            self._cjk = FontMeasure(path or self.measure.path)
        return self._cjk

    def _shape(self, balloon: Balloon, bbox: BBox, W: int, H: int, vertical: bool, angle: float):
        """Usable interior as a Shape in its own frame: (shape, page x, page y, frame w, h).

        For tilted text the interior is rotated upright first, so lines are laid out along the
        text direction and the drawn block is rotated back.
        """
        if balloon.closed:
            m = balloon.mask
            bb = balloon.bbox
            inset = max(3, round(MARGIN * min(bb.w, bb.h)))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1))
            inner = cv2.erode(m, k, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            if inner.any():
                ys, xs = np.nonzero(inner)
                y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                inner = inner[y1:y2, x1:x2]
                ox, oy = balloon.x0 + x1, balloon.y0 + y1
                if vertical:
                    return Shape.from_mask(inner.T), ox, oy, inner.shape[0], inner.shape[1]
                if angle:
                    h, w = inner.shape
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
                    c, s = abs(M[0, 0]), abs(M[0, 1])
                    nw, nh = int(h * s + w * c) + 1, int(h * c + w * s) + 1
                    M[0, 2] += nw / 2 - w / 2
                    M[1, 2] += nh / 2 - h / 2
                    inner = cv2.warpAffine(inner, M, (nw, nh), flags=cv2.INTER_NEAREST)
                    ox, oy = ox + w / 2 - nw / 2, oy + h / 2 - nh / 2
                h, w = inner.shape
                return Shape.from_mask(inner), ox, oy, w, h
        b = bbox
        if not vertical and b.w < 0.75 * b.h:  # vertical source box, horizontal target text
            grow = int(0.75 * b.h - b.w) // 2
            b = BBox(x1=b.x1 - grow, y1=b.y1, x2=b.x2 + grow, y2=b.y2)
        b = b.clamp(W, H)
        w, h = (b.h, b.w) if vertical else (b.w, b.h)
        return Shape.rect(float(w), int(h)), b.x1, b.y1, b.w, b.h

    def _plan(
        self, inpainted: np.ndarray, original: np.ndarray, bubbles: list[Bubble]
    ) -> list[Plan]:
        H, W = inpainted.shape[:2]
        balloons = [find_balloon(inpainted, original, b.region.bbox) for b in bubbles]
        split_shared(balloons, [b.region.bbox for b in bubbles])
        plans = []
        for b, bal in zip(bubbles, balloons):
            text = b.target_text.strip()
            vertical = self.direction == "v" or (
                self.direction == "auto" and b.region.direction == "v" and is_cjk(text)
            )
            if vertical and self.direction == "v" and text and not is_cjk(text):
                log.warning("bubble %s: vertical layout of non-CJK text %r stacks single "
                            "letters (use the default renderer, direction=auto)", b.region.id, text)
            m = self._cjk_measure() if vertical else self.measure
            if not vertical and CJK.search(text) and not all(map(m.has_glyph, CJK.findall(text))):
                m = self._cjk_measure()  # Chinese/Korean set horizontally: not in tofu boxes
            text = m.normalize(text) if not vertical else " ".join(text.split())
            if self.uppercase and not vertical:
                text = text.upper()
            a = b.region.angle
            angle = a if 3 < abs(a) < 60 and not vertical else 0.0
            shape, ox, oy, fw, fh = self._shape(bal, b.region.bbox, W, H, vertical, angle)
            src = b.region.font_size
            if src and not re.search(r"[^\W\d_]", text):
                # '...' / '!!': a box of punctuation says nothing about glyph size, but its ink
                # does: drawn as large as the source marks, not at a third of their size
                if not vertical:  # three dots, not one narrow '…' glyph, span a '……' balloon
                    text = text.replace("\u2026", "...")
                src = None if vertical else _punct_size(m, text, b)
            short_side = min(shape.height, shape.width)
            max_size = self.max_font_size or (
                round(src * MAX_SCALE) if src else max(12, round(short_side * 0.3))
            )
            bgr = bal.color if bal.closed else _median(inpainted, b.region.bbox)
            plans.append(
                Plan(
                    b,
                    shape,
                    ox,
                    oy,
                    bal.closed,
                    (bgr[2], bgr[1], bgr[0]),
                    max(1, max_size + max(0, self.font_size_offset)),
                    vertical,
                    text,
                    angle=angle,
                    frame=(int(fw), int(fh)),
                    measure=m,
                )
            )
        return plans

    def _fit(self, p: Plan, min_size: int, max_size: int) -> Any:
        if p.vertical:
            from .vertical import fit_vertical

            return fit_vertical(
                p.text,
                p.shape,
                self._cjk_measure(),
                min_size=min_size,
                max_size=max_size,
                column_spacing=self.line_spacing - 1.0 + 0.1,
            )
        return fit(
            p.text,
            p.shape,
            p.measure or self.measure,
            min_size=min_size,
            max_size=max_size,
            line_spacing=self.line_spacing,
            hyphenate=self.hyphenate,
            align=self.alignment,
            padding=PADDING + self.colors(p, 100)[2] / 100,  # keep any outline clear too
        )

    # -- rendering --------------------------------------------------------------------------

    def render(self, inpainted_bgr, bubbles: list[Bubble], original_bgr) -> np.ndarray:
        inpainted_bgr, original_bgr = _bgr(inpainted_bgr), _bgr(original_bgr)
        H, W = inpainted_bgr.shape[:2]
        if original_bgr.shape[:2] != (H, W):
            original_bgr = cv2.resize(original_bgr, (W, H), interpolation=cv2.INTER_AREA)
        bubbles = [b for b in (_on_page(b, W, H) for b in bubbles if b.target_text.strip()) if b]
        if not bubbles:
            return inpainted_bgr.copy()
        min_size = self.min_font_size or page_min_size(W, H)
        plans = self._plan(inpainted_bgr, original_bgr, bubbles)
        for p in plans:
            lo = min(min_size, p.max_size)
            p.layout = self._fit(p, lo, p.max_size)
            shrink = max(8, round(min_size * OVERFLOW_SHRINK))
            if shrink >= lo:  # a balloon smaller than the page minimum: shrink from its own
                shrink = max(8, round(lo * OVERFLOW_SHRINK))
            if not p.layout.fits and shrink < lo:  # rather smaller than spilling out of the balloon
                smaller = self._fit(p, shrink, p.max_size)
                if smaller.fits:
                    p.layout = smaller
            if self.font_size_offset < 0 and p.layout.fits:
                # never below the page minimum, nor below a size the overflow shrink chose
                floor = min(min_size, p.layout.size)
                target = max(floor, p.layout.size + self.font_size_offset)
                smaller = self._fit(p, target, target)
                if smaller.fits:
                    p.layout = smaller

        if self.normalize and len(plans) >= 3:
            sizes = sorted(p.layout.size for p in plans if p.layout.fits)
            if sizes:
                cap = max(min_size, round(float(np.median(sizes)) * (1 + self.normalize_tol)))
                for p in plans:
                    if p.layout.size > cap:
                        p.layout = self._fit(p, min(min_size, cap), cap)

        pil = Image.fromarray(cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB))
        for p in plans:
            self._draw(pil, p)
            if not p.layout.fits:
                log.warning("text overflows its balloon at %dpx: %r", p.layout.size, p.text)
        self.plans = plans
        out = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        if self.debug:
            out = self._debug_overlay(out, plans)
        return out

    def colors(self, p: Plan, size: int) -> tuple[RGB, RGB, int]:
        """(fill, stroke colour, stroke px)."""
        r = p.bubble.region
        fg = self.fg_color or r.fg_color
        if fg is None:
            fg = (0, 0, 0) if luminance(p.local_bg) > 128 else (255, 255, 255)
        opposite = (255, 255, 255) if luminance(fg) < 128 else (0, 0, 0)
        sc = (
            r.bg_color
            if r.bg_color and abs(luminance(r.bg_color) - luminance(fg)) >= LOW_CONTRAST
            else opposite
        )
        if self.stroke_width is not None:
            sw = round(self.stroke_width * size)
        else:
            weak = abs(luminance(fg) - luminance(p.local_bg)) < LOW_CONTRAST
            sw = max(2, round(AUTO_STROKE * size)) if (not p.closed or weak) else 0
        return tuple(fg), tuple(sc), sw  # type: ignore[return-value]

    def _draw(self, pil: Image.Image, p: Plan) -> None:
        lay = p.layout
        fg, sc, sw = self.colors(p, lay.size)
        if p.vertical:
            from .vertical import draw_vertical

            draw_vertical(pil, lay, self._cjk_measure(), p.ox, p.oy, fg, sc, sw)
            return
        font = (p.measure or self.measure).font(lay.size)
        if not p.angle:
            draw = ImageDraw.Draw(pil)
            for ln in lay.lines:
                draw.text(
                    (p.ox + ln.x, p.oy + ln.y),
                    ln.text,
                    font=font,
                    fill=fg,
                    anchor="la",
                    stroke_width=sw,
                    stroke_fill=sc,
                )
            return
        # tilted text: draw upright in the de-rotated frame, rotate it back about its centre
        fw, fh = p.frame
        layer = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for ln in lay.lines:
            d.text(
                (ln.x, ln.y),
                ln.text,
                font=font,
                fill=(*fg, 255),
                anchor="la",
                stroke_width=sw,
                stroke_fill=(*sc, 255),
            )
        rot = layer.rotate(p.angle, resample=Image.BICUBIC, expand=True)
        cx, cy = p.ox + fw / 2, p.oy + fh / 2
        pil.paste(rot, (round(cx - rot.width / 2), round(cy - rot.height / 2)), rot)

    def _debug_overlay(self, out: np.ndarray, plans: list[Plan]) -> np.ndarray:
        out = out.copy()
        for p in plans:
            if p.angle:
                continue  # the shape lives in a rotated frame
            col = (0, 180, 0) if p.closed else (0, 0, 255)
            for y in range(0, p.shape.height, 2):
                a, b = p.shape.left[y], p.shape.right[y]
                if b > a:
                    if p.vertical:
                        pts = [(int(p.ox + y), int(p.oy + a)), (int(p.ox + y), int(p.oy + b - 1))]
                    else:
                        pts = [(int(p.ox + a), int(p.oy + y)), (int(p.ox + b - 1), int(p.oy + y))]
                    for q in pts:
                        cv2.circle(out, q, 1, col, -1)
        return out


def _bgr(img: np.ndarray) -> np.ndarray:
    """Pages may arrive grayscale or with alpha; the renderer works on 3-channel BGR."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _on_page(b: Bubble, W: int, H: int) -> Bubble | None:
    """Bubble with its box clipped to the page and at least 1 px wide and high; None when the
    box lies entirely off the page (hand-edited PageResult JSON is not validated upstream)."""
    r = b.region.bbox
    x1, y1, x2, y2 = max(0, r.x1), max(0, r.y1), min(W, r.x2), min(H, r.y2)
    if x1 >= W or y1 >= H or x2 <= 0 or y2 <= 0 or (x2 < x1) or (y2 < y1):
        log.warning("bubble %s: box %s is off the page, not rendered", b.region.id, r)
        return None
    x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
    if (x1, y1, x2, y2) != (r.x1, r.y1, r.x2, r.y2):
        box = BBox(x1=min(x1, W - 1), y1=min(y1, H - 1), x2=x2, y2=y2)
        b = b.model_copy(update={"region": b.region.model_copy(update={"bbox": box})})
    return b


def _punct_size(m: Any, text: str, b: Bubble) -> int | None:
    """Font size at which punctuation-only `text` ('?', '...', '!!') has the ink extent of the
    source marks. A horizontal (or one-glyph) source is matched on both axes; a vertical stack
    ('！！' in one column) on the height of one of its glyphs, since `text` is set across."""
    l, t, r, btm = m.font(100).getbbox(text)
    ink_w, ink_h = max(1, r - l), max(1, btm - t)
    bb = b.region.bbox
    src_w, src_h = max(1, bb.x2 - bb.x1), max(1, bb.y2 - bb.y1)
    n = len("".join(unicodedata.normalize("NFKC", b.source_text).split())) or 1
    if b.region.direction == "v" and n > 1:
        size = 100 * (src_h / n) / ink_h
    else:
        size = 100 * min(src_w / ink_w, src_h / ink_h)
    return max(12, round(size)) if size > 0 else None


def _median(img: np.ndarray, b: BBox) -> tuple[int, int, int]:
    H, W = img.shape[:2]
    c = b.clamp(W, H)
    px = img[c.y1 : c.y2, c.x1 : c.x2].reshape(-1, 3)
    med = np.median(px, axis=0) if len(px) else np.array([255, 255, 255])
    return int(med[0]), int(med[1]), int(med[2])


def build(settings, **opts: Any) -> DefaultRenderer:
    font = opts.pop("font_path", None) or settings.font_path
    opts.setdefault("cjk_font_path", getattr(settings, "cjk_font_path", None))
    known = inspect.signature(DefaultRenderer).parameters
    unknown = [k for k in opts if k not in known]
    if unknown:
        raise ValueError(f"unknown renderer option(s): {', '.join(unknown)}")
    return DefaultRenderer(font, **opts)


@RENDERERS.register(
    "default",
    description="Balloon-aware typesetting: finds each balloon's interior, fits the text to "
    "its shape (font-size search, Indonesian hyphenation, no orphans), page-consistent sizes, "
    "outline over art; vertical columns for CJK targets.",
)
def _build(settings, **opts):
    return build(settings, **opts)


__all__ = ["DefaultRenderer", "Layout", "Plan", "build"]
