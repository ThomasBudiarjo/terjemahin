"""Glyph-level erase masks.

The goal is the smallest mask an inpainter needs: every glyph pixel (anti-aliased fringe,
furigana and punctuation included) and nothing else, so balloon outlines and artwork next to the
text survive. Ideas follow manga-image-translator's `complete_mask` (connected components of the
detector's text mask assigned to text lines, then dilation by font size), with the dense-CRF step
replaced by a cheap colour test against the local background, and an explicit guard that keeps
long ink structures (balloon outlines, art lines) out of the mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..models import TextRegion
from ..registry import MASK_REFINERS


@dataclass
class Crop:
    """One region's working window: page slice plus where the text can be."""

    x1: int
    y1: int
    x2: int
    y2: int
    font: int
    lines: np.ndarray  # bool, the text line quads (or the box when there are none)
    near: np.ndarray  # bool, pixels close enough to a text line to belong to it
    box: tuple[int, int, int, int]  # region bbox in crop coords
    cut: np.ndarray  # bool, crop border pixels that are not page border: things continue there

    @property
    def sl(self) -> tuple[slice, slice]:
        return slice(self.y1, self.y2), slice(self.x1, self.x2)


def font_size(r: TextRegion) -> int:
    """Glyph size in px: detector estimate, else the short side of the text lines or box."""
    if r.font_size:
        return max(4, int(r.font_size))
    if r.lines:
        sides = []
        for q in r.lines:
            p = np.asarray(q, np.float32)
            a, b = np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[1])
            sides.append(min(a, b))
        return max(4, int(np.median(sides)))
    return max(4, min(r.bbox.w, r.bbox.h))


def ellipse(d: int) -> np.ndarray:
    d = max(1, int(d) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def make_crop(r: TextRegion, W: int, H: int, reach: float) -> Crop:
    """Window around the region; `near` = text lines (or the box) grown by `reach` x font."""
    fs = font_size(r)
    grow = max(2, round(fs * reach))
    pad = 2 * grow + max(4, fs // 4)
    b = r.bbox
    x1, y1 = max(0, b.x1 - pad), max(0, b.y1 - pad)
    x2, y2 = min(W, b.x2 + pad), min(H, b.y2 + pad)
    near = np.zeros((y2 - y1, x2 - x1), np.uint8)
    if r.lines:
        polys = [np.asarray(q, np.int32) - (x1, y1) for q in r.lines if len(q) >= 3]
        cv2.fillPoly(near, polys, 255)
    if not near.any():
        cv2.rectangle(near, (b.x1 - x1, b.y1 - y1), (b.x2 - x1 - 1, b.y2 - y1 - 1), 255, -1)
    lines = near > 0
    near = cv2.dilate(near, ellipse(2 * grow + 1)) > 0
    cut = np.zeros_like(near)
    cut[0, :], cut[-1, :], cut[:, 0], cut[:, -1] = y1 > 0, y2 < H, x1 > 0, x2 < W
    return Crop(x1, y1, x2, y2, fs, lines, near, (b.x1 - x1, b.y1 - y1, b.x2 - x1, b.y2 - y1), cut)


def color_distance(img: np.ndarray, color: np.ndarray) -> np.ndarray:
    d = img.astype(np.float32) - color.astype(np.float32)
    return np.sqrt((d * d).sum(axis=2) / 3.0)


def border_background(
    img: np.ndarray, near: np.ndarray, box: tuple[int, int, int, int] | None = None
) -> tuple[np.ndarray, float]:
    """(median colour, spread) of the paper around the text, which decides whether the text is
    darker or lighter. Two candidate rings: the text box border (hugs the glyphs, stays inside
    narrow balloons) and the ring just outside the text zone (better when the box spills over
    art); the flatter one wins."""
    rings = [cv2.dilate(near.astype(np.uint8), ellipse(9)).astype(bool) & ~near]
    if box is not None:
        x1, y1, x2, y2 = box
        band = np.zeros(near.shape, bool)
        band[max(0, y1 - 2) : y2 + 2, max(0, x1 - 2) : x2 + 2] = True
        band[y1 + 1 : y2 - 1, x1 + 1 : x2 - 1] = False
        rings.append(band)
    best: tuple[np.ndarray, float] | None = None
    for ring in rings:
        px = img[ring] if ring.sum() >= 16 else img.reshape(-1, 3)
        med = np.median(px, axis=0)
        spread = float(np.percentile(color_distance(px[None], med)[0], 60))
        if best is None or spread < best[1]:
            best = (med, spread)
    assert best is not None
    return best


def hysteresis(weak: np.ndarray, strong: np.ndarray, c: Crop, min_inside: float = 0.5):
    """Components of `weak` that contain a `strong` pixel, lie mostly inside the text zone and
    do not run out of the window (the reference drops components larger than their line)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(weak, bool)
    flat = labels.ravel()
    has_strong = np.bincount(flat[strong.ravel()], minlength=n) > 0
    inside = np.bincount(flat[c.near.ravel()], minlength=n)
    leaves = np.bincount(flat[c.cut.ravel()], minlength=n) > 0
    keep = has_strong & ~leaves & (inside >= min_inside * stats[:, cv2.CC_STAT_AREA])
    keep[0] = False
    return keep[labels]


def blend_ink(img: np.ndarray, bg: np.ndarray, fg: np.ndarray, alpha: float) -> np.ndarray:
    """Pixels that look like a mix of glyph and paper colour with at least `alpha` glyph:
    anti-aliased edges and faint strokes, but not differently coloured art."""
    axis = fg.astype(np.float32) - bg.astype(np.float32)
    n2 = float(axis @ axis)
    if n2 < 1.0:
        return np.zeros(img.shape[:2], bool)
    d = img.astype(np.float32) - bg.astype(np.float32)
    a = (d @ axis) / n2
    off = d - a[..., None] * axis
    resid = np.sqrt((off * off).sum(axis=2) / 3.0)
    return (a >= alpha) & (resid <= max(16.0, 0.3 * np.sqrt(n2 / 3.0)))


def structure(ink: np.ndarray, c: Crop) -> np.ndarray:
    """Ink components bigger than a glyph that are line-like or leave the text zone: balloon
    outlines, panel borders, art lines. They must never be erased."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(ink, bool)
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    flat = labels.ravel()
    outside = np.bincount(flat[~c.near.ravel()], minlength=n)
    leaves = np.bincount(flat[c.cut.ravel()], minlength=n) > 0
    thin = area < 0.2 * bw * bh
    keep = (np.maximum(bw, bh) > 1.5 * c.font) & ((outside > 0.25 * area) | leaves | thin)
    keep[0] = False
    return keep[labels]


def ring_median(img: np.ndarray, m: np.ndarray, inner: int, outer: int) -> np.ndarray | None:
    m8 = m.astype(np.uint8)
    ring = (cv2.dilate(m8, ellipse(outer)) > 0) & ~(cv2.dilate(m8, ellipse(inner)) > 0)
    return np.median(img[ring], axis=0) if ring.sum() >= 16 else None


METHODS = ("auto", "seg", "pixels")


class GlyphMask:
    """Detector text mask (or local binarisation) -> line-assigned components -> completion.

    method: "auto" uses the detector mask when there is one and binarises otherwise, "seg" trusts
    only the detector mask (regions it missed stay untouched), "pixels" ignores it.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        strong: float = 0.6,
        reach: float = 0.5,
        grow: float = 0.2,
        dilation: float = 0.12,
        min_dilate: int = 3,
        ink: float = 0.25,
        protect: bool = True,
        flat_spread: float = 40.0,
        method: str = "auto",
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"mask method must be one of {METHODS}, got {method!r}")
        self.method = method
        self.threshold, self.strong = threshold, strong
        self.reach, self.grow = reach, grow
        self.dilation, self.min_dilate = dilation, min_dilate
        self.ink, self.protect, self.flat_spread = ink, protect, flat_spread

    def refine(
        self, image_bgr: np.ndarray, regions: list[TextRegion], raw_mask: np.ndarray | None
    ) -> np.ndarray:
        img = image_bgr if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        img = img[..., :3]
        H, W = img.shape[:2]
        if raw_mask is not None and raw_mask.shape[:2] != (H, W):
            raw_mask = cv2.resize(raw_mask, (W, H), interpolation=cv2.INTER_LINEAR)
        out = np.zeros((H, W), np.uint8)
        for r in regions:
            c = make_crop(r, W, H, self.reach)
            crop = img[c.sl]
            seg = raw_mask[c.sl] if raw_mask is not None and self.method != "pixels" else None
            if seg is not None and seg.max() >= self.strong * 255:
                m = self._from_seg(crop, seg, c)
            elif self.method == "seg":
                m = np.zeros(crop.shape[:2], np.uint8)
            else:
                m = self._from_pixels(crop, c)
            out[c.sl] |= m
        return out

    def _from_seg(self, crop: np.ndarray, seg: np.ndarray, c: Crop) -> np.ndarray:
        sure = seg >= self.strong * 255
        core = hysteresis(seg >= self.threshold * 255, sure, c)
        if not core.any():
            return self._from_pixels(crop, c)
        # paper = the few pixels right around the glyphs; glyph colour = confident seg pixels
        bg = ring_median(crop, core, 5, max(9, c.font // 3))
        if bg is None:
            bg, _ = border_background(crop, c.near, c.box)
        body = cv2.erode(sure.astype(np.uint8), ellipse(3)) > 0
        fg = np.median(crop[body if body.sum() >= 8 else core], axis=0)
        return self._complete(core, blend_ink(crop, bg, fg, self.ink), sure, c)

    def _from_pixels(self, crop: np.ndarray, c: Crop) -> np.ndarray:
        bg, spread = border_background(crop, c.near, c.box)
        if spread > self.flat_spread:
            # Busy background (text over art): glyphs cannot be told from art by colour, so
            # erase the text lines with a small margin rather than leave half-erased strokes.
            k = 2 * max(2, round(0.1 * c.font)) + 1
            return cv2.dilate(c.lines.astype(np.uint8), ellipse(k)) * 255
        dist = color_distance(crop, bg)
        zone = dist[c.near] if c.near.any() else dist.ravel()
        otsu, _ = cv2.threshold(
            zone.astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        t = float(np.clip(otsu, max(30.0, 2 * spread), 160.0))
        sure = (dist > t) & c.near
        core = hysteresis(dist > 0.6 * t, sure, c)
        if not core.any():
            return np.zeros(crop.shape[:2], np.uint8)
        fg = np.median(crop[sure], axis=0)
        return self._complete(core, blend_ink(crop, bg, fg, self.ink), sure, c)

    def _complete(self, core: np.ndarray, ink: np.ndarray, sure: np.ndarray, c: Crop) -> np.ndarray:
        """Grow the core into touching ink (fringe, thin strokes, dots), dilate by font size,
        and cut out balloon outlines / art lines that the dilation reached."""
        zone = cv2.dilate(core.astype(np.uint8), ellipse(2 * round(c.font * self.grow) + 1)) > 0
        cand = (ink & zone) | core
        n, labels = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
        touch = np.bincount(labels[core], minlength=n) > 0
        touch[0] = False
        m = touch[labels]
        d = max(self.min_dilate, round(c.font * self.dilation) * 2 + 1)
        m = cv2.dilate(m.astype(np.uint8), ellipse(d)) > 0
        if self.protect:
            # confident text pixels on a line still go: that is a glyph touching the outline
            on_line = cv2.dilate(c.lines.astype(np.uint8), ellipse(2 * (c.font // 10) + 1)) > 0
            guard = structure(ink, c) & ~(sure & on_line)
            m &= ~(cv2.dilate(guard.astype(np.uint8), ellipse(3)) > 0)
        return m.astype(np.uint8) * 255


class DilateMask:
    """Threshold the detector mask (or binarise) inside the text zone and dilate. No component
    filtering or outline guard: predictable, slightly greedier."""

    def __init__(
        self,
        threshold: float = 0.3,
        reach: float = 0.25,
        dilation: float = 0.15,
        min_dilate: int = 3,
    ) -> None:
        self.threshold, self.reach = threshold, reach
        self.dilation, self.min_dilate = dilation, min_dilate

    def refine(
        self, image_bgr: np.ndarray, regions: list[TextRegion], raw_mask: np.ndarray | None
    ) -> np.ndarray:
        img = image_bgr if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        H, W = img.shape[:2]
        if raw_mask is not None and raw_mask.shape[:2] != (H, W):
            raw_mask = cv2.resize(raw_mask, (W, H), interpolation=cv2.INTER_LINEAR)
        out = np.zeros((H, W), np.uint8)
        for r in regions:
            c = make_crop(r, W, H, self.reach)
            if raw_mask is not None:
                m = raw_mask[c.sl] >= self.threshold * 255
            else:
                crop = img[c.sl][..., :3]
                bg, spread = border_background(crop, c.near, c.box)
                m = color_distance(crop, bg) > max(40.0, 2 * spread)
            m &= c.near
            d = max(self.min_dilate, round(c.font * self.dilation) * 2 + 1)
            out[c.sl] |= (cv2.dilate(m.astype(np.uint8), ellipse(d)) > 0).astype(np.uint8) * 255
        return out


@MASK_REFINERS.register(
    "default",
    description="Glyph pixels only: detector mask components assigned to text lines (or local "
    "binarisation), grown into anti-aliased edges, dilated by font size, balloon outlines kept.",
)
def _build_default(
    settings,
    threshold: float = 0.3,
    strong: float = 0.6,
    reach: float = 0.5,
    grow: float = 0.2,
    dilation: float = 0.12,
    min_dilate: int = 3,
    ink: float = 0.25,
    protect: bool = True,
    flat_spread: float = 40.0,
    method: str = "auto",
    **_,
):
    return GlyphMask(
        float(threshold),
        float(strong),
        float(reach),
        float(grow),
        float(dilation),
        int(min_dilate),
        float(ink),
        _bool(protect),
        float(flat_spread),
        str(method),
    )


@MASK_REFINERS.register(
    "dilate", description="Detector mask (or binarised box) inside the text zone, dilated."
)
def _build_dilate(
    settings,
    threshold: float = 0.3,
    reach: float = 0.25,
    dilation: float = 0.15,
    min_dilate: int = 3,
    **_,
):
    return DilateMask(float(threshold), float(reach), float(dilation), int(min_dilate))


def _bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
