"""Glyph-level erase masks.

The goal is the smallest mask an inpainter needs: every glyph pixel (anti-aliased fringe,
furigana and punctuation included) and nothing else, so balloon outlines and artwork next to the
text survive. Ideas follow manga-image-translator's `complete_mask` (connected components of the
detector's text mask assigned to text lines, then dilation by font size), with the dense-CRF step
replaced by a cheap colour test against the local background, and an explicit guard that keeps
long ink structures (balloon outlines, art lines) out of the mask.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from ..models import BBox, TextRegion
from ..registry import MASK_REFINERS

log = logging.getLogger(__name__)


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
    has_lines: bool = True  # False when `lines` is just the box

    @property
    def sl(self) -> tuple[slice, slice]:
        return slice(self.y1, self.y2), slice(self.x1, self.x2)


def font_size(r: TextRegion, img: np.ndarray | None = None) -> int:
    """Glyph size in px: detector estimate, else the short side of the text lines, else measured
    from the ink inside the box (a multi-column box's short side is several glyphs wide)."""
    if r.font_size:
        return max(4, int(r.font_size))
    if r.lines:
        sides = []
        for q in r.lines:
            p = np.asarray(q, np.float32)
            if len(p) < 3:
                continue
            a, b = np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[1])
            sides.append(min(a, b))
        if sides:
            return max(4, int(np.median(sides)))
    b = r.bbox
    short = max(4, min(b.w, b.h))
    if img is None:
        return short
    est = measure_glyphs(img[max(0, b.y1) : max(0, b.y2), max(0, b.x1) : max(0, b.x2)])
    return short if est is None else int(np.clip(est, 6, short))


def measure_glyphs(crop: np.ndarray) -> int | None:
    """Font size from the ink components inside a text box: Otsu on grey, minority class = ink,
    components touching the box edge (outline, art) ignored. The 90th percentile of component
    extents is a whole glyph; detectors report about 1.15x that (line width, not ink extent)."""
    if crop.size == 0 or min(crop.shape[:2]) < 6:
        return None
    g = cv2.cvtColor(crop[..., :3], cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, ink = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if ink.mean() > 0.5:
        ink = 1 - ink
    st = cv2.connectedComponentsWithStats(ink, connectivity=8)[2]
    st = st[1:]
    h, w = g.shape[:2]
    x, y, bw, bh, area = st.T
    inner = (x > 0) & (y > 0) & (x + bw < w) & (y + bh < h) & (area >= 6)
    if inner.sum() < 2:
        return None
    return round(1.15 * float(np.percentile(np.maximum(bw, bh)[inner], 90)))


def ellipse(d: int) -> np.ndarray:
    d = max(1, int(d) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def make_crop(r: TextRegion, img: np.ndarray, reach: float) -> Crop | None:
    """Window around the region; `near` = text lines (or the box) grown by `reach` x font.
    None for a box that is empty or entirely off the page."""
    H, W = img.shape[:2]
    b = r.bbox
    bx1, bx2 = sorted((b.x1, b.x2))
    by1, by2 = sorted((b.y1, b.y2))
    if min(bx2, W) <= max(bx1, 0) or min(by2, H) <= max(by1, 0):
        return None
    if (bx1, by1) != (b.x1, b.y1):
        r = r.model_copy(update={"bbox": BBox(x1=bx1, y1=by1, x2=bx2, y2=by2)})
    fs = font_size(r, img)
    grow = max(2, round(fs * reach))
    pad = 2 * grow + max(4, fs // 4)
    x1, y1 = max(0, bx1 - pad), max(0, by1 - pad)
    x2, y2 = min(W, bx2 + pad), min(H, by2 + pad)
    near = np.zeros((y2 - y1, x2 - x1), np.uint8)
    if r.lines:
        polys = [np.asarray(q, np.int32) - (x1, y1) for q in r.lines if len(q) >= 3]
        cv2.fillPoly(near, polys, 255)
    has_lines = bool(near.any())
    if not has_lines:
        cv2.rectangle(near, (bx1 - x1, by1 - y1), (bx2 - x1 - 1, by2 - y1 - 1), 255, -1)
    lines = near > 0
    near = cv2.dilate(near, ellipse(2 * grow + 1)) > 0
    cut = np.zeros_like(near)
    cut[0, :], cut[-1, :], cut[:, 0], cut[:, -1] = y1 > 0, y2 < H, x1 > 0, x2 < W
    box = (bx1 - x1, by1 - y1, bx2 - x1, by2 - y1)
    return Crop(x1, y1, x2, y2, fs, lines, near, box, cut, has_lines)


def as_u8_mask(raw: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Detector mask as single-channel uint8 0..255 at page size, whatever it came as
    (bool, float 0..1, uint16, 3-channel, lower resolution)."""
    if raw is None:
        return None
    m = np.asarray(raw)
    if m.ndim == 3:
        m = m.max(axis=2)
    if m.dtype == bool:
        m = m.astype(np.uint8) * 255
    elif m.dtype != np.uint8:
        f = m.astype(np.float32)
        top = float(f.max()) if f.size else 0.0
        scale = 255.0 if top <= 1.0 else 255.0 / 65535.0 if top > 255.0 else 1.0
        m = np.clip(f * scale, 0, 255).astype(np.uint8)
    if m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return m


def as_bgr(image: np.ndarray) -> np.ndarray:
    img = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img[..., :3]
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        f = img.astype(np.float32)
        img = np.clip(f * (255.0 if f.size and f.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def color_distance(img: np.ndarray, color: np.ndarray) -> np.ndarray:
    d = img.astype(np.float32) - color.astype(np.float32)
    return np.sqrt((d * d).sum(axis=2) / 3.0)


def paper_candidates(
    img: np.ndarray, near: np.ndarray, box: tuple[int, int, int, int] | None = None
) -> list[tuple[np.ndarray, float]]:
    """(median colour, spread) of the paper around the text, flattest first; the paper colour
    decides whether the text is darker or lighter. Rings: the text box border (hugs the glyphs,
    stays inside narrow balloons) and the ring just outside the text zone (better when the box
    spills over art)."""
    rings = [cv2.dilate(near.astype(np.uint8), ellipse(9)).astype(bool) & ~near]
    if box is not None:
        x1, y1, x2, y2 = box
        band = np.zeros(near.shape, bool)
        band[max(0, y1 - 2) : y2 + 2, max(0, x1 - 2) : x2 + 2] = True
        band[y1 + 1 : y2 - 1, x1 + 1 : x2 - 1] = False
        rings.append(band)
    out = []
    for ring in rings:
        px = img[ring] if ring.sum() >= 16 else img.reshape(-1, 3)
        med = np.median(px, axis=0)
        out.append((med, float(np.percentile(color_distance(px[None], med)[0], 60))))
    return sorted(out, key=lambda c: c[1])


def border_background(
    img: np.ndarray, near: np.ndarray, box: tuple[int, int, int, int] | None = None
) -> tuple[np.ndarray, float]:
    return paper_candidates(img, near, box)[0]


def hysteresis(
    weak: np.ndarray,
    strong: np.ndarray,
    c: Crop,
    min_inside: float = 0.5,
    retry: tuple[np.ndarray, ...] = (),
    min_area: int = 1,
) -> np.ndarray:
    """Components of `weak` that contain a `strong` pixel, lie mostly inside the text zone and
    do not run out of the window (the reference drops components larger than their line).
    A rejected component is retried on the stricter `retry` masks in turn, so a glyph is not
    lost because weak detector bleed or a faint art edge joins it to something larger; retried
    pieces must lie almost wholly in the text zone and be no mere speck (specks of an outline
    would seed the completion step)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(weak, bool)
    flat = labels.ravel()
    area = stats[:, cv2.CC_STAT_AREA]
    has_strong = np.bincount(flat[strong.ravel()], minlength=n) > 0
    inside = np.bincount(flat[c.near.ravel()], minlength=n)
    leaves = np.bincount(flat[c.cut.ravel()], minlength=n) > 0
    keep = has_strong & ~leaves & (inside >= min_inside * area) & (area >= min_area)
    keep[0] = False
    out = keep[labels]
    if retry:
        rejected = has_strong & ~keep & (area >= min_area)
        rejected[0] = False
        if rejected.any():
            again = retry[0] & rejected[labels]
            speck = max(4, round(0.1 * c.font) ** 2)
            out |= hysteresis(again, strong & again, c, max(min_inside, 0.9), retry[1:], speck)
    return out


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


def binarise(
    crop: np.ndarray, c: Crop, max_ink: float = 0.5
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """(paper colour, distance from it, spread) for the flattest paper candidate whose ink does
    not swamp the text lines. Text covers well under half of its own lines; when "ink" covers
    more, the ring sampled art or the balloon interior and the polarity is upside down."""
    for bg, spread in paper_candidates(crop, c.near, c.box):
        dist = color_distance(crop, bg)
        if (dist[c.lines] > max(30.0, 2 * spread)).mean() <= max_ink:
            return bg, dist, spread
    return None


def busy_box(c: Crop) -> np.ndarray:
    """Text over art: glyphs cannot be told from art by colour, so erase the text lines with
    a small margin rather than leave half-erased strokes."""
    k = 2 * max(2, round(0.1 * c.font)) + 1
    return cv2.dilate(c.lines.astype(np.uint8), ellipse(k)) * 255


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
        img = as_bgr(image_bgr)
        H, W = img.shape[:2]
        raw = as_u8_mask(raw_mask, (H, W)) if self.method != "pixels" else None
        out = np.zeros((H, W), np.uint8)
        for r in regions:
            c = make_crop(r, img, self.reach)
            if c is None:
                continue
            crop = img[c.sl]
            seg = raw[c.sl] if raw is not None else None
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
        core = hysteresis(seg >= self.threshold * 255, sure, c, retry=(sure,))
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
        found = binarise(crop, c)
        if found is None or found[2] > self.flat_spread:
            return busy_box(c)
        bg, dist, spread = found
        zone = dist[c.near] if c.near.any() else dist.ravel()
        otsu, _ = cv2.threshold(
            zone.astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        t = float(np.clip(otsu, max(30.0, 2 * spread), 160.0))
        sure = (dist > t) & c.near
        core = hysteresis(dist > 0.6 * t, sure, c, retry=tuple(dist > k * t for k in (1, 2, 3)))
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
            guard = structure(ink, c)
            if c.has_lines:
                # confident text pixels on a real text line still go: a glyph touching the outline
                k = ellipse(2 * (c.font // 10) + 1)
                guard &= ~(sure & (cv2.dilate(c.lines.astype(np.uint8), k) > 0))
            m &= ~(cv2.dilate(guard.astype(np.uint8), ellipse(3)) > 0)
        return m.astype(np.uint8) * 255


class DilateMask:
    """Threshold the detector mask inside the text zone and dilate; without one, binarise
    against the paper colour and keep the ink components that stay in the window. No colour
    completion or outline guard: predictable, slightly greedier."""

    def __init__(
        self,
        threshold: float = 0.3,
        reach: float = 0.25,
        dilation: float = 0.15,
        min_dilate: int = 3,
        flat_spread: float = 40.0,
    ) -> None:
        self.threshold, self.reach = threshold, reach
        self.dilation, self.min_dilate = dilation, min_dilate
        self.flat_spread = flat_spread

    def refine(
        self, image_bgr: np.ndarray, regions: list[TextRegion], raw_mask: np.ndarray | None
    ) -> np.ndarray:
        img = as_bgr(image_bgr)
        H, W = img.shape[:2]
        raw = as_u8_mask(raw_mask, (H, W))
        out = np.zeros((H, W), np.uint8)
        for r in regions:
            c = make_crop(r, img, self.reach)
            if c is None:
                continue
            if raw is not None:
                m = raw[c.sl] >= self.threshold * 255
            else:
                found = binarise(img[c.sl], c)
                if found is None or found[2] > self.flat_spread:
                    out[c.sl] |= busy_box(c)
                    continue
                _, dist, spread = found
                t = max(40.0, 2 * spread)
                # ink that runs out of the window is art or a balloon outline, not text
                m = hysteresis(dist > t, c.near, c, retry=(dist > 2 * t,))
            m &= c.near
            d = max(self.min_dilate, round(c.font * self.dilation) * 2 + 1)
            out[c.sl] |= (cv2.dilate(m.astype(np.uint8), ellipse(d)) > 0).astype(np.uint8) * 255
        return out


DEFAULT_OPTS = (
    "threshold", "strong", "reach", "grow", "dilation", "min_dilate", "ink", "protect",
    "flat_spread", "method",
)  # fmt: skip
DILATE_OPTS = ("threshold", "reach", "dilation", "min_dilate", "flat_spread")


def _unknown(method: str, extra: dict, known: tuple[str, ...]) -> None:
    if extra:
        log.warning(
            "mask_refiner %s ignores unknown option(s) %s; known: %s",
            method, ", ".join(sorted(extra)), ", ".join(known),
        )  # fmt: skip


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
    **extra,
):
    _unknown("default", extra, DEFAULT_OPTS)
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
    flat_spread: float = 40.0,
    **extra,
):
    _unknown("dilate", extra, DILATE_OPTS)
    return DilateMask(
        float(threshold), float(reach), float(dilation), int(min_dilate), float(flat_spread)
    )


def _bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
