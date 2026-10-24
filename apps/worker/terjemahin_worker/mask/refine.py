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
from dataclasses import dataclass, replace

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
    reach: float = 0.5  # `near` = `lines` grown by reach x font

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
    return glyph_extent(ink < 1 if ink.mean() > 0.5 else ink > 0)


def glyph_extent(ink: np.ndarray) -> int | None:
    st = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)[2][1:]
    h, w = ink.shape[:2]
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
    bx1, bx2, by1, by2 = max(bx1, 0), min(bx2, W), max(by1, 0), min(by2, H)
    if bx2 <= bx1 or by2 <= by1:
        return None
    if (bx1, by1, bx2, by2) != (b.x1, b.y1, b.x2, b.y2):
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
    return Crop(x1, y1, x2, y2, fs, lines, near, box, cut, has_lines, reach)


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


def box_zone(c: Crop, grow: float = 0.1) -> np.ndarray:
    """The region box grown by a small margin (a tight box may cut the glyphs' fringe)."""
    m = max(2, round(grow * c.font))
    x1, y1, x2, y2 = c.box
    z = np.zeros(c.near.shape, bool)
    z[max(0, y1 - m) : y2 + m, max(0, x1 - m) : x2 + m] = True
    return z


def line_kernel(length: int, angle: float) -> np.ndarray:
    k = np.zeros((length, length), np.uint8)
    h = (length - 1) / 2.0
    dx, dy = h * np.cos(angle), h * np.sin(angle)
    cv2.line(k, (round(h - dx), round(h - dy)), (round(h + dx), round(h + dy)), 1, 1)
    return k


def crossing_lines(ink: np.ndarray, c: Crop, font: int, angles: int = 16) -> np.ndarray:
    """Art lines running through a bare box: thin straight runs of ink, longer than a glyph
    stroke is wide, that go on well past the box. Glyph strokes, however long and straight,
    end near it (a tight box may cut them); each direction is tested on its own, so a glyph
    stroke that touches a line at an angle is not taken for part of it."""
    beyond = ~box_zone(c, 0.6)
    out = np.zeros(ink.shape, bool)
    n, labels = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    if n <= 1:
        return out
    flat = labels.ravel()
    cand = (np.bincount(flat[beyond.ravel()], minlength=n) >= 10) & (
        np.bincount(flat[box_zone(c).ravel()], minlength=n) > 0
    )
    cand[0] = False
    if not cand.any():
        return out
    src = cand[labels].astype(np.uint8)
    # thin structures only: inside a solid area (another background, a glyph's body) there
    # are straight runs in every direction
    solid = cv2.morphologyEx(src, cv2.MORPH_OPEN, ellipse(max(11, round(0.2 * font))))
    src[solid > 0] = 0
    # the font of a bare box is a guess that glyphs fused with a line make too small
    x1, y1, x2, y2 = c.box
    length = int(np.clip(round(0.4 * max(font, min(x2 - x1, y2 - y1))), 9, 61)) | 1
    runs = []  # (ys, xs, clear of the box) of each run that goes on past the box
    every = np.zeros(ink.shape, bool)
    for i in range(angles):
        run = cv2.morphologyEx(src, cv2.MORPH_OPEN, line_kernel(length, np.pi * i / angles))
        ys, xs = np.nonzero(run)
        if not ys.size:
            continue
        lab = cv2.connectedComponents(run, connectivity=8)[1][ys, xs]
        clear = beyond[ys, xs]
        for j in np.flatnonzero(np.bincount(lab[clear])):
            if j:
                sel = lab == j
                runs.append((ys[sel], xs[sel], clear[sel]))
                every[ys[sel & clear], xs[sel & clear]] = True
    fy, fx = np.nonzero(every)
    for ys, xs, clear in runs:
        keep = straight_part(ys, xs, clear, fy, fx)
        out[ys[keep], xs[keep]] = True
    return out


def fit_line(ys: np.ndarray, xs: np.ndarray):
    """Distance to the line fitted through the given pixels, as a function of (y, x)."""
    step = max(1, ys.size // 1000)  # a thousand pixels fix a straight line well enough
    pts = np.stack([xs[::step], ys[::step]], axis=1).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return lambda yy, xx: np.abs((xx - x0) * vy - (yy - y0) * vx)


def straight_part(
    ys: np.ndarray, xs: np.ndarray, clear: np.ndarray, fy: np.ndarray, fx: np.ndarray
) -> np.ndarray:
    """Which pixels of a straight run (`ys, xs`) lie within the width its line has clear of
    the box (`clear`; `fy, fx`: the runs of every direction there): next to the text, the
    strokes of a glyph touching the line fit the line kernel too. A kernel a little off the
    line's angle cuts the band short, so the line is fitted again on all the runs near the
    first fit."""
    if clear.sum() < 8:
        return np.ones(ys.shape, bool)
    perp = fit_line(ys[clear], xs[clear])
    band = perp(fy, fx) <= np.percentile(perp(ys[clear], xs[clear]), 98) + 3
    perp = fit_line(fy[band], fx[band])
    half = float(np.percentile(perp(fy[band], fx[band]), 98)) + 3
    return perp(ys, xs) <= half


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
    if not c.has_lines:
        # a bare box's text zone reaches half a glyph past it (the box may cut the glyphs):
        # ink mostly beyond the box is a neighbour's text or art
        keep &= np.bincount(flat[box_zone(c).ravel()], minlength=n) >= 0.5 * area
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


def dominant(px: np.ndarray) -> np.ndarray:
    """Median of the most frequent 16-level colour bin: the paper, even with lots of ink."""
    q = (px >> 4).astype(np.int32)
    key = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    return np.median(px[key == np.bincount(key).argmax()], axis=0)


def balloon_paper(
    crop: np.ndarray, c: Crop, tol: float = 30.0
) -> tuple[np.ndarray, np.ndarray, float, Crop] | None:
    """Paper for a box that spills over the balloon outline or onto art, where the border rings
    mix paper with outline and art: the dominant colour of the box (or of its middle, when the
    box holds more background than balloon) and its solid connected area, the balloon interior.
    The text zone is cut to that area's filled outline, so art and outlines beyond it are out of
    reach. Of the candidates that work, the one leaving the least ink on the text lines wins."""
    x1, y1, x2, y2 = c.box
    if (x2 - x1) * (y2 - y1) < 64:
        return None
    dy, dx = (y2 - y1) // 4, (x2 - x1) // 4
    cands = [dominant(crop[y1:y2, x1:x2].reshape(-1, 3))]
    mid = crop[y1 + dy : y2 - dy, x1 + dx : x2 - dx].reshape(-1, 3)
    if len(mid) >= 16:
        alt = dominant(mid)
        if color_distance(alt[None, None], cands[0])[0, 0] > tol / 2:
            cands.append(alt)
    best = None
    for bg in cands:
        found = paper_area(crop, c, bg, tol)
        if found is not None and (best is None or found[0] < best[0]):
            best = found
    return None if best is None else best[1:]


def paper_area(
    crop: np.ndarray,
    c: Crop,
    bg: np.ndarray,
    tol: float,
    solid: int = 7,
    min_solid: float = 0.15,
    max_ink: float = 0.65,
) -> tuple[float, np.ndarray, np.ndarray, float, Crop] | None:
    """(ink share of the lines, paper colour, distance, spread, narrowed crop) for paper `bg`.
    None when it has no solid area in the box: an opening by `solid` px removes the thin gaps
    between screentone dots or hatching."""
    x1, y1, x2, y2 = c.box
    dist = color_distance(crop, bg)
    paper = (dist < tol).astype(np.uint8)
    n, labels = cv2.connectedComponents(paper, connectivity=4)
    core = cv2.erode(paper, ellipse(solid)) > 0
    inbox = np.zeros(paper.shape, bool)
    inbox[y1:y2, x1:x2] = True
    votes = np.bincount(labels[core & inbox], minlength=n)
    votes[0] = 0
    # every solid paper piece counts: a glyph touching the outline can cut a balloon in
    # pieces, and a loose box can hold a neighbour balloon bigger than the text's own
    area = np.bincount(labels[inbox], minlength=n)
    keep = (votes >= max(64, 0.02 * votes.max())) & (votes >= 0.25 * area)
    if votes[keep].sum() < min_solid * inbox.sum():
        return None
    region = keep[labels].astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(region)
    cv2.drawContours(filled, contours, -1, 1, -1)
    spread = float(np.percentile(dist[(region > 0) & inbox], 60))
    ink = dist > max(30.0, 2 * spread)
    font = c.font
    if not c.has_lines:  # a loose box measured background shapes too; the balloon's ink is text
        est = glyph_extent(ink & (filled > 0))
        font = font if est is None else int(np.clip(est, 6, font))
    # glyphs touching the outline dent the paper's contour instead of being holes in it
    interior = cv2.dilate(filled, ellipse(2 * max(2, round(0.15 * font)) + 1)) > 0
    lines = c.lines & interior
    if not lines.any():
        return None
    share = float(ink[lines].mean())
    if share > max_ink:
        return None
    near = c.near & interior
    if font < c.font:
        grow = max(2, round(font * c.reach))
        near = (cv2.dilate(lines.astype(np.uint8), ellipse(2 * grow + 1)) > 0) & interior
    return share, bg, dist, spread, replace(c, font=font, lines=lines, near=near)


def textured(dist: np.ndarray, c: Crop, tol: float = 30.0, solid: int = 7) -> bool:
    """Screentone or hatching passes the ring test (most ring pixels are the paper between the
    dots) but has no solid paper anywhere near the text: the gaps vanish under an opening."""
    paper = cv2.erode((dist < tol).astype(np.uint8), ellipse(solid)) > 0
    return float(paper[c.near].mean()) < 0.05


def find_paper(
    crop: np.ndarray, c: Crop, flat_spread: float
) -> tuple[np.ndarray, np.ndarray, float, Crop] | None:
    """Rings around the text lines first; the balloon interior when they are mixed, and for a
    bare box (whose edge may cross the outline or art) before the rings."""
    if not c.has_lines and (b := balloon_paper(crop, c)) is not None:
        return b
    found = binarise(crop, c)
    if found is not None and found[2] <= flat_spread and not textured(found[1], c):
        return (*found, c)
    return None if not c.has_lines else balloon_paper(crop, c)


def stroke_width(m: np.ndarray) -> float:
    """Typical stroke thickness of a binary shape: 2 x area / boundary length (exact for long
    strokes, half the diameter for a dot)."""
    m8 = np.pad(m.astype(np.uint8), 1)
    contours, _ = cv2.findContours(m8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    length = sum(cv2.arcLength(k, True) for k in contours)
    return max(1.0, 2.0 * float(m8.sum()) / length) if length > 0 else 1.0


def nearest(px: np.ndarray, colors: list[np.ndarray]) -> np.ndarray:
    """Index of the closest of `colors` for every pixel."""
    return np.argmin(np.stack([color_distance(px, k) for k in colors]), axis=0)


def segment_distance(px: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance of each pixel colour (N x 3) to the nearest mix of colours `a` and `b`."""
    ab = (b - a).astype(np.float32)
    d = px.astype(np.float32) - a.astype(np.float32)
    t = np.clip((d @ ab) / max(float(ab @ ab), 1.0), 0.0, 1.0)
    off = d - t[:, None] * ab
    return np.sqrt((off * off).sum(axis=1) / 3.0)


def flat_surround(px: np.ndarray, fg: np.ndarray, tol: float = 30.0) -> float:
    """Share of `px` that is one colour, or two (paper and an outline ring, or two papers),
    or their anti-aliased mixes with each other or the glyph: a glyph's surround. Art with
    its shading and lines rarely is."""
    o1 = dominant(px)
    close = color_distance(px[None], o1)[0] < tol
    if close.all() or (~close).sum() < 8:
        return float(close.mean())
    o2 = dominant(px[~close])
    d = np.minimum.reduce([segment_distance(px, a, b) for a, b in ((o1, o2), (o1, fg), (o2, fg))])
    return float((d < tol).mean())


def glyph_candidates(
    ink: np.ndarray,
    zone: np.ndarray,
    c: Crop,
    font: int,
    speck: int,
    split: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(labels, stats, glyph-like, fused) per label. Glyph-like: in the zone, glyph-sized,
    no speck; strokes `split` off an art line may be several touching glyphs. Fused: ink with
    a fair share in the zone that is not glyph-like, as text fused with a line."""
    n, labels, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    flat = labels.ravel()
    inside = np.bincount(flat[zone.ravel()], minlength=n)
    leaves = np.bincount(flat[c.cut.ravel()], minlength=n) > 0
    area = st[:, cv2.CC_STAT_AREA]
    # a bare box's font size is a guess (lettering sizes vary): any glyph up to its short side
    x1, y1, x2, y2 = c.box
    size = 1.5 * font if c.has_lines else max(1.5 * font, min(x2 - x1, y2 - y1))
    small = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) <= size
    if split is not None:
        small |= np.bincount(flat[split.ravel()], minlength=n) >= 0.9 * area
    ok = (inside >= 0.9 * area) & ~leaves & small & (area >= speck)
    fused = ~ok & (inside >= 4 * speck)
    ok[0] = fused[0] = False
    return labels, st, ok, fused


STUCK = 2.0  # stroke area per line width and extent: one line crossing the zone stays below


def split_fused(comp: np.ndarray, zone: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """Text strokes fused with an art line or outline that runs through the text zone, split
    off by stroke width: an opening just wider than the line (measured where it runs outside
    the zone) removes the line and keeps the thicker glyph strokes, which then grow back over
    their own edges. Returns (strokes or None, whether the zone part looks like text that
    could not be split: far more stroke than one line crossing the zone would have)."""
    line, inner = comp & ~zone, comp & zone
    if line.sum() < 20:
        return None, False
    lw = stroke_width(line)
    k = int(np.ceil(lw)) + 2
    body = cv2.morphologyEx(inner.astype(np.uint8), cv2.MORPH_OPEN, ellipse(k)) > 0
    # strokes clearly thicker than the line, not just the knots where lines cross or bend
    if body.sum() >= 0.25 * inner.sum() and stroke_width(body) >= 1.4 * lw:
        return inner & (cv2.dilate(body.astype(np.uint8), ellipse(k)) > 0), False
    ys, xs = np.nonzero(inner)
    span = max(int(xs.max() - xs.min()), int(ys.max() - ys.min())) + 1
    return None, float(inner.sum()) >= STUCK * lw * span


def outlined_glyphs(
    crop: np.ndarray,
    c: Crop,
    bg: np.ndarray,
    fg: np.ndarray,
    font: int,
    outlined_only: bool = False,
    alpha: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Glyphs found one by one, whatever the paper detection made of the text zone: solid
    components of the text colour that fit a glyph, lie in the text zone and sit on a flat
    surround (paper, or an outline ring on anything; art around an eye or a hair strand is
    neither). A glyph with an outline ring (black SFX with a white edge) comes with its ring,
    and when most of the text is outlined, ink without a ring is art showing between glyphs.
    Glyphs fused with an art line are split from it by stroke width.
    This finds the glyphs of text on two backgrounds (only one can be the paper) or touching
    an art line (the paper contour dents around them), which the paper area leaves out.
    Returns (glyphs with rings, text-like ink fused with art that could not be split)."""
    out = np.zeros(crop.shape[:2], bool)
    stuck = np.zeros_like(out)
    ink = blend_ink(crop, bg, fg, alpha)
    if not ink.any():
        return out, stuck
    margin = max(2, round(0.1 * font))
    zone = cv2.dilate(c.lines.astype(np.uint8), ellipse(2 * margin + 1)) > 0
    speck = max(6, round(0.04 * font) ** 2)
    labels, st, ok, fused = glyph_candidates(ink, zone, c, font, speck)
    if fused.any():
        body = np.zeros_like(out)
        deep = cv2.erode(zone.astype(np.uint8), ellipse(2 * round(0.25 * font) + 1)) > 0
        for i in np.flatnonzero(fused):
            x, y, w, h = st[i, :4]
            comp = labels[y : y + h, x : x + w] == i
            part, text = split_fused(comp, zone[y : y + h, x : x + w])
            if part is not None:
                body[y : y + h, x : x + w] |= part
            elif text:
                inner = comp & zone[y : y + h, x : x + w]
                # text in the middle of the box, on flat paper: not a balloon outline that a
                # loose box cuts at its edge
                if (inner & deep[y : y + h, x : x + w]).sum() >= 0.5 * inner.sum() and on_paper(
                    crop[y : y + h, x : x + w], inner, bg
                ):
                    stuck[y : y + h, x : x + w] |= inner
        if body.any():
            parts = (ink & ~fused[labels]) | body
            labels, st, ok, _ = glyph_candidates(parts, zone, c, font, speck, split=body)
    x, y, bw, bh = st[:, :4].T
    soft = blend_ink(crop, bg, fg, 0.35)
    H, W = ink.shape
    found = []  # (window, glyph, ring or None)
    tone = max(40.0, 0.3 * float(color_distance(fg[None, None], bg)[0, 0]))
    for i in np.flatnonzero(ok):
        g = labels[y[i] : y[i] + bh[i], x[i] : x[i] + bw[i]] == i
        sw, extent = stroke_width(g), max(bw[i], bh[i])
        if sw > 0.3 * extent and extent > 0.4 * font:
            continue  # a solid blob bigger than a dot: a shadow, not strokes
        gx = crop[y[i] : y[i] + bh[i], x[i] : x[i] + bw[i]][g]
        if color_distance(np.median(gx, axis=0)[None, None], fg)[0, 0] > tone:
            continue  # merely darker than the paper (shading), not the text colour
        r = int(np.clip(round(0.5 * sw), 3, max(3, round(0.15 * font))))
        ro = r + max(3, round(0.4 * sw))  # outline search radius
        p = ro + 6
        sy = slice(max(0, y[i] - p), min(H, y[i] + bh[i] + p))
        sx = slice(max(0, x[i] - p), min(W, x[i] + bw[i] + p))
        k = (labels[sy, sx] == i).astype(np.uint8)
        sub = crop[sy, sx]
        rim = (cv2.dilate(k, ellipse(5)) > 0) & (k == 0)
        near_ring = (cv2.dilate(k, ellipse(2 * r + 3)) > 0) & ~rim & (k == 0)
        if near_ring.sum() < 8 or soft[sy, sx][near_ring].mean() > 0.25:
            continue  # more ink around it: art, not a glyph
        if flat_surround(sub[near_ring], fg) < 0.8:
            continue
        do = cv2.dilate(k, ellipse(2 * ro + 1)) > 0
        far = (cv2.dilate(k, ellipse(2 * ro + 11)) > 0) & ~do
        ring = None
        if far.sum() >= 8:
            # outline = what hugs the glyph and is neither paper nor a glyph/paper blend
            paper = dominant(sub[far])
            odd = (color_distance(sub, paper) > 40) & ~blend_ink(sub, paper, fg, 0.1)
            if odd[rim].mean() >= 0.4:
                o = np.median(sub[rim & odd], axis=0)
                ring = do & (k == 0) & (nearest(sub, [o, paper, fg]) == 0)
                nn, lab = cv2.connectedComponents(ring.astype(np.uint8), connectivity=8)
                touch = np.bincount(lab[rim & ring], minlength=nn) > 0
                touch[0] = False
                ring = touch[lab]
        found.append(((sy, sx), k > 0, ring))
    lettered = sum(int(g.sum()) for _, g, ring in found if ring is not None)
    outlined = lettered >= 0.5 * sum(int(g.sum()) for _, g, _ in found)
    if outlined_only and not outlined:
        return out, stuck
    for (sy, sx), g, ring in found:
        if ring is not None:
            out[sy, sx] |= g | ring
        elif not outlined and not outlined_only:
            out[sy, sx] |= g
    return out, stuck


def text_color(crop: np.ndarray, c: Crop, bg: np.ndarray) -> np.ndarray:
    """Glyph colour when no pixel is surely text: the text zone's pixels farthest from paper."""
    px = crop[c.lines] if c.lines.any() else crop.reshape(-1, 3)
    d = color_distance(px[None], bg)[0]
    return np.median(px[d >= np.percentile(d, 95)], axis=0)


def glyph_color(
    crop: np.ndarray, sure: np.ndarray, core: np.ndarray, c: Crop, bg: np.ndarray
) -> np.ndarray:
    """Text colour: the confident ink of the glyph-like components. In a bare box, art lines
    crossing it are confident ink too, and their colour mixed in makes a text colour that
    neither the glyphs nor the lines have."""
    own = sure & core
    if not c.has_lines and own.sum() >= max(16, round(0.1 * c.font) ** 2):
        return np.median(crop[own], axis=0)
    return np.median(crop[sure], axis=0) if sure.any() else text_color(crop, c, bg)


def extremes(crop: np.ndarray, c: Crop) -> list[np.ndarray]:
    """Darkest and lightest colours of the text zone: fill and edge of outlined lettering,
    whichever of them the paper detection took for text."""
    px = crop[c.lines] if c.lines.any() else crop.reshape(-1, 3)
    lum = px.astype(np.float32) @ np.float32([0.114, 0.587, 0.299])
    lo, hi = np.percentile(lum, (2, 98))
    return [np.median(px[lum <= lo], axis=0), np.median(px[lum >= hi], axis=0)]


def lettering(
    crop: np.ndarray, c: Crop, bg: np.ndarray, fg: np.ndarray, font: int
) -> tuple[np.ndarray, np.ndarray]:
    """Glyphs found one by one in the text colour, plus outlined lettering in the zone's
    darkest or lightest colour (the paper detection may have taken the outline, or another
    background, for the text); and the text-like ink fused with art that no stroke width
    could split. A text colour that is no extreme of the zone is a mix of fill, outline and
    background: only outlined lettering is looked for then."""
    ext = extremes(crop, c)
    dist = [float(color_distance(e[None, None], fg)[0, 0]) for e in ext]
    if min(dist) < 60:
        extra, stuck = outlined_glyphs(crop, c, bg, fg, font)
    else:
        extra = stuck = np.zeros(crop.shape[:2], bool)
    for alt, d_fg in zip(ext, dist, strict=True):
        # not the text colour again, nor the paper (a light balloon's lightest colour)
        if d_fg > 40 and color_distance(alt[None, None], bg)[0, 0] > 30:
            more, fused = outlined_glyphs(crop, c, bg, alt, font, outlined_only=True)
            extra, stuck = extra | more, stuck | fused
    return extra, stuck & ~extra


def on_paper(crop: np.ndarray, comp: np.ndarray, bg: np.ndarray, tol: float = 30.0) -> bool:
    """Ink with paper all around it (a line crossing text on flat paper), unlike a balloon
    outline, which has the balloon on one side and the art on the other."""
    k = comp.astype(np.uint8)
    ring = (cv2.dilate(k, ellipse(9)) > 0) & ~(cv2.dilate(k, ellipse(3)) > 0)
    ring[[0, -1], :] = ring[:, [0, -1]] = False  # the window edge cuts the ring short
    if ring.sum() < 16:
        return False
    return float((color_distance(crop, bg)[ring] < tol).mean()) >= 0.85


def stuck_fill(stuck: np.ndarray, erased: np.ndarray, c: Crop) -> np.ndarray:
    """Box erase, within the text zone, of each blob of text that stays fused with an art
    line and is not erased already: half erased strokes are worse than an inpainted patch."""
    out = np.zeros(stuck.shape, np.uint8)
    pad = max(2, round(0.1 * c.font))
    n, labels, st, _ = cv2.connectedComponentsWithStats(stuck.astype(np.uint8), connectivity=8)
    done = np.bincount(labels[erased & stuck], minlength=n)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area >= max(16, round(0.1 * c.font) ** 2) and done[i] < 0.9 * area:
            cv2.rectangle(out, (x - pad, y - pad), (x + w + pad - 1, y + h + pad - 1), 255, -1)
    return out & busy_box(c)


def leftover_glyphs(
    page: np.ndarray,
    c: Crop,
    bg: np.ndarray,
    fg: np.ndarray,
    spread: float,
    erased: np.ndarray,
    font: int,
    grow: int,
) -> np.ndarray:
    """Glyphs of a bare box that the passes above left, dilated by `grow`; typically ones
    fused with an art line (together they are one long structure, which no pass may erase):
    components of the text colour once the art lines running through the box are cut out of
    the ink (crossing_lines). A component counts when it lies in the box, is no speck, no
    long thin curve and no ring round a big hole (a balloon outline), is not erased yet in
    good part, stays clear of the window edge and has flat paper round it (bar the cut
    lines); and when it touched a cut line, that line is art, not a balloon outline cut where
    it runs straight (as dark as its text, the outline's pieces pass every test above): it
    has another colour than the glyph (measured clear of the text), or the glyph lies beside
    it (see beside). A line of the glyph's colour may still be an outline: the dilation
    stops at it. Where a glyph crosses a line of another colour, its pixels on the line go
    too.

    Lines are told by how they run on clear of the box, so this pass looks at a window of
    its own, as wide as the widest refiner's (`erased` and the result are in `c`'s)."""
    if c.has_lines:
        return np.zeros(c.near.shape, bool)
    if c.reach >= 0.5:
        return _leftover(page[c.sl], c, bg, fg, spread, erased, font, grow)
    x1, y1, x2, y2 = c.box
    region = TextRegion(id=0, bbox=BBox(x1=c.x1 + x1, y1=c.y1 + y1, x2=c.x1 + x2, y2=c.y1 + y2))
    wide = make_crop(region, page, max(c.reach, 0.5))
    if wide is None:
        return np.zeros(c.near.shape, bool)
    oy, ox = c.y1 - wide.y1, c.x1 - wide.x1
    inner = (slice(oy, oy + c.near.shape[0]), slice(ox, ox + c.near.shape[1]))
    done = np.zeros(wide.near.shape, bool)
    done[inner] = erased
    return _leftover(page[wide.sl], wide, bg, fg, spread, done, font, grow)[inner]


def _leftover(
    crop: np.ndarray,
    c: Crop,
    bg: np.ndarray,
    fg: np.ndarray,
    spread: float,
    erased: np.ndarray,
    font: int,
    grow: int,
) -> np.ndarray:
    out = np.zeros(crop.shape[:2], bool)
    ink = blend_ink(crop, bg, fg, 0.5)
    dist = color_distance(crop, bg)
    lines = crossing_lines((dist > 40) | ink, c, font)
    art = cv2.dilate(lines.astype(np.uint8), ellipse(3)) > 0
    line_lab, line_color = line_colors(crop, lines, dist, c)
    ink &= ~art
    n, labels, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    if n <= 1:
        return out
    flat = labels.ravel()
    x, y, w, h, area = st.T
    H, W = ink.shape
    speck = max(16, round(0.1 * font) ** 2)
    inbox = np.bincount(flat[box_zone(c).ravel()], minlength=n)
    done = np.bincount(flat[erased.ravel()], minlength=n)
    edge = (x == 0) | (y == 0) | (x + w == W) | (y + h == H)
    curve = (np.maximum(w, h) > 1.5 * font) & (area < 0.15 * w * h)
    # a glyph the passes above cut in half is a component part erased: the rest still goes
    cand = (inbox >= 0.9 * area) & (area >= speck) & (area - done >= speck)
    cand &= ~edge & ~curve
    cand[0] = False
    flat_tol = max(12.0, 3 * spread)

    stop = np.zeros_like(out)  # lines of the text colour: the dilation does not cross them

    def art_line(j: int, comp: np.ndarray, own: np.ndarray) -> bool:
        col = line_color[j]
        if col is not None and color_distance(col[None, None], own)[0, 0] >= 25:
            return True
        if beside(line_lab == j, comp, font, line_lab > 0):
            stop[line_lab == j] = True
            return True
        return False

    for i in np.flatnonzero(cand):
        p = 9
        sy = slice(max(0, y[i] - p), min(H, y[i] + h[i] + p))
        sx = slice(max(0, x[i] - p), min(W, x[i] + w[i] + p))
        k = (labels[sy, sx] == i).astype(np.uint8)
        ring = (cv2.dilate(k, ellipse(9)) > 0) & ~(cv2.dilate(k, ellipse(3)) > 0) & ~art[sy, sx]
        if ring.sum() < 16:
            continue
        d = dist[sy, sx][ring]
        if (d < 30).mean() < 0.85 or np.percentile(d, 75) > flat_tol:
            continue
        holes = cv2.findContours(k, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0][1:]
        if any(cv2.contourArea(q) > font * font for q in holes):
            continue
        near = line_lab[sy, sx][cv2.dilate(k, ellipse(7)) > 0]
        hit = np.unique(near[near > 0])
        if hit.size:
            own = core_color(crop[sy, sx][k > 0], dist[sy, sx][k > 0])
            if not all(art_line(j, labels == i, own) for j in hit):
                continue
        out[sy, sx] |= k > 0
    # where the text crosses a line of another colour, its pixels lie in the cut
    other = [
        j
        for j, col in enumerate(line_color)
        if col is not None and color_distance(col[None, None], fg)[0, 0] >= 25
    ]
    if other:
        on = cv2.dilate(np.isin(line_lab, other).astype(np.uint8), ellipse(3)) > 0
        on &= box_zone(c) & (color_distance(crop, fg) < 30)
        on &= nearest(crop, [fg, bg, *(line_color[j] for j in other)]) == 0
        text = (out | (erased & ink)).astype(np.uint8)
        out |= on & (cv2.dilate(text, ellipse(17)) > 0)
    if not out.any():
        return out
    grown = cv2.dilate(out.astype(np.uint8), ellipse(grow)) > 0
    if stop.any():
        wall = cv2.dilate(stop.astype(np.uint8), ellipse(3)) > 0
        n, lab = cv2.connectedComponents((grown & ~wall).astype(np.uint8), connectivity=8)
        seed = np.bincount(lab[out & ~wall], minlength=n) > 0
        seed[0] = False
        grown = seed[lab] | out
    return grown


def core_color(px: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Colour of a stroke's core: its pixels furthest from the paper (`d`), not the
    anti-aliased fringe, which is a mix with the paper."""
    return np.median(px[d >= np.percentile(d, 60)], axis=0)


def line_colors(
    crop: np.ndarray, lines: np.ndarray, dist: np.ndarray, c: Crop
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    """Labels of the components of `lines` and the core colour of each, sampled where it
    runs well clear of the box: near the text the line mask also holds the crossing glyph
    strokes, whose colour would be taken for the line's. None for a line with too little of
    it clear of the box."""
    beyond = ~box_zone(c, 0.6)
    n, lab = cv2.connectedComponents(lines.astype(np.uint8), connectivity=8)
    out: list[np.ndarray | None] = [None] * n
    for j in range(1, n):
        sel = (lab == j) & beyond
        if sel.sum() >= 8:
            out[j] = core_color(crop[sel], dist[sel])
    return lab, out


def beside(line: np.ndarray, comp: np.ndarray, font: int, lines: np.ndarray) -> bool:
    """Whether `comp` lies beside the straight `line`, as a glyph touching a line that runs
    past it does, rather than carrying it on: a balloon outline cut where it runs straight
    leaves its curved rest, which goes on from the end of the cut run. The run's extent is
    that of all the `lines` in line with it: a glyph fused with the line may break it."""
    ys, xs = np.nonzero(comp)
    pad = font
    y1, y2 = max(0, ys.min() - pad), ys.max() + pad + 1
    x1, x2 = max(0, xs.min() - pad), xs.max() + pad + 1
    ly, lx = np.nonzero(line[y1:y2, x1:x2])
    if ly.size < 8:
        return False
    pts = np.stack([lx, ly], axis=1).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    ay, ax = np.nonzero(lines[y1:y2, x1:x2])
    run = np.abs((ax - x0) * vy - (ay - y0) * vx) <= max(3.0, 0.1 * font)
    t = (ax[run] - x0) * vx + (ay[run] - y0) * vy
    if t.size < 8:
        return False
    tc = (xs - x1 - x0) * vx + (ys - y1 - y0) * vy
    return float(((tc > t.min() + 2) & (tc < t.max() - 2)).mean()) >= 0.7


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
        glyphs: bool = True,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"mask method must be one of {METHODS}, got {method!r}")
        self.method = method
        self.threshold, self.strong = threshold, strong
        self.reach, self.grow = reach, grow
        self.dilation, self.min_dilate = dilation, min_dilate
        self.ink, self.protect, self.flat_spread = ink, protect, flat_spread
        self.glyphs = glyphs

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
                m = self._from_seg(crop, seg, c, img)
            elif self.method == "seg":
                m = np.zeros(crop.shape[:2], np.uint8)
            else:
                m = self._from_pixels(crop, c, img)
            out[c.sl] |= m
        return out

    def _from_seg(self, crop: np.ndarray, seg: np.ndarray, c: Crop, page: np.ndarray) -> np.ndarray:
        sure = seg >= self.strong * 255
        core = hysteresis(seg >= self.threshold * 255, sure, c, retry=(sure,))
        if not core.any():
            return self._from_pixels(crop, c, page)
        # paper = the few pixels right around the glyphs; glyph colour = confident seg pixels
        bg = ring_median(crop, core, 5, max(9, c.font // 3))
        if bg is None:
            bg, _ = border_background(crop, c.near, c.box)
        body = cv2.erode(sure.astype(np.uint8), ellipse(3)) > 0
        fg = np.median(crop[body if body.sum() >= 8 else core], axis=0)
        return self._complete(core, blend_ink(crop, bg, fg, self.ink), c, sure)

    def _from_pixels(self, crop: np.ndarray, c: Crop, page: np.ndarray) -> np.ndarray:
        found = find_paper(crop, c, self.flat_spread)
        if found is None:
            return busy_box(c)
        bg, dist, spread, cn = found
        zone = dist[cn.near] if cn.near.any() else dist.ravel()
        otsu, _ = cv2.threshold(
            zone.astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        t = float(np.clip(otsu, max(30.0, 2 * spread), 160.0))
        sure = (dist > t) & cn.near
        core = hysteresis(dist > 0.6 * t, sure, cn, retry=tuple(dist > k * t for k in (1, 2, 3)))
        fg = glyph_color(crop, sure, core, c, bg)
        extra = stuck = None
        if self.glyphs:
            extra, stuck = lettering(crop, c, bg, fg, max(c.font, cn.font))
        if not core.any() and (extra is None or not extra.any()):
            m = np.zeros(crop.shape[:2], np.uint8)
        else:
            m = self._complete(core, blend_ink(crop, bg, fg, self.ink), cn, extra=extra)
        if stuck is not None and stuck.any():
            m |= stuck_fill(stuck, m > 0, c)
        if self.glyphs:
            font = max(c.font, cn.font)
            d = max(self.min_dilate, round(font * self.dilation) * 2 + 1)
            left = leftover_glyphs(page, c, bg, fg, spread, m > 0, font, d)
            m |= left.astype(np.uint8) * 255
        return m

    def _complete(
        self,
        core: np.ndarray,
        ink: np.ndarray,
        c: Crop,
        sure: np.ndarray | None = None,
        extra: np.ndarray | None = None,
    ) -> np.ndarray:
        """Grow the core into touching ink (fringe, thin strokes, dots), dilate by font size,
        and cut out balloon outlines / art lines that the dilation reached. `sure` = pixels the
        detector calls text; colour alone cannot tell a glyph from the outline it touches.
        `extra` = glyphs (and their outline rings) found one by one: no structure."""
        zone = cv2.dilate(core.astype(np.uint8), ellipse(2 * round(c.font * self.grow) + 1)) > 0
        cand = (ink & zone) | core
        n, labels = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
        touch = np.bincount(labels[core], minlength=n) > 0
        touch[0] = False
        m = touch[labels]
        if extra is not None:  # complete already; `ink` may be the wrong colour for them
            m |= extra
        guard = None
        if self.protect:
            # a glyph found on its own is no structure, nor is its fringe; an art line that
            # touched it through the fringe still is
            free = ink
            if extra is not None:
                free = ink & ~(cv2.dilate(extra.astype(np.uint8), ellipse(5)) > 0)
            guard = structure(free, c)
            if c.has_lines and sure is not None:
                # confident text pixels on a real text line still go: a glyph touching the outline
                k = ellipse(2 * (c.font // 10) + 1)
                guard &= ~(sure & (cv2.dilate(c.lines.astype(np.uint8), k) > 0))
            m &= ~guard  # before dilating, or the outline's own dilation hugs it as a ring
        d = max(self.min_dilate, round(c.font * self.dilation) * 2 + 1)
        m = cv2.dilate(m.astype(np.uint8), ellipse(d)) > 0
        if guard is not None:
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
        glyphs: bool = True,
    ) -> None:
        self.threshold, self.reach = threshold, reach
        self.dilation, self.min_dilate = dilation, min_dilate
        self.flat_spread, self.glyphs = flat_spread, glyphs

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
            box = left = None
            if raw is not None:
                m = raw[c.sl] >= self.threshold * 255
                m &= c.near
            else:
                crop = img[c.sl]
                found = find_paper(crop, c, self.flat_spread)
                if found is None:
                    out[c.sl] |= busy_box(c)
                    continue
                bg, dist, spread, cn = found
                t = max(40.0, 2 * spread)
                # ink that runs out of the window is art or a balloon outline, not text
                m = hysteresis(dist > t, cn.near, cn, retry=(dist > 2 * t,))
                m &= cn.near
                if self.glyphs:  # see GlyphMask._from_pixels
                    sure = m & (dist > t)
                    fg = np.median(crop[sure], axis=0) if sure.any() else text_color(crop, c, bg)
                    extra, stuck = lettering(crop, c, bg, fg, max(c.font, cn.font))
                    if stuck.any():
                        box = stuck_fill(stuck, m | extra, c)
                    m |= extra
                    erased = m if box is None else m | (box > 0)
                    font = max(c.font, cn.font)
                    d = max(self.min_dilate, round(cn.font * self.dilation) * 2 + 1)
                    left = leftover_glyphs(img, c, bg, fg, spread, erased, font, d)
                c = cn
            d = max(self.min_dilate, round(c.font * self.dilation) * 2 + 1)
            out[c.sl] |= (cv2.dilate(m.astype(np.uint8), ellipse(d)) > 0).astype(np.uint8) * 255
            if left is not None:
                out[c.sl] |= left.astype(np.uint8) * 255
            if box is not None:
                out[c.sl] |= box
        return out


DEFAULT_OPTS = (
    "threshold", "strong", "reach", "grow", "dilation", "min_dilate", "ink", "protect",
    "flat_spread", "method", "glyphs",
)  # fmt: skip
DILATE_OPTS = ("threshold", "reach", "dilation", "min_dilate", "flat_spread", "glyphs")


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
    glyphs: bool = True,
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
        _bool(glyphs),
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
    glyphs: bool = True,
    **extra,
):
    _unknown("dilate", extra, DILATE_OPTS)
    return DilateMask(
        float(threshold),
        float(reach),
        float(dilation),
        int(min_dilate),
        float(flat_spread),
        _bool(glyphs),
    )


def _bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
