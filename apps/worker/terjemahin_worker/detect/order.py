"""Panel-aware reading order.

Panels are found by recursive XY-cuts. A row (column) is gutter when nearly every pixel on it
is flat along the row and gutter-toned, or sits on a long straight frame edge (so a borderless
panel that touches its neighbour's frame still splits); a band of such rows becomes a cut only
if it runs along a frame edge, which tells a real gutter from a flat patch of sky inside a
panel. Balloons and text boxes are ignored because they routinely cross gutters, and a small
fraction of blocked pixels is tolerated for art that breaks out of its frame. Horizontal cuts
come first (manga tiers), except on a two-page spread and where full-height gutters cut the
page into tall 4-koma strips: there the vertical cut comes first. Pieces of a vertical cut are
read right-to-left (`rtl=False` for western comics). Inside a panel, blocks whose vertical spans overlap form a row read in the
same horizontal direction; rows go top to bottom.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence

import cv2
import numpy as np

from ..models import BBox, TextRegion

log = logging.getLogger(__name__)

WORK_SIZE = 1024  # long side used for panel analysis


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(np.int8), [0]])))
    return list(zip(idx[::2].tolist(), idx[1::2].tolist()))


class _Maps:
    """Per-pixel cut evidence at work resolution."""

    def __init__(self, gray: np.ndarray, ignore: np.ndarray | None) -> None:
        h, w = gray.shape
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        gx = np.abs(cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3))
        k = max(2, min(h, w) // 300)
        frame = np.concatenate(
            [blur[:k].ravel(), blur[-k:].ravel(), blur[:, :k].ravel(), blur[:, -k:].ravel()]
        )
        level = float(np.median(frame))
        # gutters share the page margin's tone; a flat black coat is not a gutter
        tone = (blur >= level - 70) if level >= 128 else (blur <= level + 70)
        edge_x = cv2.morphologyEx(
            (gx > 60).astype(np.uint8), cv2.MORPH_OPEN, np.ones((max(8, h // 10), 1), np.uint8)
        )
        edge_y = cv2.morphologyEx(
            (gy > 60).astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, max(8, w // 10)), np.uint8)
        )
        self.hedge = edge_y > 0  # long straight horizontal edges (frames)
        self.vedge = edge_x > 0
        self.row_sep = ((gx < 40) & tone) | (cv2.dilate(edge_y, np.ones((5, 9), np.uint8)) > 0)
        self.col_sep = ((gy < 40) & tone) | (cv2.dilate(edge_x, np.ones((9, 5), np.uint8)) > 0)
        if ignore is not None:
            ign = cv2.dilate(ignore, np.ones((3, 3), np.uint8)) > 0
            self.row_sep |= ign
            self.col_sep |= ign


def _cut_points(
    sep: np.ndarray, edge: np.ndarray, min_size: int, tol: float, support: float
) -> list[int]:
    """Cut positions along axis 0 of `sep` (rows): the middle of each accepted gutter band
    that has at least `min_size` of content on both sides."""
    n, span = sep.shape
    gutter = (~sep).sum(axis=1) <= 2 + tol * span
    cover = edge.sum(axis=1)
    keep = np.ones(n, bool)
    for a, b in _runs(gutter):
        lo, hi = max(0, a - 3), min(n, b + 3)
        if a == 0 or b == n or cover[lo:hi].max() >= support * span:
            keep[a:b] = False
    pieces = [(a, b) for a, b in _runs(keep) if b - a >= min_size]
    return [(pieces[i][1] + pieces[i + 1][0]) // 2 for i in range(len(pieces) - 1)]


def _xy_cut(
    maps: _Maps,
    box: tuple[int, int, int, int],
    rtl: bool,
    min_size: int,
    out: list[tuple[int, int, int, int]],
    depth: int = 0,
) -> None:
    x1, y1, x2, y2 = box
    found = {}
    for axis in (0, 1):  # 0: horizontal cuts (tiers), 1: vertical cuts (columns)
        if axis == 0:
            sep, edge, lo, hi = maps.row_sep[y1:y2, x1:x2], maps.hedge[y1:y2, x1:x2], y1, y2
        else:
            sep, edge, lo, hi = maps.col_sep[y1:y2, x1:x2].T, maps.vedge[y1:y2, x1:x2].T, x1, x2
        found[axis] = (_cut_points(sep, edge, min_size, tol=0.12, support=0.4), lo, hi)
    order = (0, 1)
    vcuts, vlo, vhi = found[1]
    w = x2 - x1
    if vcuts and depth == 0 and w >= SPREAD_ASPECT * (y2 - y1):
        # two pages: cut at the gap between them only, each page is cut on its own
        mid = min(vcuts, key=lambda c: abs(c - w / 2))
        if abs(mid - w / 2) <= 0.15 * w:
            found[1] = ([mid], vlo, vhi)
            order = (1, 0)
    elif vcuts and _strips(maps, vcuts, box):
        order = (1, 0)
    for axis in order:
        cuts, lo, hi = found[axis]
        if cuts and depth < 12:
            bounds = [lo, *(lo + c for c in cuts), hi]
            spans = list(itertools.pairwise(bounds))
            if axis == 1 and rtl:
                spans.reverse()
            for a, b in spans:
                sub = (x1, a, x2, b) if axis == 0 else (a, y1, b, y2)
                _xy_cut(maps, sub, rtl, min_size, out, depth + 1)
            return
    out.append(box)


SPREAD_ASPECT = 1.2  # a page this much wider than tall is two pages: read the right one first
STRIP_ASPECT = 2.0  # columns this tall (h / w) are 4-koma strips: read each top to bottom


def _strips(maps: _Maps, cuts: list[int], box: tuple[int, int, int, int]) -> bool:
    """Whether full-height vertical cuts split the box into tall strips (4-koma columns),
    which are read one after the other, each top to bottom, instead of tier by tier. A strip
    gutter has panel frames along nearly all of its height; a cut that is gutter only because
    it crosses white art in a wide panel below (the panel spans both "strips") does not."""
    x1, y1, x2, y2 = box
    bounds = [0, *cuts, x2 - x1]
    if not all((y2 - y1) >= STRIP_ASPECT * (b - a) for a, b in itertools.pairwise(bounds)):
        return False
    reach = max(10, round(0.03 * (x2 - x1)))
    for c in cuts:
        band = maps.vedge[y1:y2, max(x1, x1 + c - reach) : min(x2, x1 + c + reach)]
        if band.size == 0 or band.any(axis=1).mean() < 0.75:
            return False
    return True


def find_panels(
    image_bgr: np.ndarray,
    rtl: bool = True,
    ignore: np.ndarray | None = None,
    min_panel: float = 0.06,
) -> list[BBox]:
    """Panel cells in reading order; they partition the page (gutters split down the middle).
    `ignore` (HxW, non-zero = skip) masks balloons and text."""
    H, W = image_bgr.shape[:2]
    s = min(1.0, WORK_SIZE / max(H, W))
    small = (
        cv2.resize(image_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if s < 1
        else image_bgr
    )
    gray = small if small.ndim == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    ign = None
    if ignore is not None:
        ign = cv2.resize(
            (ignore > 0).astype(np.uint8), gray.shape[::-1], interpolation=cv2.INTER_NEAREST
        )
    h, w = gray.shape
    boxes: list[tuple[int, int, int, int]] = []
    _xy_cut(_Maps(gray, ign), (0, 0, w, h), rtl, max(8, int(min(h, w) * min_panel)), boxes)
    return [
        BBox(
            x1=int(a / s),
            y1=int(b / s),
            x2=min(W, int(np.ceil(c / s))),
            y2=min(H, int(np.ceil(d / s))),
        )
        for a, b, c, d in boxes
    ]


def balloon_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Filled white balloons (from the classical detector) so gutter-crossing balloons do not
    block cuts."""
    from .classical import ClassicalDetector

    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    for b in ClassicalDetector().balloons(gray):
        bh, bw = b.filled.shape
        roi = mask[b.y : b.y + bh, b.x : b.x + bw]
        np.maximum(roi, b.filled, out=roi)
    return cv2.dilate(mask, np.ones((9, 9), np.uint8))


def _ignore_mask(image_bgr: np.ndarray, regions: Sequence[TextRegion]) -> np.ndarray:
    H, W = image_bgr.shape[:2]
    ignore = balloon_mask(image_bgr)
    for r in regions:
        b = r.bbox.pad(max(4, (r.font_size or 16) // 2)).clamp(W, H)
        ignore[b.y1 : b.y2, b.x1 : b.x2] = 255
    return ignore


def _panel_of(r: TextRegion, panels: Sequence[BBox]) -> int:
    cx, cy = r.bbox.cx, r.bbox.cy
    for i, p in enumerate(panels):
        if p.x1 <= cx < p.x2 and p.y1 <= cy < p.y2:
            return i

    def overlap(p: BBox) -> float:
        return max(0, min(p.x2, r.bbox.x2) - max(p.x1, r.bbox.x1)) * max(
            0, min(p.y2, r.bbox.y2) - max(p.y1, r.bbox.y1)
        )

    def dist(p: BBox) -> float:
        return max(p.x1 - cx, 0, cx - p.x2) ** 2 + max(p.y1 - cy, 0, cy - p.y2) ** 2

    return max(range(len(panels)), key=lambda i: (overlap(panels[i]), -dist(panels[i])))


def order_within(
    regions: Sequence[TextRegion], rtl: bool = True, overlap: float = 0.15
) -> list[TextRegion]:
    """Rows of vertically overlapping blocks, top to bottom; each row right-to-left (rtl)."""
    rows: list[tuple[float, float, list[TextRegion]]] = []
    for r in sorted(regions, key=lambda r: (r.bbox.y1, -r.bbox.cx if rtl else r.bbox.cx)):
        if rows:
            top, bottom, members = rows[-1]
            shared = min(bottom, r.bbox.y2) - max(top, r.bbox.y1)
            if shared > overlap * min(r.bbox.h, bottom - top):
                rows[-1] = (top, max(bottom, r.bbox.y2), members + [r])
                continue
        rows.append((r.bbox.y1, r.bbox.y2, [r]))
    ordered: list[TextRegion] = []
    for _, _, members in rows:
        ordered.extend(sorted(members, key=lambda r: -r.bbox.cx if rtl else r.bbox.cx))
    return ordered


def sort_regions(
    regions: Sequence[TextRegion],
    image_bgr: np.ndarray | None = None,
    rtl: bool = True,
    panels: Sequence[BBox] | None = None,
) -> list[TextRegion]:
    """Reading order; with an image (or explicit panels) blocks are grouped per panel first.
    Returned regions carry new ids 0..n-1 in reading order."""
    regions = list(regions)
    if not regions:
        return []
    if panels is None and image_bgr is not None:
        try:
            panels = find_panels(image_bgr, rtl=rtl, ignore=_ignore_mask(image_bgr, regions))
        except (cv2.error, ValueError, MemoryError):  # ordering must never fail a page
            log.warning("panel detection failed, ordering without panels", exc_info=True)
    if panels:
        groups: dict[int, list[TextRegion]] = {}
        for r in regions:
            groups.setdefault(_panel_of(r, panels), []).append(r)
        ordered = [r for i in sorted(groups) for r in order_within(groups[i], rtl)]
    else:
        ordered = order_within(regions, rtl)
    return [r.model_copy(update={"id": i}) for i, r in enumerate(ordered)]
