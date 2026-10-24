"""Merge text-line quads into text blocks (port of manga_translator/textline_merge).

Lines that are geometrically compatible (close, similar font size and orientation) are linked
in a graph; each connected component is then split recursively along the longest edge of its
minimum spanning tree while that edge is an outlier. Pure numpy/OpenCV, no networkx/shapely.
Any detector that yields line quads can call `merge_lines` + `group_to_region`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import cv2
import numpy as np

from ..models import BBox, TextRegion

Direction = Literal["h", "v"]


def sort_points(pts: np.ndarray) -> tuple[np.ndarray, bool]:
    """Order 4 points clockwise from top-left and tell whether the quad is a vertical line."""
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    vec = (pts[:, None] - pts[None]).reshape(16, 2)
    norms = np.linalg.norm(vec, axis=1)
    long_sides = vec[np.argsort(norms, kind="stable")[[8, 10]]].copy()
    if (long_sides[0] * long_sides[1]).sum() < 0:
        long_sides[0] = -long_sides[0]
    struct = np.abs(long_sides.mean(axis=0))
    vertical = bool(struct[0] <= struct[1])
    if vertical:
        pts = pts[np.argsort(pts[:, 1], kind="stable")]
        top, bottom = pts[:2], pts[2:]
        pts = np.concatenate([top[np.argsort(top[:, 0])], bottom[np.argsort(bottom[:, 0])[::-1]]])
    else:
        pts = pts[np.argsort(pts[:, 0], kind="stable")]
        left, right = pts[:2], pts[2:]
        left, right = left[np.argsort(left[:, 1])], right[np.argsort(right[:, 1])]
        pts = np.array([left[0], right[0], right[1], left[1]])
    return pts, vertical


@dataclass(eq=False)
class Line:
    """One text line (a column for vertical text) as a quad."""

    pts: np.ndarray
    score: float = 1.0
    fg: tuple[int, int, int] | None = None
    bg: tuple[int, int, int] | None = None
    direction: Direction = field(init=False)

    def __post_init__(self) -> None:
        self.pts, vertical = sort_points(self.pts)
        self.direction = "v" if vertical else "h"

    @classmethod
    def from_box(cls, x1: float, y1: float, x2: float, y2: float, **kw) -> Line:
        return cls(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64), **kw)

    @cached_property
    def structure(self) -> tuple[np.ndarray, ...]:
        p = self.pts
        return ((p[0] + p[1]) / 2, (p[2] + p[3]) / 2, (p[1] + p[2]) / 2, (p[3] + p[0]) / 2)

    @cached_property
    def _axes(self) -> tuple[np.ndarray, np.ndarray]:
        a, b, c, d = self.structure
        return b - a, d - c

    @cached_property
    def font_size(self) -> float:
        v1, v2 = self._axes
        return float(max(min(np.linalg.norm(v1), np.linalg.norm(v2)), 1.0))

    @cached_property
    def aspect_ratio(self) -> float:
        """horizontal / vertical extent."""
        v1, v2 = self._axes
        return float(np.linalg.norm(v2) / max(np.linalg.norm(v1), 1e-6))

    @cached_property
    def angle(self) -> float:
        v1, _ = self._axes
        cos = v1[0] / max(np.linalg.norm(v1), 1e-6)
        return float(math.fmod(math.acos(float(np.clip(cos, -1, 1))) + math.pi, math.pi))

    @cached_property
    def centroid(self) -> np.ndarray:
        return self.pts.mean(axis=0)

    @cached_property
    def aabb(self) -> tuple[float, float, float, float]:
        x1, y1 = self.pts.min(axis=0)
        x2, y2 = self.pts.max(axis=0)
        return float(x1), float(y1), float(x2 - x1), float(y2 - y1)

    @cached_property
    def area(self) -> float:
        return hull_area(self.pts)

    @cached_property
    def approx_axis_aligned(self) -> bool:
        for v in self._axes:
            u = v / max(np.linalg.norm(v), 1e-6)
            if abs(u[0]) < 0.05 or abs(u[1]) < 0.05:
                return True
        return False


def hull_area(pts: np.ndarray) -> float:
    return float(cv2.contourArea(cv2.convexHull(np.asarray(pts, dtype=np.float32))))


def _seg_point_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p, q, r) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def _inside(p: np.ndarray, poly: np.ndarray) -> bool:
    return (
        cv2.pointPolygonTest(poly.astype(np.float32).reshape(-1, 1, 2), tuple(map(float, p)), False)
        >= 0
    )


def poly_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two convex quads (0 when they touch or overlap)."""
    if any(_inside(p, b) for p in a) or any(_inside(p, a) for p in b):
        return 0.0
    ea = [(a[i], a[(i + 1) % 4]) for i in range(4)]
    eb = [(b[i], b[(i + 1) % 4]) for i in range(4)]
    if any(_segments_cross(p, q, r, s) for p, q in ea for r, s in eb):
        return 0.0
    d = min(_seg_point_dist(p, r, s) for p in a for r, s in eb)
    return min(d, min(_seg_point_dist(p, r, s) for p in b for r, s in ea))


def can_merge(
    a: Line,
    b: Line,
    ratio: float = 1.9,
    discard_connection_gap: float = 2,
    char_gap_tolerance: float = 1,
    char_gap_tolerance2: float = 3,
    font_size_ratio_tol: float = 2,
    aspect_ratio_tol: float = 1.3,
) -> bool:
    """Whether two lines may belong to the same block (`quadrilateral_can_merge_region`)."""
    char_size = min(a.font_size, b.font_size)
    x1, y1, w1, h1 = a.aabb
    x2, y2, w2, h2 = b.aabb
    dist = poly_distance(a.pts, b.pts)
    if dist > discard_connection_gap * char_size:
        return False
    if max(a.font_size, b.font_size) / char_size > font_size_ratio_tol:
        return False
    if a.aspect_ratio > aspect_ratio_tol and b.aspect_ratio < 1 / aspect_ratio_tol:
        return False
    if b.aspect_ratio > aspect_ratio_tol and a.aspect_ratio < 1 / aspect_ratio_tol:
        return False
    if a.approx_axis_aligned and b.approx_axis_aligned:
        if dist >= char_size * char_gap_tolerance:
            return False
        if abs(x1 + w1 // 2 - (x2 + w2 // 2)) < char_gap_tolerance2:
            return True
        if (w1 > h1 * ratio and h2 > w2 * ratio) or (w2 > h2 * ratio and h1 > w1 * ratio):
            return False
        tol = char_size * char_gap_tolerance2
        if w1 > h1 * ratio or w2 > h2 * ratio:
            return abs(x1 - x2) < tol or abs(x1 + w1 - (x2 + w2)) < tol
        if h1 > w1 * ratio or h2 > w2 * ratio:
            return abs(y1 - y2) < tol or abs(y1 + h1 - (y2 + h2)) < tol
        return False
    if abs(a.angle - b.angle) < 15 * math.pi / 180:
        fs = min(a.font_size, b.font_size)
        if dist > fs * char_gap_tolerance2:
            return False
        return abs(a.font_size - b.font_size) / fs <= 0.25
    return False


def line_distance(a: Line, b: Line, rho: float = 0.5) -> float:
    """Anchor distance between two lines (`Quadrilateral.distance_impl`): picks the alignment
    (start, end or middle) the two lines share and measures between those anchors."""
    fs = max(a.font_size, b.font_size)
    pa, pb = a.pts, b.pts
    if a.direction == "h":
        d1 = hull_area(np.array([pa[0], pa[3], pb[0], pb[3]])) / fs
        d2 = hull_area(np.array([pa[2], pa[1], pb[2], pb[1]])) / fs
        sa, sb = a.structure, b.structure
        d3 = hull_area(np.array([sa[0], sa[1], sb[0], sb[1]])) / fs
        pattern = "left"
        if d1 < fs * rho:
            pattern = "left"
        if d2 < fs * rho and d2 < d1:
            pattern = "right"
        if d3 < fs * rho and d3 < d1 and d3 < d2:
            pattern = "middle"
        if pattern == "left":
            return float(np.linalg.norm(pa[0] - pb[0]))
        if pattern == "right":
            return float(np.linalg.norm(pa[1] - pb[1]))
        return float(np.linalg.norm(sa[0] - sb[0]))
    d1 = hull_area(np.array([pa[0], pa[1], pb[0], pb[1]])) / fs
    d2 = hull_area(np.array([pa[2], pa[3], pb[2], pb[3]])) / fs
    if d2 < fs * rho and d2 < d1:
        return float(np.linalg.norm(pa[2] - pb[2]))
    return float(np.linalg.norm(pa[0] - pb[0]))


class _DSU:
    def __init__(self, items: Sequence[int]) -> None:
        self.parent = {i: i for i in items}

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = {}
        for i in self.parent:
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())


def _split(
    lines: Sequence[Line], idx: list[int], gamma: float = 0.5, sigma: float = 2
) -> list[list[int]]:
    """Recursively split a connected group at outlier MST edges (`split_text_region`)."""
    if len(idx) == 1:
        return [idx]
    if len(idx) == 2:
        a, b = lines[idx[0]], lines[idx[1]]
        fs = max(a.font_size, b.font_size)
        if line_distance(a, b) < (1 + gamma) * fs and abs(a.angle - b.angle) < 0.2 * math.pi:
            return [idx]
        return [[idx[0]], [idx[1]]]

    pairs = sorted(
        (
            (line_distance(lines[u], lines[v]), u, v)
            for i, u in enumerate(idx)
            for v in idx[i + 1 :]
        ),
    )
    dsu = _DSU(idx)
    mst = [(w, u, v) for w, u, v in pairs if dsu.union(u, v)]
    mst.sort(key=lambda e: -e[0])
    weights = np.array([w for w, _, _ in mst])
    fontsize = float(np.mean([lines[i].font_size for i in idx]))
    std, mean = float(weights.std()), float(weights.mean())
    std_threshold = max(0.3 * fontsize + 5, 5)
    _, u, v = mst[0]
    b1, b2 = lines[u], lines[v]
    gap = poly_distance(b1.pts, b2.pts)
    alignment = float(np.min(np.abs(b1.centroid - b2.centroid)))
    if (weights[0] <= mean + std * sigma or weights[0] <= fontsize * (1 + gamma)) and (
        std < std_threshold or (gap == 0 and alignment < 5)
    ):
        return [idx]
    dsu = _DSU(idx)
    for _, a, b in mst[1:]:
        dsu.union(a, b)
    out: list[list[int]] = []
    for g in dsu.groups():
        out.extend(_split(lines, sorted(g)))
    return out


def majority_direction(lines: Sequence[Line]) -> Direction:
    top = Counter(ln.direction for ln in lines).most_common(2)
    if len(top) == 1 or top[0][1] != top[1][1]:
        return top[0][0]
    best = max(lines, key=lambda ln: max(ln.aspect_ratio, 1 / max(ln.aspect_ratio, 1e-6)))
    return best.direction


def merge_lines(lines: Sequence[Line], **merge_opts) -> list[tuple[list[int], Direction]]:
    """Group line indices into blocks; each group is sorted in reading order of its direction
    (vertical: right-to-left columns, horizontal: top-to-bottom rows)."""
    n = len(lines)
    dsu = _DSU(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if can_merge(lines[i], lines[j], **merge_opts):
                dsu.union(i, j)
    blocks: list[tuple[list[int], Direction]] = []
    for comp in dsu.groups():
        for g in _split(lines, sorted(comp)):
            members = [lines[i] for i in g]
            d = majority_direction(members)
            key = (
                (lambda i: -lines[i].centroid[0]) if d == "v" else (lambda i: lines[i].centroid[1])
            )
            blocks.append((sorted(g, key=key), d))
    return blocks


def group_to_region(lines: Sequence[Line], group: list[int], direction: Direction) -> TextRegion:
    """Build a TextRegion (bbox, line quads, font size, angle, colours) from merged lines."""
    members = [lines[i] for i in group]
    pts = np.concatenate([ln.pts for ln in members])
    x1, y1 = np.floor(pts.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
    areas = np.array([max(ln.area, 1.0) for ln in members])
    scores = np.clip([ln.score for ln in members], 1e-6, 1.0)
    score = float(np.exp((np.log(scores) * areas).sum() / areas.sum()))
    angle = float(np.rad2deg(np.mean([ln.angle for ln in members])) - 90)
    if abs(angle) < 3:
        angle = 0.0

    def mean_colour(attr: str) -> tuple[int, int, int] | None:
        cols = [getattr(ln, attr) for ln in members if getattr(ln, attr) is not None]
        return tuple(round(c) for c in np.mean(cols, axis=0)) if cols else None  # type: ignore[return-value]

    return TextRegion(
        id=0,
        bbox=BBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)),
        score=score,
        lines=[[(round(x), round(y)) for x, y in ln.pts] for ln in members],
        direction=direction,
        angle=angle,
        font_size=round(min(ln.font_size for ln in members)),
        fg_color=mean_colour("fg"),
        bg_color=mean_colour("bg"),
    )
