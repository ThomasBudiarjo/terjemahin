"""Weight-free text detector for manga: bright balloon interiors that contain glyph strokes.

Manga dialogue lives in white balloons, so the pipeline is: threshold bright pixels, keep
compact closed components (holes filled = balloon), take the dark strokes strictly inside each
balloon as glyphs, split them into columns/rows with projection profiles, and merge those
lines into blocks with `merge.merge_lines`. Text drawn directly on artwork is not found; use a
learned detector for that.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..models import TextRegion
from ..registry import DETECTORS
from .base import Detection
from .merge import Line, group_to_region, merge_lines


@dataclass
class Balloon:
    x: int
    y: int
    filled: np.ndarray  # uint8 crop mask, holes filled
    interior: np.ndarray  # filled eroded away from the outline


def _runs(profile: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Runs of non-zero bins, joining runs separated by fewer than `min_gap` empty bins."""
    on = np.flatnonzero(profile)
    if on.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(on) > min_gap)
    starts = np.concatenate([[on[0]], on[breaks + 1]])
    ends = np.concatenate([on[breaks], [on[-1]]]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _unit(gray: np.ndarray) -> float:
    """Pixel scale relative to a ~1500 px wide page, so kernels track the scan resolution."""
    return min(gray.shape) / 1500


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ff = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    return (padded | cv2.bitwise_not(flood))[1:-1, 1:-1]


class ClassicalDetector:
    def __init__(
        self,
        bright: int = 215,
        dark: int = 140,
        min_balloon: float = 0.0008,
        max_balloon: float = 0.12,
        min_solidity: float = 0.8,
        min_font: int = 8,  # px at a 1500 px wide page
    ) -> None:
        from ..opts import as_float, as_int

        self.bright = as_int(bright, "detector bright", 0, 255)
        self.dark = as_int(dark, "detector dark", 0, 255)
        self.min_balloon = as_float(min_balloon, "detector min_balloon", 0.0, 1.0)
        self.max_balloon = as_float(max_balloon, "detector max_balloon", 0.0, 1.0)
        self.min_solidity = as_float(min_solidity, "detector min_solidity", 0.0, 1.0)
        self.min_font = as_int(min_font, "detector min_font", 1)

    def balloons(self, gray: np.ndarray) -> list[Balloon]:
        h, w = gray.shape
        page = h * w
        bright = (gray >= self.bright).astype(np.uint8) * 255
        # cut thin bright bridges (gutter lines, speed lines) that touch balloon interiors
        k = max(3, round(5 * _unit(gray)))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=4)
        out: list[Balloon] = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if bw * bh < self.min_balloon * page or area > self.max_balloon * page:
                continue
            if max(bw, bh) > 6 * min(bw, bh):
                continue
            comp = (labels[y : y + bh, x : x + bw] == i).astype(np.uint8) * 255
            filled = _fill_holes(comp)
            fill_area = int(np.count_nonzero(filled))
            cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea)))
            if hull <= 0 or fill_area / hull < self.min_solidity:
                continue
            # nothing inside: plain white area; mostly holes: a gutter frame or outline, not paper
            if fill_area > self.max_balloon * page or not 0.5 <= area / fill_area <= 0.99:
                continue
            k = max(3, round(min(bw, bh) * 0.03) | 1)
            interior = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            out.append(Balloon(int(x), int(y), filled, interior))
        return out

    def lines_in(
        self, image_bgr: np.ndarray, gray: np.ndarray, b: Balloon
    ) -> tuple[list[Line], np.ndarray]:
        """Line quads (page coords) and glyph mask (crop coords) of one balloon."""
        bh, bw = b.filled.shape
        crop = gray[b.y : b.y + bh, b.x : b.x + bw]
        inside = b.interior > 0
        ink = ((crop < self.dark) & inside).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        keep = np.zeros(n, bool)
        big = 0.9 * max(bw, bh)
        keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 4) & (
            np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) <= big
        )
        glyph = keep[labels].astype(np.uint8)
        ink_px = int(glyph.sum())
        if ink_px < 20 or ink_px > 0.5 * max(int(inside.sum()), 1) or keep.sum() < 2:
            return [], glyph
        vert = self._split(glyph, axis=0)
        horz = self._split(glyph, axis=1)
        boxes = vert if self._elongation(vert, 0) >= self._elongation(horz, 1) else horz
        min_font = max(4, round(self.min_font * _unit(gray)))
        boxes = [
            (x1, y1, x2, y2)
            for x1, y1, x2, y2 in boxes
            if min(x2 - x1, y2 - y1) >= min_font and 0.06 <= glyph[y1:y2, x1:x2].mean() <= 0.7
        ]
        if not boxes:
            return [], glyph
        # several lines, one as thick as the blob itself: art on a white patch, not lettering
        thickest = max(min(x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes)
        if thickest > 0.6 * min(bw, bh) and len(boxes) > 1:
            return [], glyph
        # antialiased glyph edges belong to the text too: grow into the non-white neighbourhood
        soft = cv2.dilate(glyph, np.ones((3, 3), np.uint8), iterations=3) & (crop < 245) & inside
        pix = image_bgr[b.y : b.y + bh, b.x : b.x + bw]
        fg = tuple(int(v) for v in np.median(pix[glyph > 0], axis=0)[::-1])
        paper = inside & (soft == 0)
        bg = (
            tuple(int(v) for v in np.median(pix[paper], axis=0)[::-1])
            if paper.any()
            else (255, 255, 255)
        )
        lines = [
            Line.from_box(x1 + b.x, y1 + b.y, x2 + b.x, y2 + b.y, fg=fg, bg=bg)
            for x1, y1, x2, y2 in boxes
        ]
        return lines, soft | glyph

    @staticmethod
    def _split(glyph: np.ndarray, axis: int) -> list[tuple[int, int, int, int]]:
        """Columns (axis=0) or rows (axis=1) from projection gaps, each split again where its
        own run of ink has a gap larger than the line thickness."""
        prof = glyph.sum(axis=axis) > 0
        boxes = []
        for a, b in _runs(prof, 1):
            strip = glyph[:, a:b] if axis == 0 else glyph[a:b, :]
            thick = b - a
            along = strip.sum(axis=1 - axis) > 0
            for c, d in _runs(along, max(2, int(thick * 1.2))):
                sub = strip[c:d] if axis == 0 else strip[:, c:d]
                across = np.flatnonzero(sub.sum(axis=axis) > 0)
                lo, hi = a + across[0], a + across[-1] + 1
                boxes.append((lo, c, hi, d) if axis == 0 else (c, lo, d, hi))
        return boxes

    @staticmethod
    def _elongation(boxes: list[tuple[int, int, int, int]], axis: int) -> float:
        if not boxes:
            return 0.0
        ratios = [
            ((y2 - y1) / max(x2 - x1, 1)) if axis == 0 else ((x2 - x1) / max(y2 - y1, 1))
            for x1, y1, x2, y2 in boxes
        ]
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        return float(np.average(np.minimum(ratios, 8), weights=areas))

    def detect(self, image_bgr: np.ndarray) -> Detection:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, np.uint8)
        regions: list[TextRegion] = []
        for b in self.balloons(gray):
            lines, glyph = self.lines_in(image_bgr, gray, b)
            if not lines:
                continue
            bh, bw = glyph.shape
            roi = mask[b.y : b.y + bh, b.x : b.x + bw]
            np.maximum(roi, glyph * 255, out=roi)
            # merge per balloon: neighbouring balloons never share a block
            for idx, direction in merge_lines(lines):
                region = group_to_region(lines, idx, direction)
                regions.append(region.model_copy(update={"id": len(regions), "kind": "bubble"}))
        return Detection(regions=regions, mask=mask)


@DETECTORS.register(
    "classical",
    description="OpenCV only, no weights: white balloons + glyph projection + line merge. "
    "Misses text outside balloons.",
)
def _build(settings, **opts) -> ClassicalDetector:
    return ClassicalDetector(**opts)
