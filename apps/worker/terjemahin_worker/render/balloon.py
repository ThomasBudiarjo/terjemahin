"""Find the speech balloon around a text region, to typeset into its real interior.

The reference (manga-image-translator rendering/ballon_extractor.py) flood-fills Canny edges.
We grow a colour-similarity region instead, seeded by the text box on the *inpainted* page (the
glyphs are gone there, so the interior is one flat colour) and walled by ink from the original
page: pixels close to the interior colour, connected to the text box, after a small opening that
cuts leaks through 1-2 px gaps in the outline. A region that runs into the search window's edge
(not the page edge, which may legitimately crop a balloon) is not a closed balloon: the text sits
over art or a borderless background, and the caller falls back to the text box.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..models import BBox

COLOR_TOL = 14.0  # Lab distance from the interior colour that still counts as interior
INK_TOL = 60  # original-page pixels this much darker than the interior are outline/art
MAX_AREA_RATIO = 30.0  # balloon / text-box area beyond which it is a background, not a balloon
MIN_COVER = 0.6  # share of the text box the balloon must contain
WINDOW_SCALES = (0.9, 2.0, 3.5)  # search padding, x the box's longer side


@dataclass
class Balloon:
    mask: np.ndarray  # uint8 0/255, crop coordinates
    x0: int
    y0: int
    closed: bool
    color: tuple[int, int, int]  # interior colour, BGR

    @property
    def bbox(self) -> BBox:
        ys, xs = np.nonzero(self.mask)
        if not len(xs):
            return BBox(x1=self.x0, y1=self.y0, x2=self.x0 + 1, y2=self.y0 + 1)
        return BBox(
            x1=self.x0 + int(xs.min()),
            y1=self.y0 + int(ys.min()),
            x2=self.x0 + int(xs.max()) + 1,
            y2=self.y0 + int(ys.max()) + 1,
        )

    def page_slice(self) -> tuple[slice, slice]:
        h, w = self.mask.shape
        return slice(self.y0, self.y0 + h), slice(self.x0, self.x0 + w)


def _window(b: BBox, W: int, H: int, scale: float) -> BBox:
    pad = int(max(b.w, b.h) * scale) + 24
    return BBox(x1=b.x1 - pad, y1=b.y1 - pad, x2=b.x2 + pad, y2=b.y2 + pad).clamp(W, H)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    ff[1:-1, 1:-1] = mask
    cv2.floodFill(ff, None, (0, 0), 128)
    return np.where(ff[1:-1, 1:-1] == 128, 0, 255).astype(np.uint8)


def find_balloon(
    inpainted_bgr: np.ndarray,
    original_bgr: np.ndarray,
    bbox: BBox,
    color_tol: float = COLOR_TOL,
) -> Balloon:
    """Balloon around `bbox`; the search window grows when a short text sits in a big balloon."""
    H, W = inpainted_bgr.shape[:2]
    for scale in WINDOW_SCALES:
        bal, touches = _extract(
            inpainted_bgr, original_bgr, bbox, _window(bbox, W, H, scale), color_tol
        )
        if not touches:
            break
    return bal


def _extract(
    inpainted_bgr: np.ndarray,
    original_bgr: np.ndarray,
    bbox: BBox,
    win: BBox,
    color_tol: float,
) -> tuple[Balloon, bool]:
    H, W = inpainted_bgr.shape[:2]
    crop = inpainted_bgr[win.y1 : win.y2, win.x1 : win.x2]
    orig = original_bgr[win.y1 : win.y2, win.x1 : win.x2]
    bx1, by1 = bbox.x1 - win.x1, bbox.y1 - win.y1
    bx2, by2 = bx1 + bbox.w, by1 + bbox.h
    box = np.zeros(crop.shape[:2], np.uint8)
    box[max(0, by1) : by2, max(0, bx1) : bx2] = 255

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = lab[box > 0]
    ref = np.median(inside, axis=0)
    color = tuple(int(v) for v in np.median(crop[box > 0], axis=0))
    dist = np.linalg.norm(lab - ref, axis=2)
    gray_o = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.int16)
    ink = gray_o < ref[0] - INK_TOL  # OpenCV 8-bit L is 0..255 like gray
    cand = ((dist < color_tol) & ~(ink & (box == 0))).astype(np.uint8) * 255
    k = max(3, round(min(bbox.w, bbox.h, 60) / 12) | 1)
    cand = cv2.morphologyEx(
        cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )

    # the interior is the colour-similar component that overlaps the text box most; glyph holes
    # (when the page was not erased) are filled below
    _, labels = cv2.connectedComponents(cand, connectivity=4)
    ids, counts = np.unique(labels[box > 0], return_counts=True)
    counts[ids == 0] = 0
    if counts.max() == 0:
        return Balloon(box, win.x1, win.y1, False, color), False  # type: ignore[arg-type]
    region = (labels == ids[int(np.argmax(counts))]).astype(np.uint8) * 255
    region = _fill_holes(region)
    region = cv2.morphologyEx(
        region,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1)),
    )

    touches = (
        (win.x1 > 0 and region[:, 0].any())
        or (win.y1 > 0 and region[0, :].any())
        or (win.x2 < W and region[:, -1].any())
        or (win.y2 < H and region[-1, :].any())
    )
    area = int((region > 0).sum())
    cover = float((region[box > 0] > 0).mean())
    closed = not touches and area <= MAX_AREA_RATIO * max(1, bbox.w * bbox.h) and cover >= MIN_COVER
    return Balloon(region, win.x1, win.y1, closed, color), touches  # type: ignore[arg-type]


def split_shared(balloons: list[Balloon], boxes: list[BBox]) -> None:
    """When two text regions found the same balloon, give each the half nearer its own box.

    Mutates the masks in place, so two translations never get laid out on top of each other.
    """

    def dist_to_box(b: Balloon, box: BBox) -> np.ndarray:
        h, w = b.mask.shape
        xs = np.arange(w) + b.x0
        ys = np.arange(h) + b.y0
        dx = np.maximum(np.maximum(box.x1 - xs, xs - box.x2), 0).astype(np.float32)
        dy = np.maximum(np.maximum(box.y1 - ys, ys - box.y2), 0).astype(np.float32)
        return np.hypot(dx[None, :], dy[:, None])

    for i, bi in enumerate(balloons):
        if not bi.closed:
            continue
        for j, bj in enumerate(balloons):
            if i == j or not bj.closed:
                continue
            a, c = bi.bbox, bj.bbox
            if a.x2 <= c.x1 or c.x2 <= a.x1 or a.y2 <= c.y1 or c.y2 <= a.y1:
                continue
            # does box j lie inside balloon i?
            jb = boxes[j]
            cx, cy = int(jb.cx) - bi.x0, int(jb.cy) - bi.y0
            h, w = bi.mask.shape
            if not (0 <= cx < w and 0 <= cy < h and bi.mask[cy, cx]):
                continue
            nearer_j = dist_to_box(bi, boxes[j]) < dist_to_box(bi, boxes[i])
            bi.mask[nearer_j] = 0
