"""Post-processing for comic-text-detector: YOLO blocks, DBNet text lines, line grouping.

Pure numpy/OpenCV so it is unit-testable without the model. Coordinates are in "work" space
(the page scaled by the letterbox ratio) until the detector maps them back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MIN_LINE_SIDE = 2.0
ANGLE_SNAP = 3.0  # degrees; below this a block is treated as upright


@dataclass
class Block:
    xyxy: np.ndarray  # float (4,)
    score: float
    lines: list[np.ndarray] = field(default_factory=list)  # each (4, 2) float, tl tr br bl
    line_scores: list[float] = field(default_factory=list)
    vertical: bool = True
    angle: float = 0.0
    font_size: float = 0.0
    from_lines: bool = False  # created from DB lines that no YOLO block claimed
    lang: int = 1  # YOLO class: 0 latin, 1 japanese


def decode_blocks(
    blk: np.ndarray, conf: float, iou: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """YOLOv5 rows [cx, cy, w, h, obj, cls_eng, cls_ja] -> NMSed (xyxy, score, cls)."""
    if blk.ndim == 3:
        blk = blk[0]
    scores = blk[:, 4] * blk[:, 5:].max(axis=1)
    keep = scores > conf
    blk, scores = blk[keep], scores[keep]
    if not len(blk):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    xywh = blk[:, :4].copy()
    xywh[:, :2] -= xywh[:, 2:] / 2
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, iou)
    idx = np.asarray(idx, np.int64).reshape(-1)
    xyxy = np.concatenate([xywh[idx, :2], xywh[idx, :2] + xywh[idx, 2:]], axis=1)
    return xyxy.astype(np.float32), scores[idx].astype(np.float32), blk[idx, 5:].argmax(1)


def nms_xyxy(xyxy: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    if not len(xyxy):
        return np.zeros(0, np.int64)
    xywh = np.concatenate([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]], axis=1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, iou)
    return np.asarray(idx, np.int64).reshape(-1)


def order_quad(rect: tuple) -> np.ndarray:
    """cv2.minAreaRect -> 4 corners clockwise from top-left."""
    pts = sorted(cv2.boxPoints(rect).tolist(), key=lambda p: p[0])
    left = sorted(pts[:2], key=lambda p: p[1])
    right = sorted(pts[2:], key=lambda p: p[1])
    return np.array([left[0], right[0], right[1], left[1]], np.float32)


def db_lines(
    prob: np.ndarray, thresh: float = 0.3, box_thresh: float = 0.6, unclip: float = 1.5,
    max_candidates: int = 1000,
) -> tuple[list[np.ndarray], list[float]]:
    """DBNet shrink map -> text-line quads.

    The shrink map marks a line's core; unclipping a rectangle by `area * ratio / perimeter`
    restores its extent. For a rectangle the Vatti offset is just w+2d, h+2d, so no polygon
    clipping library is needed.
    """
    bitmap = (prob > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads, scores = [], []
    for c in contours[:max_candidates]:
        (cx, cy), (w, h), a = cv2.minAreaRect(c)
        if min(w, h) < MIN_LINE_SIDE:
            continue
        score = contour_score(prob, c.reshape(-1, 2))
        if score < box_thresh:
            continue
        d = w * h * unclip / (2 * (w + h))
        quads.append(order_quad(((cx, cy), (w + 2 * d, h + 2 * d), a)))
        scores.append(score)
    return quads, scores


def contour_score(prob: np.ndarray, pts: np.ndarray) -> float:
    """Mean probability inside the contour (DBNet box_score_fast)."""
    h, w = prob.shape
    x0, y0 = np.clip(pts.min(0), 0, [w - 1, h - 1]).astype(int)
    x1, y1 = np.clip(pts.max(0), 0, [w - 1, h - 1]).astype(int)
    m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(m, [(pts - [x0, y0]).astype(np.int32)], 1)
    return float(cv2.mean(prob[y0 : y1 + 1, x0 : x1 + 1], m)[0])


def quad_xyxy(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q.min(0), q.max(0)])


def line_geometry(q: np.ndarray) -> tuple[bool, float, float, float]:
    """(vertical, angle_deg, length, thickness) of one line quad.

    Angle is the counter-clockwise rotation of the text relative to upright, so a vertical
    column whose bottom leans right and a horizontal line whose right end rises are both > 0.
    """
    top, right = (q[1] - q[0]), (q[3] - q[0])  # along top edge, down left edge
    wlen, hlen = float(np.hypot(*top)), float(np.hypot(*right))
    if hlen >= wlen:
        v = (q[3] + q[2] - q[0] - q[1]) / 2
        return True, float(np.degrees(np.arctan2(v[0], v[1]))), hlen, wlen
    v = (q[1] + q[2] - q[0] - q[3]) / 2
    return False, float(-np.degrees(np.arctan2(v[1], v[0]))), wlen, hlen


def _runs(profile: np.ndarray, min_count: float) -> list[tuple[int, int]]:
    on = np.concatenate([[False], profile > min_count, [False]])
    d = np.flatnonzero(np.diff(on.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(d[::2], d[1::2]) if b - a >= 2]


def mask_lines(
    mask: np.ndarray, xyxy: np.ndarray, vertical: bool | None = None, thresh: float = 0.3
) -> list[np.ndarray]:
    """Text lines from seg-mask projection profiles inside a box.

    Fallback for when the DB head missed a block's lines or fused neighbouring columns (it
    happens at low resolution). With `vertical=None` the orientation giving more elongated
    lines wins, columns on a tie: Japanese is mostly vertical.
    """
    h, w = mask.shape
    x0, y0 = np.clip(np.floor(xyxy[:2]).astype(int), 0, [w - 1, h - 1])
    x1, y1 = np.clip(np.ceil(xyxy[2:]).astype(int), [x0 + 1, y0 + 1], [w, h])
    crop = mask[y0:y1, x0:x1] > thresh
    if crop.sum() < 4:
        return []
    best: list[np.ndarray] = []
    best_score = -1
    for vert in (True, False) if vertical is None else (vertical,):
        prof = crop.sum(0 if vert else 1)
        quads = []
        for a, b in _runs(prof, max(1.0, 0.02 * crop.shape[0 if vert else 1])):
            band = crop[:, a:b] if vert else crop[a:b]
            idx = np.flatnonzero(band.any(1 if vert else 0))
            lo, hi = idx[0], idx[-1] + 1
            qx0, qy0, qx1, qy1 = (a, lo, b, hi) if vert else (lo, a, hi, b)
            quads.append(
                np.array([[qx0, qy0], [qx1, qy0], [qx1, qy1], [qx0, qy1]], np.float32)
                + [x0, y0]
            )
        score = sum(1 for q in quads if line_geometry(q)[0] == vert)
        if score > best_score:
            best, best_score = quads, score
    return best


def split_fat_lines(blk: Block, mask: np.ndarray, factor: float = 1.6) -> None:
    """Re-cut DB lines much thicker than their siblings (fused columns) from the mask profile."""
    if len(blk.lines) < 2:
        return
    geo = [line_geometry(q) for q in blk.lines]
    vertical = sum(g[0] for g in geo) * 2 >= len(geo)
    thick = float(np.median([g[3] for g in geo]))
    lines, scores = [], []
    for q, g, s in zip(blk.lines, geo, blk.line_scores or [blk.score] * len(geo)):
        if abs(g[1]) < ANGLE_SNAP and g[3] > factor * thick:
            parts = mask_lines(mask, quad_xyxy(q), vertical)
            if len(parts) > 1:
                lines += parts
                scores += [s] * len(parts)
                continue
        lines.append(q)
        scores.append(s)
    blk.lines, blk.line_scores = lines, scores


def examine(blk: Block) -> None:
    """Direction, angle, font size and line order of a block from its lines."""
    if not blk.lines:
        x0, y0, x1, y1 = blk.xyxy
        blk.vertical = (y1 - y0) >= (x1 - x0)
        blk.font_size = float(min(x1 - x0, y1 - y0))
        return
    geo = [line_geometry(q) for q in blk.lines]
    vlen = sum(g[2] for g in geo if g[0])
    hlen = sum(g[2] for g in geo if not g[0])
    if len(blk.lines) == 1 and abs(vlen - hlen) < 1e-6:
        vlen += 1
    blk.vertical = vlen >= hlen
    same = [g for g in geo if g[0] == blk.vertical] or geo
    blk.font_size = float(np.median([g[3] for g in same]))
    angle = float(np.average([g[1] for g in same], weights=[g[2] for g in same]))
    blk.angle = 0.0 if abs(angle) < ANGLE_SNAP else round(angle, 1)
    order = np.argsort(
        [-q[:, 0].mean() if blk.vertical else q[:, 1].mean() for q in blk.lines], kind="stable"
    )
    blk.lines = [blk.lines[i] for i in order]
    blk.line_scores = [blk.line_scores[i] for i in order] if blk.line_scores else []


def intersect_area(a: np.ndarray, b: np.ndarray) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return float(max(w, 0) * max(h, 0))


def mask_score(mask: np.ndarray | None, xyxy: np.ndarray) -> float:
    if mask is None:
        return 1.0
    h, w = mask.shape
    x0, y0 = np.clip(np.floor(xyxy[:2]).astype(int), 0, [w - 1, h - 1])
    x1, y1 = np.clip(np.ceil(xyxy[2:]).astype(int), [x0 + 1, y0 + 1], [w, h])
    return float(mask[y0:y1, x0:x1].mean())


def split_block(blk: Block, gap: float) -> list[Block]:
    """Split a block whose consecutive lines are more than `gap * font_size` apart.

    The YOLO head sometimes boxes two neighbouring balloons together; their columns are
    separated by far more than normal line spacing.
    """
    if gap <= 0 or len(blk.lines) < 2:
        return [blk]
    axis = 0 if blk.vertical else 1
    groups: list[list[int]] = [[0]]
    for i in range(1, len(blk.lines)):
        prev, cur = blk.lines[i - 1], blk.lines[i]
        if blk.vertical:  # lines ordered right to left
            space = prev[:, 0].min() - cur[:, 0].max()
        else:
            space = cur[:, 1].min() - prev[:, 1].max()
        cross = 1 - axis
        overlap = min(prev[:, cross].max(), cur[:, cross].max()) - max(
            prev[:, cross].min(), cur[:, cross].min()
        )
        if space > gap * blk.font_size or overlap < -blk.font_size:
            groups.append([i])
        else:
            groups[-1].append(i)
    if len(groups) == 1:
        return [blk]
    out = []
    for g in groups:
        lines = [blk.lines[i] for i in g]
        b = Block(
            np.concatenate([np.min([q.min(0) for q in lines], 0), np.max([q.max(0) for q in lines], 0)]),
            blk.score,
            lines,
            [blk.line_scores[i] for i in g] if blk.line_scores else [],
        )
        examine(b)
        out.append(b)
    return out


def merge_scattered(blocks: list[Block]) -> list[Block]:
    """Union single-line blocks that continue each other (same direction, similar size)."""
    n = len(blocks)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = blocks[i], blocks[j]
            if a.vertical != b.vertical:
                continue
            fs = max(a.font_size, b.font_size)
            if min(a.font_size, b.font_size) < fs / 1.6:
                continue
            ax, bx = a.xyxy, b.xyxy
            gx = max(ax[0], bx[0]) - min(ax[2], bx[2])
            gy = max(ax[1], bx[1]) - min(ax[3], bx[3])
            near = (gx < fs * 0.8 and gy < -fs) if a.vertical else (gy < fs * 0.8 and gx < -fs)
            if near:
                parent[find(j)] = find(i)
    groups: dict[int, list[Block]] = {}
    for i, b in enumerate(blocks):
        groups.setdefault(find(i), []).append(b)
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append(g[0])
            continue
        lines = [q for b in g for q in b.lines]
        scores = [s for b in g for s in b.line_scores]
        m = Block(
            np.concatenate([np.min([b.xyxy[:2] for b in g], 0), np.max([b.xyxy[2:] for b in g], 0)]),
            float(np.mean([b.score for b in g])),
            lines,
            scores,
            from_lines=True,
        )
        examine(m)
        out.append(m)
    return out


def group_lines(
    xyxy: np.ndarray,
    scores: np.ndarray,
    lines: list[np.ndarray],
    line_scores: list[float],
    mask: np.ndarray | None = None,
    assign_thresh: float = 0.4,
    mask_thresh: float = 0.1,
    split_gap: float = 2.0,
    keep_scattered: bool = True,
    classes: np.ndarray | None = None,
) -> list[Block]:
    """Assign DB lines to YOLO blocks (port of comic-text-detector's group_output).

    A line joins the block covering most of its box; lines nobody claims become new blocks if
    the segmentation mask confirms text there; blocks without lines survive only with mask
    support and get their own box as a single line. `mask` is the seg probability (0..1).
    """
    if classes is None:
        classes = np.ones(len(xyxy), np.int64)
    blocks = [
        Block(b.astype(np.float32), float(s), lang=int(c)) for b, s, c in zip(xyxy, scores, classes)
    ]
    scattered: list[Block] = []
    for q, s in zip(lines, line_scores):
        lb = quad_xyxy(q)
        area = max(float((lb[2] - lb[0]) * (lb[3] - lb[1])), 1.0)
        best, best_i = 0.0, -1
        for i, b in enumerate(blocks):
            barea = max(float((b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1])), 1.0)
            r = intersect_area(b.xyxy, lb) / min(area, barea)  # line in block or block in line
            if r > best:
                best, best_i = r, i
        if best > assign_thresh:
            blocks[best_i].lines.append(q)
            blocks[best_i].line_scores.append(s)
        elif keep_scattered and mask_score(mask, lb) >= mask_thresh:
            nb = Block(lb.astype(np.float32), s, [q], [s], from_lines=True)
            examine(nb)
            scattered.append(nb)

    out: list[Block] = []
    for b in blocks:
        if not b.lines:
            if mask_score(mask, b.xyxy) < mask_thresh:
                continue
            # without DB lines a text grid reads equally well as rows or columns, so the
            # orientation comes from the YOLO language class and the box shape
            w, h = b.xyxy[2:] - b.xyxy[:2]
            vertical = h * 2.5 > w if b.lang == 1 else h > w * 2.5
            b.lines = mask_lines(mask, b.xyxy, vertical) if mask is not None else []
            if not b.lines:
                x0, y0, x1, y1 = b.xyxy
                b.lines = [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)]
            b.line_scores = [b.score] * len(b.lines)
        elif mask is not None:
            split_fat_lines(b, mask)
        examine(b)
        out.extend(split_block(b, split_gap))
    out.extend(merge_scattered(scattered))
    for b in out:  # YOLO boxes are loose (they often poke out of the balloon); lines are tight
        b.xyxy = np.concatenate(
            [np.min([q.min(0) for q in b.lines], 0), np.max([q.max(0) for q in b.lines], 0)]
        )
    return out
