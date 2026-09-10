from __future__ import annotations

import cv2
import numpy as np

from ..models import BBox, TextRegion


def letterbox(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to a square canvas of `size`.

    Returns (rgb float32 NCHW tensor, scale, (pad_x, pad_y)).
    """
    h, w = image_bgr.shape[:2]
    scale = size / max(h, w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None], scale, (pad_x, pad_y)


def xywh_to_regions(
    boxes_xywh: np.ndarray,
    scores: np.ndarray,
    scale: float,
    pad: tuple[int, int],
    orig_wh: tuple[int, int],
    conf_thres: float,
    iou_thres: float,
) -> list[TextRegion]:
    """Filter by confidence, NMS, un-letterbox, and return regions in original pixel coords."""
    keep = scores >= conf_thres
    boxes_xywh, scores = boxes_xywh[keep], scores[keep]
    if len(boxes_xywh) == 0:
        return []

    cx, cy, w, h = boxes_xywh.T
    x1 = (cx - w / 2 - pad[0]) / scale
    y1 = (cy - h / 2 - pad[1]) / scale
    bw, bh = w / scale, h / scale

    rects = [(float(a), float(b), float(c), float(d)) for a, b, c, d in zip(x1, y1, bw, bh)]
    idx = cv2.dnn.NMSBoxes(rects, scores.astype(np.float32).tolist(), conf_thres, iou_thres)
    if len(idx) == 0:
        return []

    W, H = orig_wh
    regions = []
    for i in np.asarray(idx).flatten():
        bx, by, bw_, bh_ = rects[i]
        bbox = BBox(x1=round(bx), y1=round(by), x2=round(bx + bw_), y2=round(by + bh_)).clamp(W, H)
        if bbox.w < 4 or bbox.h < 4:
            continue
        regions.append(TextRegion(id=len(regions), bbox=bbox, score=float(scores[i])))
    return regions
