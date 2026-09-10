from __future__ import annotations

import numpy as np

from .models import BBox

FILL_PAD = 3


def estimate_background(image_bgr: np.ndarray, bbox: BBox) -> tuple[int, int, int]:
    """Median colour of a 2px ring just outside the box; that ring is bubble interior when the
    detector box hugs the text, which is the common case."""
    H, W = image_bgr.shape[:2]
    outer = bbox.pad(FILL_PAD + 2).clamp(W, H)
    inner = bbox.pad(FILL_PAD).clamp(W, H)
    patch = image_bgr[outer.y1 : outer.y2, outer.x1 : outer.x2].copy()
    ix1, iy1 = inner.x1 - outer.x1, inner.y1 - outer.y1
    ix2, iy2 = ix1 + inner.w, iy1 + inner.h
    mask = np.ones(patch.shape[:2], dtype=bool)
    mask[iy1:iy2, ix1:ix2] = False
    ring = patch[mask]
    if len(ring) == 0:
        ring = patch.reshape(-1, 3)
    med = np.median(ring, axis=0)
    return tuple(int(v) for v in med)


def fill_region(image_bgr: np.ndarray, bbox: BBox) -> tuple[int, int, int]:
    """Erase the original text by painting the padded box with the estimated background.
    Mutates `image_bgr` in place and returns the fill colour (BGR)."""
    H, W = image_bgr.shape[:2]
    color = estimate_background(image_bgr, bbox)
    b = bbox.pad(FILL_PAD).clamp(W, H)
    image_bgr[b.y1 : b.y2, b.x1 : b.x2] = color
    return color
