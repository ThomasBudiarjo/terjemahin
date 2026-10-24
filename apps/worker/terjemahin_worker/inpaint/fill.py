from __future__ import annotations

import cv2
import numpy as np

from ..registry import INPAINTERS


class FillInpainter:
    """Paint each masked connected component with the median colour of the ring around it.
    Instant and perfect on flat white bubbles; smears on screentone or art."""

    def __init__(self, ring: int = 3) -> None:
        self.ring = ring

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = image_bgr.copy()
        n, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.ring + 1, 2 * self.ring + 1))
        for i in range(1, n):
            comp = (labels == i).astype(np.uint8)
            ring = cv2.dilate(comp, k) & (mask == 0)
            px = image_bgr[ring.astype(bool)]
            color = np.median(px, axis=0) if len(px) else np.array([255, 255, 255])
            out[comp.astype(bool)] = color.astype(np.uint8)
        return out


@INPAINTERS.register("fill", description="Flat fill with the surrounding median colour (fast).")
def _build(settings, ring: int = 3, **extra):
    from ..opts import as_int, warn_unknown

    warn_unknown("inpainter 'fill'", extra)
    return FillInpainter(as_int(ring, "inpainter ring", 1, 100))
