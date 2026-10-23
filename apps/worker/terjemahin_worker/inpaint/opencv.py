"""Classical diffusion inpainting (cv2.inpaint): no weights, fine on flat balloons and thin
strokes, smears on screentone and large holes. Runs per hole cluster on a padded crop because
cv2.inpaint's cost grows with the image it is handed, not just with the hole."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..registry import INPAINTERS


class CvInpainter:
    def __init__(self, flag: int, radius: int = 5, merge: int = 32) -> None:
        self.flag, self.radius, self.merge = flag, radius, merge

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hole = ((mask.max(axis=2) if mask.ndim == 3 else mask) > 0).astype(np.uint8)
        out = image_bgr.copy()
        if not hole.any():
            return out
        grow = cv2.dilate(hole, np.ones((self.merge, self.merge), np.uint8)) if self.merge else hole
        stats = cv2.connectedComponentsWithStats(grow, connectivity=8)[2]
        H, W = hole.shape
        pad = 2 * self.radius + 2
        for x, y, w, h, _ in stats[1:]:
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
            m = hole[y1:y2, x1:x2]
            res = cv2.inpaint(image_bgr[y1:y2, x1:x2], m * 255, self.radius, self.flag)
            sel = m.astype(bool)
            out[y1:y2, x1:x2][sel] = res[sel]
        return out


def _opts(o: dict[str, Any]) -> dict[str, Any]:
    return {k: int(v) for k, v in o.items() if k in ("radius", "merge")}


@INPAINTERS.register("telea", description="OpenCV Telea fast-marching inpaint (no weights, fast).")
def _build_telea(settings, **opts: Any) -> CvInpainter:
    return CvInpainter(cv2.INPAINT_TELEA, **_opts(opts))


@INPAINTERS.register("ns", description="OpenCV Navier-Stokes inpaint (no weights, fast).")
def _build_ns(settings, **opts: Any) -> CvInpainter:
    return CvInpainter(cv2.INPAINT_NS, **_opts(opts))
