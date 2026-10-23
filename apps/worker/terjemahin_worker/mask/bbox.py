from __future__ import annotations

import numpy as np

from ..models import TextRegion
from ..registry import MASK_REFINERS


class BoxMask:
    """Whole (padded) box: crude but never leaves glyph remnants inside the box."""

    def __init__(self, pad: int = 3) -> None:
        self.pad = pad

    def refine(self, image_bgr, regions: list[TextRegion], raw_mask) -> np.ndarray:
        H, W = image_bgr.shape[:2]
        mask = np.zeros((H, W), np.uint8)
        for r in regions:
            b = r.bbox.pad(self.pad).clamp(W, H)
            mask[b.y1 : b.y2, b.x1 : b.x2] = 255
        return mask


@MASK_REFINERS.register("bbox", description="Erase the whole padded text box.")
def _build(settings, pad: int = 3, **_):
    return BoxMask(pad)
