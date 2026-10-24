from __future__ import annotations

from typing import Protocol

import numpy as np

from ..models import TextRegion


class MaskRefiner(Protocol):
    def refine(
        self, image_bgr: np.ndarray, regions: list[TextRegion], raw_mask: np.ndarray | None
    ) -> np.ndarray:
        """Return the HxW uint8 mask (0 or 255) of pixels to erase for the given regions.

        `raw_mask` is the detector's text-pixel mask when it has one; refiners must still work
        without it (e.g. after a hand edit, compose() runs without the detector).
        """
        ...
