from __future__ import annotations

from typing import Protocol

import numpy as np


class Inpainter(Protocol):
    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return a new BGR image with the mask>0 pixels reconstructed. Same size as input."""
        ...
