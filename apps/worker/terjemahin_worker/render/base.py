from __future__ import annotations

from typing import Protocol

import numpy as np

from ..models import Bubble


class Renderer(Protocol):
    def render(
        self, inpainted_bgr: np.ndarray, bubbles: list[Bubble], original_bgr: np.ndarray
    ) -> np.ndarray:
        """Draw each bubble's target_text onto the inpainted page and return the result.

        `original_bgr` is the untouched page, for colour sampling and balloon detection.
        """
        ...
