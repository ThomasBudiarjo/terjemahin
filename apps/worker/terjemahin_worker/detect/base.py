from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..models import TextRegion


@dataclass
class Detection:
    regions: list[TextRegion]
    mask: np.ndarray | None = None
    """Optional HxW uint8 text-pixel mask (0..255) at original resolution. Mask refiners use it
    to erase only glyph pixels instead of whole boxes."""


class Detector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> Detection:
        """Return text regions (and optionally a text mask). IDs are assigned by the caller."""
        ...


def sort_reading_order(
    regions: list[TextRegion],
    row_tolerance: float = 0.5,
    image: np.ndarray | None = None,
    rtl: bool = True,
) -> list[TextRegion]:
    """Reading order (manga: right-to-left, top-to-bottom); ids are reassigned 0..n-1.

    With the page image, panels are detected first and blocks are ordered panel by panel
    (see order.py); without it the whole page is treated as one panel. `row_tolerance` is
    kept for compatibility only: rows are now formed by vertical overlap.
    """
    from .order import sort_regions

    return sort_regions(regions, image, rtl=rtl)
