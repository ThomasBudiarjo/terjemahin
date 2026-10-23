from __future__ import annotations

import numpy as np

from ..models import TextRegion
from ..registry import OCRS
from .base import OcrResult


class NoOcr:
    """Leaves source text empty: use with a vision translator (e.g. gemini) that reads it."""

    def recognize(self, image_bgr: np.ndarray, regions: list[TextRegion]) -> list[OcrResult]:
        return [OcrResult(text="", confidence=0.0) for _ in regions]


@OCRS.register("none", description="No OCR; a vision-capable translator reads the crops itself.")
def _build(settings, **_):
    return NoOcr()
