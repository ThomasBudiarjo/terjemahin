from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel

from ..models import RGB, TextRegion


class OcrResult(BaseModel):
    text: str
    confidence: float = 1.0
    fg_color: RGB | None = None  # predicted glyph colour, used by renderers
    bg_color: RGB | None = None  # predicted outline/background colour


class OCR(Protocol):
    def recognize(self, image_bgr: np.ndarray, regions: list[TextRegion]) -> list[OcrResult]:
        """Read the source-language text of each region. Exactly one result per region, in order."""
        ...
