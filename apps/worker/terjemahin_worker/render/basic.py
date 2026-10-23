from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..clean import estimate_background
from ..models import Bubble
from ..registry import RENDERERS
from ..typeset import Typesetter


class BasicRenderer:
    """v1 typesetter: horizontal, word-wrapped, centred in the detector box."""

    def __init__(self, font_path) -> None:
        self.typesetter = Typesetter(font_path)

    def render(self, inpainted_bgr, bubbles: list[Bubble], original_bgr) -> np.ndarray:
        pil = Image.fromarray(cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB))
        for b in bubbles:
            if not b.target_text:
                continue
            bgr = estimate_background(inpainted_bgr, b.region.bbox)
            self.typesetter.render(pil, b.region.bbox, b.target_text, (bgr[2], bgr[1], bgr[0]))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


@RENDERERS.register("basic", description="Horizontal centred text in the detector box (v1).")
def _build(settings, font_path=None, **_):
    return BasicRenderer(font_path or settings.font_path)
