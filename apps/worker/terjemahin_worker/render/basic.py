from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..clean import estimate_background
from ..models import BBox, Bubble
from ..opts import warn_unknown
from ..registry import RENDERERS
from ..typeset import Typesetter


class BasicRenderer:
    """v1 typesetter: horizontal, word-wrapped, centred in the detector box."""

    def __init__(self, font_path) -> None:
        self.typesetter = Typesetter(font_path)

    def render(self, inpainted_bgr, bubbles: list[Bubble], original_bgr) -> np.ndarray:
        H, W = inpainted_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB))
        for b in bubbles:
            if not b.target_text.strip():
                continue
            box = _on_page(b.region.bbox, W, H)  # laid out where it is visible, not clipped
            if box is None:
                continue
            bgr = estimate_background(inpainted_bgr, box)
            self.typesetter.render(pil, box, b.target_text, (bgr[2], bgr[1], bgr[0]))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _on_page(b: BBox, W: int, H: int) -> BBox | None:
    x1, x2 = sorted((b.x1, b.x2))
    y1, y2 = sorted((b.y1, b.y2))
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


@RENDERERS.register("basic", description="Horizontal centred text in the detector box (v1).")
def _build(settings, font_path=None, **extra):
    warn_unknown("renderer 'basic'", extra)
    if font_path and not Path(font_path).is_file():
        raise ValueError(f"renderer font_path: no such font file: {font_path}")
    return BasicRenderer(font_path or settings.font_path)
