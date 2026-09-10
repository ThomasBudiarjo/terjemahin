from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from PIL import Image

from .clean import fill_region
from .config import Settings
from .detect import Detector, build_detector, sort_reading_order
from .models import Bubble, PageContext, PageResult
from .translate import GeminiTranslator, Translator
from .typeset import Typesetter

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, detector: Detector, translator: Translator, typesetter: Typesetter) -> None:
        self.detector = detector
        self.translator = translator
        self.typesetter = typesetter

    @classmethod
    def from_settings(cls, settings: Settings) -> "Pipeline":
        return cls(
            detector=build_detector(settings),
            translator=GeminiTranslator(settings.gemini_api_key, settings.gemini_model),
            typesetter=Typesetter(settings.font_path),
        )

    def run(self, page_bgr: np.ndarray, context: PageContext) -> tuple[np.ndarray, PageResult]:
        t0 = time.perf_counter()
        regions = sort_reading_order(self.detector.detect(page_bgr))
        t1 = time.perf_counter()
        bubbles = self.translator.translate(page_bgr, regions, context)
        t2 = time.perf_counter()
        out = self.compose(page_bgr, bubbles)
        t3 = time.perf_counter()
        log.info(
            "page done: %d regions | detect %.2fs, translate %.2fs, typeset %.2fs",
            len(regions), t1 - t0, t2 - t1, t3 - t2,
        )
        H, W = page_bgr.shape[:2]
        return out, PageResult(bubbles=bubbles, width=W, height=H)

    def compose(self, page_bgr: np.ndarray, bubbles: list[Bubble]) -> np.ndarray:
        """Clean + typeset only. Reused when the user edits a translation by hand."""
        canvas = page_bgr.copy()
        fills: list[tuple[Bubble, tuple[int, int, int]]] = []
        for b in bubbles:
            if not b.target_text:
                continue
            bgr = fill_region(canvas, b.region.bbox)
            fills.append((b, (bgr[2], bgr[1], bgr[0])))

        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        for b, rgb in fills:
            self.typesetter.render(pil, b.region.bbox, b.target_text, rgb)
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
