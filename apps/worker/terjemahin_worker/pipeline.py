from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Settings
from .detect import sort_reading_order
from .models import Bubble, PageContext, PageResult, PipelineConfig, TextRegion
from .registry import REGISTRIES, STAGES, load_all
from .translate import TranslationRequest

log = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    image: np.ndarray
    result: PageResult
    debug: dict[str, np.ndarray] = field(default_factory=dict)
    """Intermediate images: detect, mask, inpainted, final. Filled when debug=True."""


class Pipeline:
    """detect -> OCR -> translate -> mask refine -> inpaint -> render.

    Every stage is looked up in its registry by name, so one Pipeline serves jobs with different
    method choices. Built stage instances are cached by (stage, name, options).
    """

    def __init__(self, settings: Settings) -> None:
        load_all()
        self.settings = settings
        self._cache: dict[tuple[str, str, str], Any] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> "Pipeline":
        return cls(settings)

    def resolve(self, config: PipelineConfig | None) -> PipelineConfig:
        cfg = (config or PipelineConfig()).model_copy(deep=True)
        for stage in STAGES:
            if getattr(cfg, stage) is None:
                setattr(cfg, stage, getattr(self.settings, stage))
        return cfg

    def stage(self, stage: str, cfg: PipelineConfig) -> Any:
        name = getattr(cfg, stage)
        opts = cfg.options.get(stage, {})
        key = (stage, name, repr(sorted(opts.items())))
        if key not in self._cache:
            log.info("loading %s=%s %s", stage, name, opts or "")
            self._cache[key] = REGISTRIES[stage].build(name, self.settings, **opts)
        return self._cache[key]

    def run(
        self,
        page_bgr: np.ndarray,
        context: PageContext | None = None,
        config: PipelineConfig | None = None,
        debug: bool = False,
    ) -> PipelineOutput:
        cfg = self.resolve(config)
        context = context or PageContext()
        timings: dict[str, float] = {}
        dbg: dict[str, np.ndarray] = {}

        t = time.perf_counter()
        detection = self.stage("detector", cfg).detect(page_bgr)
        regions = sort_reading_order(detection.regions)
        timings["detect"] = time.perf_counter() - t

        t = time.perf_counter()
        ocr_results = self.stage("ocr", cfg).recognize(page_bgr, regions)
        regions = [
            r.model_copy(
                update={
                    "fg_color": o.fg_color or r.fg_color,
                    "bg_color": o.bg_color or r.bg_color,
                }
            )
            for r, o in zip(regions, ocr_results)
        ]
        timings["ocr"] = time.perf_counter() - t

        t = time.perf_counter()
        translator = self.stage("translator", cfg)
        items = translator.translate(
            TranslationRequest(
                page_bgr=page_bgr,
                regions=regions,
                sources=[o.text for o in ocr_results],
                context=context,
                source_lang=cfg.source_lang,
                target_lang=cfg.target_lang,
            )
        )
        if len(items) != len(regions):
            raise RuntimeError(f"translator returned {len(items)} items for {len(regions)} regions")
        timings["translate"] = time.perf_counter() - t

        bubbles = [
            Bubble(
                region=r,
                source_text=it.source or o.text,
                target_text=it.target,
                speaker=it.speaker,
                ocr_confidence=o.confidence,
            )
            for r, o, it in zip(regions, ocr_results, items)
        ]
        # A region with no readable source text is a false positive: leave the page untouched.
        bubbles = [b for b in bubbles if b.source_text.strip()]

        image = self.compose(page_bgr, bubbles, cfg, raw_mask=detection.mask, timings=timings, debug=dbg)
        if debug:
            dbg["detect"] = draw_regions(page_bgr, regions)
        H, W = page_bgr.shape[:2]
        log.info(
            "page done: %d regions, %d bubbles | %s",
            len(regions), len(bubbles), ", ".join(f"{k} {v:.2f}s" for k, v in timings.items()),
        )
        result = PageResult(bubbles=bubbles, width=W, height=H, pipeline=cfg, timings=timings)
        return PipelineOutput(image=image, result=result, debug=dbg if debug else {})

    def compose(
        self,
        page_bgr: np.ndarray,
        bubbles: list[Bubble],
        config: PipelineConfig | None = None,
        raw_mask: np.ndarray | None = None,
        timings: dict[str, float] | None = None,
        debug: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Mask -> inpaint -> render. Reused when the user edits a translation by hand."""
        cfg = self.resolve(config)
        timings = timings if timings is not None else {}
        erase = [b.region for b in bubbles if b.source_text.strip() or b.target_text.strip()]

        t = time.perf_counter()
        mask = self.stage("mask_refiner", cfg).refine(page_bgr, erase, raw_mask)
        timings["mask"] = time.perf_counter() - t

        t = time.perf_counter()
        inpainted = self.stage("inpainter", cfg).inpaint(page_bgr, mask) if mask.any() else page_bgr.copy()
        timings["inpaint"] = time.perf_counter() - t

        t = time.perf_counter()
        out = self.stage("renderer", cfg).render(inpainted, [b for b in bubbles if b.target_text], page_bgr)
        timings["render"] = time.perf_counter() - t

        if debug is not None:
            debug["mask"] = mask
            debug["inpainted"] = inpainted
            debug["final"] = out
        return out


def draw_regions(page_bgr: np.ndarray, regions: list[TextRegion]) -> np.ndarray:
    import cv2

    img = page_bgr.copy()
    for r in regions:
        b = r.bbox
        cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), 2)
        for line in r.lines:
            cv2.polylines(img, [np.array(line, np.int32)], True, (0, 200, 0), 1)
        cv2.putText(img, str(r.id), (b.x1, max(12, b.y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return img
