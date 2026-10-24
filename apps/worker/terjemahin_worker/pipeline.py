from __future__ import annotations

import gc
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Settings
from .detect import sort_reading_order
from .imageio import to_bgr8
from .models import BBox, Bubble, PageContext, PageResult, PipelineConfig, TextRegion
from .registry import REGISTRIES, STAGES, load_all
from .translate import TranslationRequest

log = logging.getLogger(__name__)

COMPOSE_STAGES = ("mask_refiner", "inpainter", "renderer")
RTL_SOURCES = ("ja", "auto")  # manga reads right to left; webtoons, manhua, comics left to right
JAPANESE_OCR = ("manga_ocr", "manga_ocr_hf")  # OCR methods that read Japanese only
MAX_FONT_SIZE = 512  # px; a hand-edited font_size above this only makes the mask refiner crawl


def lang_code(code: str | None) -> str:
    """ISO code of a language setting ('auto' kept); ValueError names the bad value."""
    from .translate import textproc as tp

    if code is None or str(code).strip().lower() == "auto":
        return "auto"
    return tp.iso(str(code))


def _is_japanese(lang: str | None) -> bool:
    try:
        return lang_code(lang) in ("ja", "auto")
    except ValueError:
        return True  # reported by the language check


@dataclass
class PipelineOutput:
    image: np.ndarray
    result: PageResult
    debug: dict[str, np.ndarray] = field(default_factory=dict)
    """Intermediate images: detect, mask, inpainted, final. Filled when debug=True."""


class Pipeline:
    """detect -> OCR -> translate -> mask refine -> inpaint -> render.

    Every stage is looked up in its registry by name, so one Pipeline serves jobs with different
    method choices. Built stage instances are cached by (stage, name, options), at most
    `settings.stage_cache_size` per stage, least recently used dropped first: every distinct
    option set builds a new instance, and an unbounded cache let any API client grow a
    long-lived server by ~1 GB per job until the machine ran out of memory.
    """

    def __init__(self, settings: Settings) -> None:
        load_all()
        self.settings = settings
        self._cache: OrderedDict[tuple[str, str, str], Any] = OrderedDict()
        self._cache_lock = threading.RLock()

    @classmethod
    def from_settings(cls, settings: Settings) -> Pipeline:
        return cls(settings)

    def resolve(self, config: PipelineConfig | None) -> PipelineConfig:
        cfg = (config or PipelineConfig()).model_copy(deep=True)
        if cfg.ocr is None and self.settings.ocr in JAPANESE_OCR and not _is_japanese(cfg.source_lang):
            cfg.ocr = self.settings.ocr_non_japanese  # manga_ocr turns English into fullwidth soup
        for stage in STAGES:
            if getattr(cfg, stage) is None:
                setattr(cfg, stage, getattr(self.settings, stage))
        return cfg

    def stage(self, stage: str, cfg: PipelineConfig) -> Any:
        return self._built(stage, cfg)

    def _built(self, stage: str, cfg: PipelineConfig) -> Any:
        name = getattr(cfg, stage)
        opts = cfg.options.get(stage, {})
        key = (stage, name, repr(sorted(opts.items())))
        with self._cache_lock:
            inst = self._cache.get(key)
            if inst is not None:
                self._cache.move_to_end(key)
                return inst
            log.info("loading %s=%s %s", stage, name, opts or "")
            inst = REGISTRIES[stage].build(name, self.settings, **opts)
            self._cache[key] = inst
            self._evict(stage)
            return inst

    def _evict(self, stage: str) -> None:
        keep = max(1, int(self.settings.stage_cache_size))
        same = [k for k in self._cache if k[0] == stage]
        for k in same[: max(0, len(same) - keep)]:
            inst = self._cache.pop(k)
            log.info("unloading %s=%s %s (stage cache holds %d)", k[0], k[1], k[2], keep)
            close = getattr(inst, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # freeing memory must not fail the job
                    log.exception("closing %s=%s failed", k[0], k[1])
            del inst
        if len(same) > keep:
            gc.collect()

    def check(self, cfg: PipelineConfig, stages: tuple[str, ...] = STAGES) -> None:
        """Fail fast, before any stage runs: unknown methods, bad options (factories validate
        them when building), unsupported languages, and OCR 'none' without a translator that
        reads the image itself. Builds (and caches) the stages it checks."""
        for stage in stages:
            REGISTRIES[stage].get(getattr(cfg, stage))  # unknown name: ValueError with choices
        if "translator" in stages:
            src, dst = lang_code(cfg.source_lang), lang_code(cfg.target_lang)
            if dst == "auto":
                raise ValueError("target language cannot be 'auto'")
            if cfg.ocr in JAPANESE_OCR and src not in ("ja", "auto"):
                log.warning("ocr '%s' reads Japanese only, the source language is '%s' "
                            "(try --ocr 48px)", cfg.ocr, src)
        built = {stage: self.stage(stage, cfg) for stage in stages}
        tr = built.get("translator")
        if tr is not None and cfg.ocr == "none" and not getattr(tr, "vision", False):
            raise ValueError(
                f"ocr 'none' needs a translator that reads the image itself (vision); "
                f"'{cfg.translator}' only translates OCR text, so nothing would be translated"
            )

    def min_confidence(self, cfg: PipelineConfig) -> float:
        v = cfg.min_ocr_confidence
        return float(self.settings.min_ocr_confidence if v is None else v)

    def run(
        self,
        page_bgr: np.ndarray,
        context: PageContext | None = None,
        config: PipelineConfig | None = None,
        debug: bool = False,
    ) -> PipelineOutput:
        cfg = self.resolve(config)
        context = context or PageContext()
        page_bgr = to_bgr8(page_bgr)
        self.check(cfg)
        timings: dict[str, float] = {}
        dbg: dict[str, np.ndarray] = {}

        t = time.perf_counter()
        detection = self.stage("detector", cfg).detect(page_bgr)
        rtl = lang_code(cfg.source_lang) in RTL_SOURCES
        regions = sort_reading_order(detection.regions, image=page_bgr, rtl=rtl)
        detected = regions
        timings["detect"] = time.perf_counter() - t

        t = time.perf_counter()
        ocr_results = self.stage("ocr", cfg).recognize(page_bgr, regions)
        if len(ocr_results) != len(regions):
            raise RuntimeError(f"ocr returned {len(ocr_results)} results for {len(regions)} regions")
        # A read the OCR itself doubts is a hallucination on an empty balloon, art or a rotated
        # page far more often than real text: leave the page untouched there.
        min_conf = self.min_confidence(cfg)
        keep = [
            i for i, o in enumerate(ocr_results)
            if not (o.text.strip() and o.confidence < min_conf)
        ]
        for i in sorted(set(range(len(regions))) - set(keep)):
            log.info(
                "region %d: OCR read %r with confidence %.2f < %.2f, left untouched",
                regions[i].id, ocr_results[i].text, ocr_results[i].confidence, min_conf,
            )
        regions = [regions[i] for i in keep]
        ocr_results = [ocr_results[i] for i in keep]
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

        erase_only = bool(getattr(translator, "erase_only", False))
        bubbles = []
        for r, o, it in zip(regions, ocr_results, items):
            source = it.source or o.text
            if not source.strip():
                continue  # no readable source text: a false positive, leave the page untouched
            erase = bool(it.target.strip()) or erase_only
            if not erase:
                log.warning("region %d: empty translation of %r, original text left in place",
                            r.id, source)
            bubbles.append(
                Bubble(
                    region=r,
                    source_text=source,
                    target_text=it.target,
                    speaker=it.speaker,
                    ocr_confidence=o.confidence,
                    erase=erase,
                )
            )

        image = self.compose(page_bgr, bubbles, cfg, raw_mask=detection.mask, timings=timings, debug=dbg)
        if debug:
            dbg["detect"] = draw_regions(page_bgr, detected)
        H, W = page_bgr.shape[:2]
        log.info(
            "page done: %d regions, %d bubbles | %s",
            len(detected), len(bubbles), ", ".join(f"{k} {v:.2f}s" for k, v in timings.items()),
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
        """Mask -> inpaint -> render. Reused when the user edits a translation by hand.

        A bubble's original text is erased when `bubble.erases()` (see Bubble.erase): a box
        added by hand without source text, e.g. a translator's note over art, is only drawn.
        """
        cfg = self.resolve(config)
        timings = timings if timings is not None else {}
        page_bgr = to_bgr8(page_bgr)
        if raw_mask is not None and raw_mask.shape[:2] != page_bgr.shape[:2]:
            log.warning("raw mask %s does not match the page %s, ignored", raw_mask.shape, page_bgr.shape)
            raw_mask = None
        self.check(cfg, COMPOSE_STAGES)
        bubbles = self.sanitize(page_bgr, bubbles)
        erase = [b.region for b in bubbles if b.erases()]

        t = time.perf_counter()
        mask = self.stage("mask_refiner", cfg).refine(page_bgr, erase, raw_mask)
        timings["mask"] = time.perf_counter() - t

        t = time.perf_counter()
        inpainted = self.stage("inpainter", cfg).inpaint(page_bgr, mask) if mask.any() else page_bgr.copy()
        timings["inpaint"] = time.perf_counter() - t

        t = time.perf_counter()
        render = [b for b in bubbles if b.target_text.strip()]
        out = self.stage("renderer", cfg).render(inpainted, render, page_bgr)
        timings["render"] = time.perf_counter() - t

        if debug is not None:
            debug["mask"] = mask
            debug["inpainted"] = inpainted
            debug["final"] = out
        return out


    @staticmethod
    def sanitize(page_bgr: np.ndarray, bubbles: list[Bubble]) -> list[Bubble]:
        """Bubbles as compose can trust them. A PageResult may be edited by hand (or sent by
        an API client): boxes with swapped corners are put right, boxes are clipped to the
        page (a box wholly off it is dropped with a warning), and font_size is bounded by the
        box: the mask refiner grows its search window and kernels with it, and font_size=5000
        kept a single bubble busy for minutes."""
        H, W = page_bgr.shape[:2]
        out = []
        for b in bubbles:
            r = b.region
            bb = r.bbox
            x1, x2 = sorted((bb.x1, bb.x2))
            y1, y2 = sorted((bb.y1, bb.y2))
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 - x1 < 1 or y2 - y1 < 1:
                log.warning("bubble %s: box %s is off the page %dx%d, skipped", r.id, bb, W, H)
                continue
            upd: dict[str, Any] = {}
            if (x1, y1, x2, y2) != (bb.x1, bb.y1, bb.x2, bb.y2):
                upd["bbox"] = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
            if r.font_size is not None:
                cap = max(4, min(MAX_FONT_SIZE, round(1.5 * min(x2 - x1, y2 - y1))))
                fs = r.font_size if r.font_size > 0 else None
                fs = min(fs, cap) if fs is not None else None
                if fs != r.font_size:
                    upd["font_size"] = fs
            if upd:
                b = b.model_copy(update={"region": r.model_copy(update=upd)})
            out.append(b)
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
