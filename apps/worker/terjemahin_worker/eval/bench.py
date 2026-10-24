"""End-to-end benchmark: run the real Pipeline on a page and score every stage against GT.

`--matrix` varies one stage at a time over every available registered method, others at their
defaults. Expensive stage calls (detect, OCR, translate, inpaint) are memoised on their inputs,
so a renderer variant does not re-run the detector; reported latencies are always the measured
time of the real call, never of a cache hit.

The judges are independent of the methods under test: inpainting is re-read by the baseline
detector + OCR, rendered text by the 48px OCR on a blank canvas of the drawn ink, and
translation meaning by a multilingual sentence-embedding model (eval/semantic.py).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel

from ..config import Settings
from ..models import Bubble, PipelineConfig, TextRegion
from ..pipeline import Pipeline, draw_regions
from ..registry import REGISTRIES, STAGES
from . import metrics as M

log = logging.getLogger(__name__)

CALL = {
    "detector": "detect",
    "ocr": "recognize",
    "translator": "translate",
    "mask_refiner": "refine",
    "inpainter": "inpaint",
    "renderer": "render",
}
TIMING_KEY = {
    "detector": "detect",
    "ocr": "ocr",
    "translator": "translate",
    "mask_refiner": "mask",
    "inpainter": "inpaint",
    "renderer": "render",
}
MEMO_STAGES = {"detector", "ocr", "translator", "inpainter"}
READER_OCR = "48px"  # reads rendered Latin text back: trained on mixed ja/en lines
CJK_TARGETS = {"ja", "zh", "ko"}
HEAVY_DOWNLOAD = re.compile(r"downloads? ~?\d", re.IGNORECASE)
NEEDS_SERVER = {"translator:ollama": "needs a running Ollama server (OLLAMA_BASE_URL)"}

# pass/fail thresholds of the checks in the report
THRESHOLDS = {
    "detect_recall": 12 / 13,
    "false_positives": 0,
    "ocr_cer": 0.15,
    "chrf": 25.0,
    "semantic_sim": 0.8,  # a bubble's translation counts as adequate at this cosine or above
    "adequate_frac": 0.7,  # share of GT bubbles that must be adequately translated
    "residual_regions": 0,
    "dark_residue": 0.01,
    "ghost_residue": 0.01,
    "outside_mad": 0.5,
    "outside_far_changed": 0,
    "outside_ring_px": 8,  # changes this close to the mask are edge feathering (0 = strict)
    "inside_frac": 0.99,
    "read_cer_mean": 0.1,  # rendered text read back vs the translation, letters only
    "read_cer_max": 0.34,
    "min_font_px": 10.0,
}


def _digest(obj: Any, h: Any = None) -> str:
    """Content hash of stage inputs (arrays, pydantic models, dataclasses, containers)."""
    top = h is None
    h = h or hashlib.blake2b(digest_size=16)
    if isinstance(obj, np.ndarray):
        h.update(f"nd{obj.shape}{obj.dtype}".encode())
        h.update(np.ascontiguousarray(obj).data)
    elif isinstance(obj, BaseModel):
        h.update(obj.model_dump_json().encode())
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            h.update(f.name.encode())
            _digest(getattr(obj, f.name), h)
    elif isinstance(obj, (list, tuple)):
        h.update(f"[{len(obj)}".encode())
        for o in obj:
            _digest(o, h)
    elif isinstance(obj, dict):
        for k in sorted(obj, key=repr):
            h.update(repr(k).encode())
            _digest(obj[k], h)
    else:
        h.update(repr(obj).encode())
    return h.hexdigest() if top else ""


@dataclass
class Call:
    method: str
    seconds: float
    cached: bool
    result: Any


class _Timed:
    """Stage proxy: times the real call, memoises expensive ones, forwards everything else."""

    def __init__(self, bench: BenchPipeline, stage: str, method: str, inner: Any) -> None:
        self._bench, self._stage, self._method, self._inner = bench, stage, method, inner

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name != CALL[self._stage]:
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            memo = self._bench.memo if self._stage in MEMO_STAGES and self._bench.use_memo else None
            key = (self._stage, self._method, id(self._inner), _digest((args, kwargs)))
            if memo is not None and key in memo:
                secs, res = memo[key]
                self._bench.calls[self._stage] = Call(self._method, secs, True, res)
                return copy.deepcopy(res)
            t = time.perf_counter()
            res = attr(*args, **kwargs)
            secs = time.perf_counter() - t
            if memo is not None:
                memo[key] = (secs, copy.deepcopy(res))
            self._bench.calls[self._stage] = Call(self._method, secs, False, res)
            return res

        return call


class BenchPipeline(Pipeline):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.memo: dict[tuple, tuple[float, Any]] = {}
        self.use_memo = True
        self.calls: dict[str, Call] = {}
        self.load_seconds: dict[str, float] = {}
        self.thresholds = dict(THRESHOLDS)
        self._reader: M.Reader | None | bool = False  # False: not built yet

    def reader(self) -> M.Reader | None:
        """Independent OCR for rendered Latin text (built once; None when unavailable)."""
        if self._reader is False:
            self._reader = None
            if self.settings.target_lang in CJK_TARGETS:
                log.warning("render read-back skipped: CJK target language")
            elif READER_OCR in REGISTRIES["ocr"].methods:
                try:
                    t = time.perf_counter()
                    ocr = REGISTRIES["ocr"].build(READER_OCR, self.settings, beams=1, min_line_conf=0.0)
                    self.load_seconds[f"judge:{READER_OCR}"] = time.perf_counter() - t
                    self._reader = _ocr_reader(ocr)
                except Exception as e:  # noqa: BLE001 - the bench runs on, reporting n/a
                    log.warning("render read-back unavailable (%s: %s)", type(e).__name__, e)
        return self._reader

    def stage(self, stage: str, cfg: PipelineConfig) -> Any:
        name = getattr(cfg, stage)
        opts = cfg.options.get(stage, {})
        key = (stage, name, repr(sorted(opts.items())))
        if key not in self._cache:
            t = time.perf_counter()
            inner = REGISTRIES[stage].build(name, self.settings, **opts)
            self.load_seconds[f"{stage}:{name}"] = time.perf_counter() - t
            self._cache[key] = _Timed(self, stage, name, inner)
        return self._cache[key]


def _ocr_reader(ocr: Any) -> M.Reader:
    def read(image: np.ndarray, quads: list[list[np.ndarray]]) -> list[str]:
        idx = [i for i, q in enumerate(quads) if q]
        regions = []
        for k, i in enumerate(idx):
            pts = np.concatenate(quads[i])
            (x1, y1), (x2, y2) = pts.min(0).astype(int), np.ceil(pts.max(0)).astype(int)
            lines = [[(int(x), int(y)) for x, y in q] for q in quads[i]]
            regions.append(
                TextRegion(id=k, bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2}, lines=lines, direction="h")
            )
        out = [""] * len(quads)
        for i, r in zip(idx, ocr.recognize(image, regions) if regions else []):
            out[i] = r.text
        return out

    return read


# -- one run ------------------------------------------------------------------------------


@dataclass
class GroundTruth:
    ja: list[str]
    id_ref: list[str]
    sfx: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> GroundTruth:
        d = json.loads(Path(path).read_text())
        bubbles = d["bubbles"]
        return cls(
            [b["ja"] for b in bubbles], [b.get("id_ref", "") for b in bubbles], d.get("sfx", [])
        )


def _bbox(b: Any) -> list[int] | None:
    return [b.x1, b.y1, b.x2, b.y2] if b is not None else None


def score_detection(
    texts: list[str], gt: GroundTruth, boxes: list | None = None
) -> tuple[dict, M.DetectionScore]:
    s = M.match_detections(texts, gt.ja, gt.sfx, boxes=boxes)
    return {
        "gt_bubbles": s.n_gt,
        "detections": s.n_det,
        "matched": len(s.matches),
        "recall": len(s.matches) / max(1, s.n_gt),
        "false_positives": len(s.false_positives),
        "fragments": len(s.fragments),
        "duplicates": len(s.duplicates),
        "sfx_hits": len(s.sfx),
        "missed_gt": s.missed,
    }, s


def score_ocr(s: M.DetectionScore) -> dict:
    return {
        "mean_cer": s.mean_cer,
        "mean_cer_text": s.mean_cer_text,
        "exact": sum(m.cer == 0 for m in s.matches),
        "matched": len(s.matches),
    }


def score_translation(
    bubbles: list[Bubble], gt: GroundTruth, s: M.DetectionScore, sim_min: float
) -> dict:
    """chrF against id_ref, plus meaning: embedding similarity of each GT bubble's translation
    to its own reference (missed bubbles score 0), and whether its own reference is the
    closest of all (catches outputs shifted between bubbles or padded with other lines)."""
    from .semantic import get_embedder

    by_gt = {m.gt: bubbles[m.det].target_text for m in s.matches}
    hyps_all = [by_gt.get(g, "") for g in range(len(gt.ja))]
    matched = sorted(by_gt)
    targets = [b.target_text for b in bubbles]
    sem: dict[str, Any] = {"semantic": None}
    emb = get_embedder() if gt.id_ref and all(gt.id_ref) else None
    if emb is not None:
        sim = np.array(emb.similarity(hyps_all, gt.id_ref), np.float64)
        # an empty or untranslated bubble is not adequate, however close the cross-lingual
        # embedding puts the Japanese to its reference
        sim[[not h.strip() or M.is_japanese(h) for h in hyps_all]] = 0.0
        own = np.diag(sim)
        ok = own >= sim_min
        sem = {
            "semantic": {
                "own_sim": [round(float(x), 3) for x in own],
                "mean_sim": float(own.mean()),
                "adequate": int(ok.sum()),
                "adequate_frac": float(ok.mean()),
                "aligned": int(sum(int(np.argmax(sim[g])) == g for g in range(len(own)))),
                "inadequate_gt": [int(g) for g in np.flatnonzero(~ok)],
            }
        }
    return sem | {
        "chrf": M.chrf(hyps_all, gt.id_ref),
        "chrf_pp": M.chrf(hyps_all, gt.id_ref, word_order=2),
        "chrf_matched": M.chrf([by_gt[g] for g in matched], [gt.id_ref[g] for g in matched])
        if matched
        else 0.0,
        "outputs": len(targets),
        "empty": sum(not t.strip() for t in targets),
        "japanese": sum(M.is_japanese(t) for t in targets),
    }


def score_inpaint(
    pipe: BenchPipeline,
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
    bubbles: list[Bubble],
    judge: PipelineConfig,
) -> dict:
    """Pixels outside the mask must be untouched; the judge detector + OCR must find no text
    left inside the erased areas."""
    out = {"mask_px": int((mask > 0).sum())}
    ring = int(pipe.thresholds["outside_ring_px"])
    out.update(
        {f"outside_{k}": v for k, v in M.outside_mask_change(original, inpainted, mask, ring).items()}
    )
    zone = cv2.dilate((mask > 0).astype(np.uint8), np.ones((15, 15), np.uint8))
    regions = pipe.stage("detector", judge).detect(inpainted).regions
    inside = [r for r in regions if M.box_overlap(zone, r.bbox) >= 0.5]
    texts = [o.text for o in pipe.stage("ocr", judge).recognize(inpainted, inside)] if inside else []
    residual = [t for t in texts if M.CJK.search(t)]
    out.update(
        {
            "residual_regions": len(residual),
            "residual_chars": sum(len(M.CJK.findall(t)) for t in residual),
            "residual_texts": residual,
            "erased_chars": sum(len(M.CJK.findall(b.source_text)) for b in bubbles),
        }
    )
    dr = M.dark_residue(original, inpainted, mask, [b.region.bbox for b in bubbles])
    out.update(
        {
            "dark_residue": dr["inpainted"],
            "ghost_residue": dr["ghost_max"],
            "dark_before": dr["original"],
            "dark_zone_px": dr["pixels"],
        }
    )
    return out


def score_render(
    inpainted: np.ndarray,
    final: np.ndarray,
    original: np.ndarray,
    bubbles: list[Bubble],
    plans,
    reader: M.Reader | None = None,
) -> tuple[dict, list[dict], np.ndarray]:
    drawn = [b for b in bubbles if b.target_text.strip()]
    boxes, overlay = M.render_check(inpainted, final, original, drawn, plans, reader)
    fonts = [b.font_px for b in boxes if b.font_px]
    known = [b for b in boxes if b.rendered_chars is not None]
    read = [b for b in boxes if b.read_cer is not None]
    summary = {
        "bubbles": len(boxes),
        "all_inside": all(b.inside_frac >= THRESHOLDS["inside_frac"] and b.bbox_inside for b in boxes),
        "min_inside_frac": min((b.inside_frac for b in boxes), default=1.0),
        "outside_px": sum(b.outside_px for b in boxes),
        "lost": sum(b.lost for b in boxes),
        "chars_checked": len(known),
        "chars_target": sum(b.target_chars for b in known),
        "chars_rendered": sum(b.rendered_chars or 0 for b in known),
        "read_back": bool(read),
        "read_cer_mean": float(np.mean([b.read_cer for b in read])) if read else None,
        "read_cer_max": max((b.read_cer for b in read), default=None),
        "chars_read": sum(b.read_chars or 0 for b in read),
        "chars_read_target": sum(b.target_chars for b in read),
        "overflow": sum(b.fits is False for b in boxes),
        "min_font_px": min(fonts) if fonts else None,
        "median_font_px": float(np.median(fonts)) if fonts else None,
        "font_source": boxes[0].font_source if boxes else None,
    }
    per = [
        {
            "closed_balloon": b.closed,
            "ink_px": b.ink_px,
            "inside_frac": round(b.inside_frac, 4),
            "ink_bbox": _bbox(b.ink_bbox),
            "bbox_inside": b.bbox_inside,
            "target_chars": b.target_chars,
            "rendered_chars": b.rendered_chars,
            "font_px": b.font_px,
            "fits": b.fits,
            "read_text": b.read_text,
            "read_cer": None if b.read_cer is None else round(b.read_cer, 3),
        }
        for b in boxes
    ]
    return summary, per, overlay


def checks(m: dict, T: dict | None = None) -> dict[str, bool | None]:
    """Pass/fail per criterion (None: not measurable for this run).

    Render text completeness needs proof from pixels (independent read-back) or at least the
    renderer's own layout; with neither, render is None rather than a pass.
    """
    T = T or THRESHOLDS
    d, o, t, i, r = (m.get(k) or {} for k in ("detection", "ocr", "translation", "inpaint", "render"))
    out: dict[str, bool | None] = {}
    if d:
        out["detect"] = (
            d["recall"] >= T["detect_recall"] - 1e-9
            and d["false_positives"] <= T["false_positives"]
            and d.get("duplicates", 0) == 0
        )
    if o:
        out["ocr"] = o["mean_cer"] is not None and o["mean_cer"] <= T["ocr_cer"]
    if t:
        sem = t.get("semantic")
        out["translate"] = (
            t["chrf"] >= T["chrf"]
            and t["empty"] == 0
            and t["japanese"] == 0
            and (sem is None or sem["adequate_frac"] >= T["adequate_frac"] - 1e-9)
        )
    if i:
        erased = i["mask_px"] > 0
        out["inpaint"] = (
            erased
            and i["residual_regions"] <= T["residual_regions"]
            and i["dark_residue"] <= T["dark_residue"]
            and i["ghost_residue"] <= T["ghost_residue"]
            and i["outside_mad"] <= T["outside_mad"]
            and i["outside_far_changed"] <= T["outside_far_changed"]
        ) if erased or d.get("gt_bubbles") else None
    if r and r["bubbles"]:
        read_ok = r.get("read_back") and (
            r["read_cer_mean"] <= T["read_cer_mean"] and r["read_cer_max"] <= T["read_cer_max"]
        )
        verified = bool(r.get("read_back")) or r["chars_checked"] > 0
        ok = (
            r["all_inside"]
            and r["lost"] == 0
            and r["chars_rendered"] >= r["chars_target"]
            and (r["min_font_px"] or 0) >= T["min_font_px"]
            and (read_ok if r.get("read_back") else True)
        )
        out["render"] = (ok if verified else None) if ok else False
    elif r:
        out["render"] = None
    return out


def evaluate(
    pipe: BenchPipeline,
    page: np.ndarray,
    gt: GroundTruth | None,
    cfg: PipelineConfig,
    out_dir: Path | None = None,
    judge: PipelineConfig | None = None,
) -> dict:
    """Run the pipeline once with `cfg` and score every stage. Writes images when out_dir."""
    pipe.calls.clear()
    t0 = time.perf_counter()
    res = pipe.run(page, config=cfg, debug=True)
    wall = time.perf_counter() - t0
    resolved = pipe.resolve(cfg)
    calls = dict(pipe.calls)
    latency = {TIMING_KEY[s]: round(c.seconds, 3) for s, c in calls.items()}
    cached = sorted(TIMING_KEY[s] for s, c in calls.items() if c.cached)
    bubbles = res.result.bubbles
    det_call = calls.get("detector")
    m: dict[str, Any] = {
        "config": {s: getattr(resolved, s) for s in STAGES},
        "options": resolved.options,
        "latency_s": latency,
        "latency_total_s": round(sum(latency.values()), 3),
        "latency_own_s": round(sum(c.seconds for c in calls.values() if not c.cached), 3),
        "wall_s": round(wall, 3),
        "cached_stages": cached,
        "regions_detected": len(det_call.result.regions) if det_call else None,
    }
    # score what OCR read: translators may return a cleaned-up `source` (chain drops fillers)
    ocr_texts = [o.text for o in calls["ocr"].result] if "ocr" in calls else []
    read = [
        ocr_texts[b.region.id]
        if b.region.id < len(ocr_texts) and ocr_texts[b.region.id].strip()
        else b.source_text
        for b in bubbles
    ]
    per_bubble: list[dict] = [
        {"id": b.region.id, "bbox": _bbox(b.region.bbox), "source": r, "target": b.target_text}
        | ({"translator_source": b.source_text} if b.source_text != r else {})
        for b, r in zip(bubbles, read)
    ]
    if gt is not None:
        m["detection"], s = score_detection(read, gt, [b.region.bbox for b in bubbles])
        m["ocr"] = score_ocr(s)
        m["translation"] = score_translation(bubbles, gt, s, pipe.thresholds["semantic_sim"])
        sem = m["translation"]["semantic"]
        for mt in s.matches:
            pb = per_bubble[mt.det]
            pb.update(
                gt=mt.gt,
                gt_ja=gt.ja[mt.gt],
                id_ref=gt.id_ref[mt.gt],
                cer=round(mt.cer, 3),
                chrf=round(M.chrf([pb["target"]], [gt.id_ref[mt.gt]]), 1),
            )
            if sem is not None:
                pb["sim"] = sem["own_sim"][mt.gt]
        for d in s.sfx:
            per_bubble[d]["class"] = "sfx"
        for d in s.fragments:
            per_bubble[d]["class"] = "fragment"
        for d in s.duplicates:
            per_bubble[d]["class"] = "duplicate"
        for d in s.false_positives:
            per_bubble[d]["class"] = "false_positive"
        m["missed_gt_text"] = [gt.ja[g] for g in s.missed]
    inpainted, mask, final = res.debug["inpainted"], res.debug["mask"], res.debug["final"]
    m["inpaint"] = score_inpaint(pipe, page, inpainted, mask, bubbles, judge or pipe.resolve(None))
    plans = getattr(pipe.stage("renderer", resolved), "plans", None)
    m["render"], per_render, overlay = score_render(
        inpainted, final, page, bubbles, plans, pipe.reader()
    )
    drawn = [i for i, b in enumerate(bubbles) if b.target_text.strip()]
    for i, pr in zip(drawn, per_render):
        per_bubble[i]["render"] = pr
    m["bubbles"] = per_bubble
    m["checks"] = checks(m, pipe.thresholds)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        regions = det_call.result.regions if det_call else [b.region for b in bubbles]
        for name, img in {
            "final": final,
            "inpainted": inpainted,
            "mask": mask,
            "detect": draw_regions(page, [b.region for b in bubbles] or regions),
            "render_check": overlay,
        }.items():
            cv2.imwrite(str(out_dir / f"{name}.png"), img)
        (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str))
    return m


# -- matrix -------------------------------------------------------------------------------


def warm_latency(pipe: BenchPipeline, page: np.ndarray, cfg: PipelineConfig) -> dict[str, float]:
    """Per-stage seconds of a second, uncached run: first calls include lazy model loading."""
    pipe.use_memo = False
    try:
        pipe.calls.clear()
        pipe.run(page, config=cfg)
        return {TIMING_KEY[s]: round(c.seconds, 3) for s, c in pipe.calls.items()}
    finally:
        pipe.use_memo = True


def estimate_total(run: dict, warm: dict[str, float]) -> float:
    """Pipeline seconds for this config alone: stages this run reused from the cache (the
    baseline's, often with a cold model load) count at the baseline's warm time instead."""
    lat = run["latency_s"]
    cached = set(run.get("cached_stages", []))
    return round(sum(warm.get(k, v) if k in cached else v for k, v in lat.items()), 3)


def available_methods(
    pipe: BenchPipeline, base: PipelineConfig, allow_downloads: bool, network: bool
) -> dict[str, list[tuple[str, str | None]]]:
    """Per stage: (method, skip reason or None), discovered from the registries at runtime."""
    out: dict[str, list[tuple[str, str | None]]] = {}
    for stage in STAGES:
        rows = []
        for name in REGISTRIES[stage].names():
            info = REGISTRIES[stage].get(name)
            ok, reason = info.status(pipe.settings)
            skip = None if ok else reason
            if skip is None and not info.offline and not network:
                skip = "network API (pass --network)"
            if skip is None and HEAVY_DOWNLOAD.search(info.description) and not allow_downloads:
                skip = "large download on first use (pass --allow-downloads)"
            if skip is None and f"{stage}:{name}" in NEEDS_SERVER and not network:
                skip = NEEDS_SERVER[f"{stage}:{name}"]
            if (
                skip is None
                and stage == "ocr"
                and name == "none"
                and not getattr(pipe.stage("translator", base), "vision", False)
            ):
                skip = f"needs a vision translator (default '{base.translator}' is text-only)"
            rows.append((name, skip))
        out[stage] = rows
    return out


def run_matrix(
    pipe: BenchPipeline,
    page: np.ndarray,
    gt: GroundTruth | None,
    out_dir: Path,
    base: PipelineConfig,
    stages: list[str] | None = None,
    allow_downloads: bool = False,
    network: bool = False,
    warm: bool = True,
) -> dict:
    base = pipe.resolve(base)
    runs: list[dict] = []

    def one(label: str, stage: str | None, method: str | None, cfg: PipelineConfig) -> None:
        log.info("bench run %s", label)
        try:
            m = evaluate(pipe, page, gt, cfg, out_dir / "runs" / label, judge=base)
            m.update(label=label, stage=stage, method=method, status="ok")
        except FileNotFoundError as e:  # untrained / undownloaded weights: not runnable here
            log.warning("run %s skipped: %s", label, e)
            m = {"label": label, "stage": stage, "method": method, "status": "skipped", "reason": f"missing file: {e}"}
        except Exception as e:  # a broken method is a result, not a crash
            log.exception("run %s failed", label)
            m = {"label": label, "stage": stage, "method": method, "status": "error", "error": f"{type(e).__name__}: {e}"}
        runs.append(m)

    one("baseline", None, None, base)
    if warm and runs[0]["status"] == "ok":
        runs[0]["latency_warm_s"] = warm_latency(pipe, page, base)
    warm_lat = runs[0].get("latency_warm_s") or {}
    methods = available_methods(pipe, base, allow_downloads, network)
    for stage in stages or list(STAGES):
        for name, skip in methods[stage]:
            if name == getattr(base, stage):
                continue
            label = f"{stage}-{name}"
            if skip:
                runs.append({"label": label, "stage": stage, "method": name, "status": "skipped", "reason": skip})
                continue
            cfg = base.model_copy(deep=True)
            setattr(cfg, stage, name)
            cfg.options.pop(stage, None)
            one(label, stage, name, cfg)
    for r in runs:
        if r["status"] == "ok":
            r["latency_est_s"] = estimate_total(r, warm_lat)
    report = {
        "page_shape": list(page.shape),
        "baseline": {s: getattr(base, s) for s in STAGES},
        "thresholds": pipe.thresholds,
        "load_seconds": {k: round(v, 2) for k, v in pipe.load_seconds.items()},
        "runs": runs,
    }
    return report


def contact_sheet(runs: list[dict], out_dir: Path, font_path: Path | None, tile_w: int = 360) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    ok = [r for r in runs if r.get("status") == "ok" and (out_dir / "runs" / r["label"] / "final.png").exists()]
    cols = min(6, max(1, len(ok)))
    tiles = []
    for r in ok:
        img = cv2.imread(str(out_dir / "runs" / r["label"] / "final.png"))
        h = round(img.shape[0] * tile_w / img.shape[1])
        tiles.append((r, cv2.cvtColor(cv2.resize(img, (tile_w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))
    th = max(t.shape[0] for _, t in tiles) if tiles else 1
    cap = 64
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tile_w + 8) + 8, rows * (th + cap + 8) + 8), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(str(font_path), 17) if font_path else ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()
    for k, (r, t) in enumerate(tiles):
        x, y = 8 + (k % cols) * (tile_w + 8), 8 + (k // cols) * (th + cap + 8)
        sheet.paste(Image.fromarray(t), (x, y + cap))
        c = r.get("checks", {})
        mark = " ".join(f"{s[:3]}{'+' if v else '-' if v is False else '?'}" for s, v in c.items())
        draw.text((x, y + 2), r["label"], fill=(255, 255, 255), font=font)
        draw.text((x, y + 24), mark, fill=(150, 230, 150) if all(v is not False for v in c.values()) else (255, 160, 120), font=font)
        tot = r.get("latency_est_s", r.get("latency_total_s"))
        draw.text((x, y + 44), f"{tot:.1f}s" if tot is not None else "", fill=(200, 200, 200), font=font)
    path = out_dir / "contact_sheet.jpg"
    sheet.save(path, quality=88)
    return path


# -- markdown ----------------------------------------------------------------------------


def _f(v: Any, nd: int = 2) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _ck(v: bool | None) -> str:
    return "PASS" if v else ("FAIL" if v is False else "n/a")


def run_row(r: dict) -> str:
    if r["status"] != "ok":
        why = r.get("reason") or r.get("error", "")
        return f"| {r['label']} | {r['status']} | {why} |" + " |" * 11
    d, o, t, i, rd = (r.get(k) or {} for k in ("detection", "ocr", "translation", "inpaint", "render"))
    c = r["checks"]
    lat = r["latency_s"]
    stage_lat = lat.get(TIMING_KEY[r["stage"]]) if r.get("stage") else None
    sem = t.get("semantic") or {}
    adq = f"{sem['adequate']}/{len(sem['own_sim'])}" if sem else "–"
    return (
        f"| {r['label']} | ok | "
        f"{d.get('matched', '–')}/{d.get('gt_bubbles', '–')} fp {d.get('false_positives', '–')} "
        f"dup {d.get('duplicates', '–')} {_ck(c.get('detect'))} | "
        f"{_f(o.get('mean_cer'), 3)} {_ck(c.get('ocr'))} | "
        f"{_f(t.get('chrf'), 1)} e{t.get('empty', '–')} j{t.get('japanese', '–')} | "
        f"{adq} {_f(sem.get('mean_sim'), 2)} {_ck(c.get('translate'))} | "
        f"{i.get('residual_regions', '–')} / {_f(i.get('dark_residue'), 4)} / {_f(i.get('ghost_residue'), 4)} | "
        f"{_f(i.get('outside_mad'), 3)} / {_f(i.get('outside_changed_frac'), 4)} / {i.get('outside_far_changed', '–')} {_ck(c.get('inpaint'))} | "
        f"{_f(rd.get('min_inside_frac'), 3)} lost {rd.get('lost', '–')} | "
        f"{_f(rd.get('read_cer_mean'), 3)} / {_f(rd.get('read_cer_max'), 2)} | "
        f"{_f(rd.get('min_font_px'), 0)} {_ck(c.get('render'))} | "
        f"{_f(stage_lat, 2)} | {_f(r.get('latency_est_s', r['latency_total_s']), 1)} |"
    )


def markdown(report: dict) -> str:
    runs = report["runs"]
    base = next(r for r in runs if r["label"] == "baseline")
    L = [
        "# terjemahin benchmark",
        "",
        (
            f"Page {report['page_shape'][1]}x{report['page_shape'][0]}; baseline "
            + ", ".join(f"{k}=`{v}`" for k, v in report["baseline"].items())
            + ". Each matrix row changes one stage, the rest stay at the baseline."
        ),
        "",
        (
            "Columns: detection = GT bubbles matched by OCR text (Hungarian on CER <= 0.5), false "
            "positives and duplicate detections; OCR = mean CER over matched bubbles; chrF vs `id_ref` "
            "(missing bubbles count as empty), e/j = empty / Japanese outputs; meaning = GT bubbles "
            f"whose translation has embedding cosine >= {THRESHOLDS['semantic_sim']} to its reference "
            "(paraphrase-multilingual-MiniLM) and the mean cosine; "
            "residual = text regions the judge detector+OCR "
            "still reads inside erased areas / dark and off-tone (ghost) pixel shares left in "
            "erased balloon interiors; "
            "outside = mean abs diff / changed share of pixels OUTSIDE the mask / changed pixels more "
            "than 8 px from it (must be 0: only edge feathering may touch the ring); inside = "
            "worst share of a bubble's rendered ink inside its balloon (or padded box) and bubbles "
            "whose text was lost; read-back = mean / worst CER (letters only) of the rendered text "
            f"read back from pixels by an independent OCR (`{READER_OCR}`) against the translation; "
            "font = smallest font px; total = pipeline seconds for that config, stages reused from "
            "the cache counted at the baseline's warm time."
        ),
        "",
        "| run | status | detection | OCR CER | chrF | meaning | residual | outside mask | inside | read-back CER | min font | stage s | total s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    L += [run_row(r) for r in runs]
    if base.get("status") == "ok":
        T = report["thresholds"]
        c = base["checks"]
        L += ["", "## Baseline checks", ""]
        d, o, t, i, rd = (base.get(k) or {} for k in ("detection", "ocr", "translation", "inpaint", "render"))
        if d:
            L.append(
                f"- **detect** {_ck(c.get('detect'))}: {d['matched']}/{d['gt_bubbles']} GT bubbles matched "
                f"(need >= {T['detect_recall']:.0%}), {d['false_positives']} false positives, "
                f"{d['sfx_hits']} SFX regions, {d['fragments']} fragments, {d['duplicates']} duplicates, "
                f"{base['regions_detected']} raw regions."
            )
        if o:
            L.append(
                f"- **ocr** {_ck(c.get('ocr'))}: mean CER {_f(o['mean_cer'], 3)} (need <= {T['ocr_cer']}), "
                f"{_f(o['mean_cer_text'], 3)} ignoring punctuation, {o['exact']}/{o['matched']} exact."
            )
        if t:
            L.append(
                f"- **translate** {_ck(c.get('translate'))}: chrF {_f(t['chrf'], 1)} (need >= {T['chrf']}), "
                f"chrF++ {_f(t['chrf_pp'], 1)}, {t['empty']} empty, {t['japanese']} still Japanese "
                f"of {t['outputs']}; "
                + (
                    f"meaning: {t['semantic']['adequate']}/{len(t['semantic']['own_sim'])} GT bubbles "
                    f"adequate (cosine >= {T['semantic_sim']}, need >= {T['adequate_frac']:.0%}), mean "
                    f"cosine {_f(t['semantic']['mean_sim'], 3)}, {t['semantic']['aligned']} closest to "
                    f"their own reference; inadequate GT #"
                    + ", #".join(map(str, t["semantic"]["inadequate_gt"]))
                    + "."
                    if t.get("semantic")
                    else "meaning NOT checked (embedding model unavailable)."
                )
            )
        L.append(
            f"- **inpaint** {_ck(c.get('inpaint'))}: {i['mask_px']} px erased, "
            f"{i['residual_regions']} residual text regions "
            f"({i['residual_chars']} chars of {i['erased_chars']} erased), dark residue "
            f"{_f(i['dark_residue'], 4)} (was {_f(i['dark_before'], 3)} before erasing), ghost "
            f"{_f(i['ghost_residue'], 4)} ("
            f"{i['dark_zone_px']} px), outside-mask MAD {_f(i['outside_mad'], 4)}, max "
            f"{i['outside_max']}, changed {_f(i['outside_changed_frac'], 6)} (all within "
            f"{T['outside_ring_px']} px of the mask: {i['outside_far_changed'] == 0})."
        )
        L.append(
            f"- **render** {_ck(c.get('render'))}: {rd['bubbles']} bubbles, worst inside share "
            f"{_f(rd['min_inside_frac'], 4)}, {rd['outside_px']} ink px outside, {rd['lost']} lost, "
            f"layout chars {rd['chars_rendered']}/{rd['chars_target']} ({rd['chars_checked']} layouts), "
            + (
                f"read back from pixels: {rd['chars_read']}/{rd['chars_read_target']} letters, CER mean "
                f"{_f(rd['read_cer_mean'], 3)} (need <= {T['read_cer_mean']}), worst "
                f"{_f(rd['read_cer_max'], 3)} (need <= {T['read_cer_max']}), "
                if rd.get("read_back")
                else "NO pixel read-back (judge OCR unavailable), "
            )
            + f"{rd['overflow']} overflowing, font {_f(rd['min_font_px'], 0)}..{_f(rd['median_font_px'], 0)} px "
            f"(min..median, {rd['font_source']})."
        )
        L.append(
            "- **latency** (s, first call, includes lazy model loading): "
            + ", ".join(f"{k} {v:.2f}" for k, v in base["latency_s"].items())
            + f"; total {base['latency_total_s']:.1f}."
        )
        if base.get("latency_warm_s"):
            w = base["latency_warm_s"]
            L.append(
                "- **warm latency** (s, second uncached run): "
                + ", ".join(f"{k} {v:.2f}" for k, v in w.items())
                + f"; total {sum(w.values()):.1f}."
            )
        L += [
            "",
            "## Baseline per bubble",
            "",
            "| # | GT | OCR | CER | translation | reference | chrF | cosine | read back from pixels | read CER | font | inside |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for b in base["bubbles"]:
            rr = b.get("render") or {}
            gtj = b.get("gt_ja") or f"*{b.get('class', 'unmatched')}*"
            L.append(
                f"| {b['id']} | {gtj} | {b['source']} | {_f(b.get('cer'), 2)} | {b['target']} | "
                f"{b.get('id_ref', '')} | {_f(b.get('chrf'), 1)} | {_f(b.get('sim'), 2)} | "
                f"{rr.get('read_text') or ''} | {_f(rr.get('read_cer'), 2)} | "
                f"{_f(rr.get('font_px'), 0)} | {_f(rr.get('inside_frac'), 3)} |"
            )
        if base.get("missed_gt_text"):
            L += ["", "Missed GT bubbles: " + ", ".join(base["missed_gt_text"])]
    if report.get("load_seconds"):
        L += ["", "## Model load (s)", "", ", ".join(f"{k} {v}" for k, v in report["load_seconds"].items())]
    L += ["", "![contact sheet](contact_sheet.jpg)", ""]
    return "\n".join(L)


def bench(
    page_path: Path,
    gt_path: Path | None,
    out_dir: Path,
    settings: Settings,
    matrix: bool = False,
    stages: list[str] | None = None,
    base: PipelineConfig | None = None,
    allow_downloads: bool = False,
    network: bool = False,
    warm: bool = True,
    strict_mask: bool = False,
) -> dict:
    page = cv2.imread(str(page_path))
    if page is None:
        raise FileNotFoundError(page_path)
    gt = GroundTruth.load(gt_path) if gt_path else None
    pipe = BenchPipeline(settings)
    if strict_mask:  # no pixel outside the mask may change, not even edge feathering
        pipe.thresholds["outside_ring_px"] = 0
    base = base or PipelineConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    if matrix:
        report = run_matrix(pipe, page, gt, out_dir, base, stages, allow_downloads, network, warm)
    else:
        m = evaluate(pipe, page, gt, base, out_dir / "runs" / "baseline")
        m.update(label="baseline", stage=None, method=None, status="ok")
        if warm:
            m["latency_warm_s"] = warm_latency(pipe, page, base)
        m["latency_est_s"] = m["latency_own_s"]
        report = {
            "page_shape": list(page.shape),
            "baseline": m["config"],
            "thresholds": pipe.thresholds,
            "load_seconds": {k: round(v, 2) for k, v in pipe.load_seconds.items()},
            "runs": [m],
        }
    contact_sheet(report["runs"], out_dir, settings.font_path)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    (out_dir / "report.md").write_text(markdown(report))
    return report
