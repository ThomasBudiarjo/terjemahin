"""comic-text-detector with all three heads: blocks (YOLOv5), text lines (DBNet), glyph mask.

Speed note: about 40% of this export's weights are float32 denormals. x86 handles those in
microcode, which made the full graph take ~9.4s per page on CPU (the old "ConvTranspose is slow"
diagnosis). Enabling onnxruntime's denormals-as-zero brings it to ~0.6s with bit-identical
outputs on page01, so all heads stay on by default.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np

from ..models import BBox, TextRegion
from ..opts import as_bool, as_float, as_int, optional, warn_unknown
from ..registry import DETECTORS
from ..weights import ensure
from .base import Detection
from .ctd_utils import Block, db_lines, decode_blocks, group_lines, nms_xyxy

log = logging.getLogger(__name__)

HEADS = ("blk", "seg", "det")


def make_session(path: Path, heads: tuple[str, ...] = HEADS, threads: int = 0):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.set_denormal_as_zero", "1")
    if threads > 0:
        so.intra_op_num_threads = threads
    model: str | bytes = str(path)
    if set(heads) != set(HEADS):
        # onnxruntime computes every output regardless of what run() asks for, so a cheaper
        # graph needs the unused heads cut out (blk alone: ~0.1s instead of ~0.6s).
        try:
            import onnx
            from onnx.utils import Extractor

            full = onnx.load(str(path))
            model = Extractor(full).extract_model(["images"], list(heads)).SerializeToString()
        except ImportError:
            log.warning("onnx not installed: running all ctd heads, ignoring heads=%s", heads)
    return ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])


MAX_UPSCALE = 2.0


SPREAD_ASPECT = 1.2  # width / height from which a page is read as a two-page spread
SPREAD_OVERLAP = 0.08  # share of the width each half reaches past the middle


def plan_tiles(
    w: int,
    h: int,
    size: int,
    tile_aspect: float,
    tile_shape: float = 1.5,
    overlap: float = 0.2,
    spread: float = SPREAD_ASPECT,
) -> tuple[float, list[tuple[int, int, int, int]]]:
    """(ratio, tiles as x0, y0, x1, y1). Pages up to `tile_aspect` run in one letterboxed pass;
    taller (webtoon) or wider strips are cut into overlapping tiles of `tile_shape` (long/short,
    about a manga page) along the long axis so text keeps the resolution of a normal page.

    A landscape page from `spread` (w/h) up to `tile_aspect` is a two-page scan: one pass over
    the whole of it shrinks its text by the spread's width (a 1654x1170 Manga109 spread to
    62%, where 14 px glyphs fall below what the detector finds: on ARMS_015 three ドクン
    balloons were missed or cut). It runs as two overlapping halves, each at about the scale
    of a single page. `spread=0` turns this off."""
    long, short = max(w, h), min(w, h)
    if long <= tile_aspect * short:
        if spread and w >= spread * h and w > size:
            half = min(w, math.ceil(w / 2 + SPREAD_OVERLAP * w))
            return size / max(half, h), [(0, 0, half, h), (w - half, 0, w, h)]
        return size / long, [(0, 0, w, h)]
    # never upscale a tile more than MAX_UPSCALE: an 8 px wide strip made 1024/12 = 85x tiles,
    # ~32 forward passes for a 300 px image (and a 100x12000 strip about 100)
    tile = max(round(tile_shape * short), min(long, math.ceil(size / MAX_UPSCALE)))
    if tile >= long:
        return size / long, [(0, 0, w, h)]
    step = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, long - tile, step)) + [long - tile]
    if h >= w:
        tiles = [(0, s, w, s + tile) for s in starts]
    else:
        tiles = [(s, 0, s + tile, h) for s in starts]
    return size / tile, tiles


class CtdFullDetector:
    """Blocks from the YOLO head, lines from the DB head assigned to blocks, glyph mask from seg."""

    def __init__(
        self,
        weights: Path,
        conf: float = 0.4,
        iou: float = 0.35,
        line_thresh: float = 0.3,
        line_box_thresh: float = 0.6,
        unclip: float = 1.5,
        mask_thresh: float | None = None,
        box_padding: int = 0,
        split_gap: float = 2.0,
        keep_scattered: bool = True,
        heads: tuple[str, ...] = HEADS,
        threads: int = 0,
        tile_aspect: float = 2.5,
        mask_scope: str = "blocks",
        session=None,
        spread: float = SPREAD_ASPECT,
    ) -> None:
        if mask_scope not in ("blocks", "page"):
            raise ValueError("mask_scope must be 'blocks' or 'page'")
        if "blk" not in heads and "det" not in heads:
            raise ValueError("ctd needs the 'blk' or 'det' head to find text")
        self.heads = tuple(h for h in HEADS if h in heads)
        conf = as_float(conf, "detector conf", 0.0, 1.0)
        iou = as_float(iou, "detector iou", 0.0, 1.0)
        line_thresh = as_float(line_thresh, "detector line_thresh", 0.0, 1.0)
        line_box_thresh = as_float(line_box_thresh, "detector line_box_thresh", 0.0, 1.0)
        unclip = as_float(unclip, "detector unclip", 0.1, 10.0)
        mask_thresh = optional(as_float, mask_thresh, "detector mask_thresh", 0.0, 1.0)
        box_padding = as_int(box_padding, "detector box_padding", 0, 500)
        split_gap = as_float(split_gap, "detector split_gap", 0.1)
        keep_scattered = as_bool(keep_scattered, "detector keep_scattered")
        tile_aspect = as_float(tile_aspect, "detector tile_aspect", 1.0)
        self.spread = as_float(spread, "detector spread", 0.0)
        threads = as_int(threads, "detector threads", 0, 1024)
        self.session = session or make_session(weights, self.heads, threads)
        self.size = 1024
        self.conf, self.iou = conf, iou
        self.line_thresh, self.line_box_thresh, self.unclip = line_thresh, line_box_thresh, unclip
        self.mask_thresh = mask_thresh
        self.box_padding = box_padding
        self.split_gap = split_gap
        self.keep_scattered = keep_scattered
        self.tile_aspect = tile_aspect
        self.mask_scope = mask_scope

    def _forward(self, crop_bgr: np.ndarray, ratio: float) -> dict[str, np.ndarray]:
        """Run one letterboxed tile; maps come back cropped to the resized tile."""
        h, w = crop_bgr.shape[:2]
        nw, nh = min(self.size, max(1, round(w * ratio))), min(self.size, max(1, round(h * ratio)))
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(crop_bgr, (nw, nh), interpolation=interp)
        x = canvas[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        outs = dict(zip(self.heads, self.session.run(list(self.heads), {"images": x})))
        res: dict[str, np.ndarray] = {}
        if "blk" in outs:
            res["blk"] = outs["blk"][0]
        if "seg" in outs:
            res["seg"] = outs["seg"][0, 0, :nh, :nw]
        if "det" in outs:
            res["det"] = outs["det"][0, 0, :nh, :nw]
        return res

    def detect(self, image_bgr: np.ndarray) -> Detection:
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
        H, W = image_bgr.shape[:2]
        ratio, tiles = plan_tiles(W, H, self.size, self.tile_aspect, spread=self.spread)
        found = self._detect(image_bgr, ratio, tiles)
        if len(tiles) == 2 and W >= self.spread * H and W <= self.tile_aspect * H:
            # a spread read as two halves: one pass over the whole page as well, for the text
            # the closer look misses (a small balloon pressed against another: ARMS_015)
            whole = self._detect(image_bgr, self.size / max(W, H), [(0, 0, W, H)])
            found = union_detections(found, whole)
        return found

    def _detect(
        self, image_bgr: np.ndarray, ratio: float, tiles: list[tuple[int, int, int, int]]
    ) -> Detection:
        H, W = image_bgr.shape[:2]
        ww, wh = max(1, round(W * ratio)), max(1, round(H * ratio))
        seg = np.zeros((wh, ww), np.float32) if "seg" in self.heads else None
        det = np.zeros((wh, ww), np.float32) if "det" in self.heads else None
        cand_xyxy, cand_score, cand_cls, cand_edge = [], [], [], []
        for x0, y0, x1, y1 in tiles:
            out = self._forward(image_bgr[y0:y1, x0:x1], ratio)
            ox, oy = round(x0 * ratio), round(y0 * ratio)
            for name, canvas in (("seg", seg), ("det", det)):
                if canvas is not None:
                    m = out[name]
                    th, tw = min(m.shape[0], wh - oy), min(m.shape[1], ww - ox)
                    view = canvas[oy : oy + th, ox : ox + tw]
                    np.maximum(view, m[:th, :tw], out=view)
            if "blk" in out:
                xyxy, sc, cls = decode_blocks(out["blk"], self.conf, self.iou)
                th, tw = (y1 - y0) * ratio, (x1 - x0) * ratio
                inner = np.zeros(len(xyxy), bool)  # touches a cut that is not a page edge
                if y0 > 0:
                    inner |= xyxy[:, 1] < 4
                if y1 < H:
                    inner |= xyxy[:, 3] > th - 4
                if x0 > 0:
                    inner |= xyxy[:, 0] < 4
                if x1 < W:
                    inner |= xyxy[:, 2] > tw - 4
                cand_xyxy.append(xyxy + [ox, oy, ox, oy])
                cand_score.append(sc)
                cand_cls.append(cls)
                cand_edge.append(inner)

        xyxy = np.concatenate(cand_xyxy) if cand_xyxy else np.zeros((0, 4), np.float32)
        scores = np.concatenate(cand_score) if cand_score else np.zeros(0, np.float32)
        classes = np.concatenate(cand_cls) if cand_cls else np.zeros(0, np.int64)
        if len(tiles) > 1 and len(xyxy):
            keep = merge_tile_blocks(xyxy, scores, np.concatenate(cand_edge), self.iou)
            xyxy, scores, classes = xyxy[keep], scores[keep], classes[keep]

        lines, line_scores = ([], [])
        if det is not None:
            lines, line_scores = db_lines(det, self.line_thresh, self.line_box_thresh, self.unclip)
        blocks = group_lines(
            xyxy, scores, lines, line_scores, seg,
            split_gap=self.split_gap,
            keep_scattered=self.keep_scattered or "blk" not in self.heads,
            classes=classes,
        )

        mask = None
        if seg is not None:
            mask = cv2.resize((seg * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
            if self.mask_thresh is not None:
                mask = np.where(mask >= self.mask_thresh * 255, 255, 0).astype(np.uint8)
        regions = [
            r for b in blocks if (r := self._region(b, ratio, W, H, image_bgr, mask)) is not None
        ]
        if mask is not None and self.mask_scope == "blocks":
            mask = scope_mask(mask, regions)
        return Detection([r.model_copy(update={"id": i}) for i, r in enumerate(regions)], mask)

    def _region(
        self, b: Block, ratio: float, W: int, H: int, image: np.ndarray, mask: np.ndarray | None
    ) -> TextRegion | None:
        x0, y0, x1, y1 = (b.xyxy / ratio).tolist()
        p = self.box_padding
        bbox = BBox(
            x1=int(np.floor(x0)) - p, y1=int(np.floor(y0)) - p,
            x2=int(np.ceil(x1)) + p, y2=int(np.ceil(y1)) + p,
        ).clamp(W, H)
        if bbox.w < 4 or bbox.h < 4:
            return None
        lines = [
            [(round(min(max(float(x), 0), W)), round(min(max(float(y), 0), H))) for x, y in q / ratio]
            for q in b.lines
        ]
        fg, bg = block_colors(image, mask, bbox)
        return TextRegion(
            id=0,
            bbox=bbox,
            score=round(float(b.score), 4),
            lines=lines,
            direction="v" if b.vertical else "h",
            angle=b.angle,
            font_size=max(1, round(b.font_size / ratio)),
            fg_color=fg,
            bg_color=bg,
        )


def union_detections(main: Detection, extra: Detection, cover: float = 0.3) -> Detection:
    """`main` plus what only `extra` found: each of extra's text lines that no main region
    covers (overlap over the smaller box < `cover`), grouped by the extra region it came
    from; for a region without lines, the region itself when no main region covers it.
    The glyph masks are combined by maximum."""
    from .postprocess import _qbox

    def covered(b: BBox) -> bool:
        return any(_overlap_small(b, r.bbox) >= cover for r in main.regions)

    added: list[TextRegion] = []
    for r in extra.regions:
        if not r.lines:
            if not covered(r.bbox):
                added.append(r)
            continue
        free = []
        for q in r.lines:
            x1, y1, x2, y2 = _qbox(q)
            if not covered(BBox(x1=x1, y1=y1, x2=x2 + 1, y2=y2 + 1)):
                free.append(q)
        if not free:
            continue
        boxes = [_qbox(q) for q in free]
        bb = BBox(
            x1=min(b[0] for b in boxes), y1=min(b[1] for b in boxes),
            x2=max(b[2] for b in boxes) + 1, y2=max(b[3] for b in boxes) + 1,
        )
        added.append(r.model_copy(update={"bbox": bb, "lines": free}))
    mask = main.mask
    if mask is not None and extra.mask is not None and extra.mask.shape == mask.shape:
        mask = np.maximum(mask, extra.mask)
    regions = [*main.regions, *added]
    if added:
        log.info("spread: %d region(s) found only by the whole-page pass", len(added))
    return Detection([r.model_copy(update={"id": i}) for i, r in enumerate(regions)], mask)


def _overlap_small(a: BBox, b: BBox) -> float:
    iw = min(a.x2, b.x2) - max(a.x1, b.x1)
    ih = min(a.y2, b.y2) - max(a.y1, b.y1)
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(1, min(a.w * a.h, b.w * b.h))


def merge_tile_blocks(
    xyxy: np.ndarray, scores: np.ndarray, at_cut: np.ndarray, iou: float
) -> np.ndarray:
    """Indices to keep: drop blocks clipped by a tile cut that another tile saw whole, NMS."""
    whole = xyxy[~at_cut]
    keep = np.ones(len(xyxy), bool)
    for i in np.flatnonzero(at_cut):
        b = xyxy[i]
        area = max(float((b[2] - b[0]) * (b[3] - b[1])), 1.0)
        iw = np.clip(np.minimum(whole[:, 2], b[2]) - np.maximum(whole[:, 0], b[0]), 0, None)
        ih = np.clip(np.minimum(whole[:, 3], b[3]) - np.maximum(whole[:, 1], b[1]), 0, None)
        if len(whole) and (iw * ih).max() / area > 0.5:
            keep[i] = False
    kept = np.flatnonzero(keep)
    return kept[nms_xyxy(xyxy[kept], scores[kept], iou)]


def scope_mask(mask: np.ndarray, regions: list[TextRegion]) -> np.ndarray:
    """Keep mask pixels near detected blocks only.

    The seg head also fires faintly on speed lines and undetected SFX; erasing those without
    putting a translation back would only damage the art.
    """
    H, W = mask.shape
    keep = np.zeros_like(mask)
    for r in regions:
        b = r.bbox.pad(max(4, (r.font_size or 0) // 2)).clamp(W, H)
        keep[b.y1 : b.y2, b.x1 : b.x2] = mask[b.y1 : b.y2, b.x1 : b.x2]
    return keep


def block_colors(
    image: np.ndarray, mask: np.ndarray | None, bbox: BBox
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]:
    """Median RGB of confident glyph pixels and of pixels well away from any glyph."""
    if mask is None:
        return None, None
    crop = image[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
    m = mask[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
    fg_px = crop[m > 200]
    near = cv2.dilate((m > 40).astype(np.uint8), np.ones((7, 7), np.uint8))
    bg_px = crop[near == 0]
    if len(fg_px) < 8 or len(bg_px) < 8:
        return None, None

    def rgb(px: np.ndarray) -> tuple[int, int, int]:
        b, g, r = np.median(px, axis=0).astype(int).tolist()
        return r, g, b

    return rgb(fg_px), rgb(bg_px)


def _heads(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return HEADS
    items = value.split(",") if isinstance(value, str) else list(value)
    heads = tuple(h.strip() for h in items if h.strip())
    bad = set(heads) - set(HEADS)
    if bad:
        raise ValueError(f"unknown ctd heads {sorted(bad)}, choose from {HEADS}")
    return heads


@DETECTORS.register(
    "ctd",
    description="comic-text-detector, all heads: blocks + DBNet text lines + glyph mask (~0.6s CPU)",
    weights=("ctd",),
)
def _build(
    settings,
    conf: float | None = None,
    iou: float = 0.35,
    mask_thresh: float | None = None,
    box_padding: int = 0,
    line_thresh: float = 0.3,
    line_box_thresh: float = 0.6,
    unclip: float = 1.5,
    split_gap: float = 2.0,
    keep_scattered: bool = True,
    heads: str | list[str] | None = None,
    threads: int = 0,
    tile_aspect: float = 2.5,
    mask_scope: str = "blocks",
    weights: str | None = None,
    spread: float = SPREAD_ASPECT,
    **extra,
):
    warn_unknown("detector 'ctd'", extra)
    return CtdFullDetector(
        Path(weights) if weights else ensure("ctd", settings),
        conf=settings.detect_conf if conf is None else conf,
        iou=iou,
        line_thresh=line_thresh,
        line_box_thresh=line_box_thresh,
        unclip=unclip,
        mask_thresh=mask_thresh,
        box_padding=box_padding,
        split_gap=split_gap,
        keep_scattered=keep_scattered,
        heads=_heads(heads),
        threads=threads,
        tile_aspect=tile_aspect,
        mask_scope=mask_scope,
        spread=spread,
    )
