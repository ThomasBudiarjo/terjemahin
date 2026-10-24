from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..models import BBox, TextRegion
from ..registry import DETECTORS
from .base import Detection

CTD_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/"
    "comictextdetector.pt.onnx"
)


class CtdDetector:
    """comic-text-detector's YOLOv5 box head only (blocks, no lines, no glyph mask).

    Runs `comictextdetector-blk.onnx` from training/prune_ctd.py (a full export also works: the
    other heads are cut out at load time when onnx is installed). Pre/post-processing is the
    same as the `ctd` method's block head: top-left zero-padded 1024 canvas, RGB 0..1, rows
    [cx, cy, w, h, obj, cls...], NMS at iou 0.35, so the boxes equal ctd's raw blocks.
    """

    def __init__(self, weights: Path, conf: float = 0.4, iou: float = 0.35) -> None:
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"CTD weights not found at {weights}. Run training/prune_ctd.py on "
                f"weights/koharu-comictextdetector.onnx (or {CTD_URL}); see weights/README.md."
            )
        from .ctd_full import make_session

        self.session = make_session(Path(weights), ("blk",))
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 1024
        self.conf = conf
        self.iou = iou

    def detect(self, image_bgr: np.ndarray) -> Detection:
        from .ctd_utils import decode_blocks

        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
        H, W = image_bgr.shape[:2]
        ratio = self.size / max(H, W)
        nw, nh = min(self.size, max(1, round(W * ratio))), min(self.size, max(1, round(H * ratio)))
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(image_bgr, (nw, nh), interpolation=interp)
        x = canvas[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        outputs = self.session.run(None, {self.input_name: x})
        blk = next(o for o in outputs if o.ndim == 3 and o.shape[-1] >= 6)
        xyxy, scores, _ = decode_blocks(blk, self.conf, self.iou)
        regions = []
        for (x0, y0, x1, y1), s in zip((xyxy / ratio).tolist(), scores.tolist()):
            bbox = BBox(
                x1=int(np.floor(x0)), y1=int(np.floor(y0)), x2=int(np.ceil(x1)), y2=int(np.ceil(y1))
            ).clamp(W, H)
            if bbox.w < 4 or bbox.h < 4:
                continue
            regions.append(TextRegion(id=len(regions), bbox=bbox, score=round(float(s), 4)))
        return Detection(regions)


@DETECTORS.register(
    "ctd_blk",
    description="comic-text-detector pruned to the box head (training/prune_ctd.py). Boxes only.",
)
def _build(settings, conf: float | None = None, iou: float = 0.35, weights=None, **_):
    return CtdDetector(
        Path(weights or settings.ctd_weights),
        settings.detect_conf if conf is None else conf,
        iou,
    )
