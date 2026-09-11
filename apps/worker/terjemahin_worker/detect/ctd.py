from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..models import TextRegion
from .onnx_common import letterbox, xywh_to_regions

CTD_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/"
    "comictextdetector.pt.onnx"
)


class CtdDetector:
    """Fallback detector using comic-text-detector (YOLOv5 head) ONNX weights.

    Only the box head is used; the mask and line outputs are ignored because cleaning is
    done by background fill. Output layout follows YOLOv5: (1, N, 5 + num_classes) with
    [cx, cy, w, h, obj, cls...].
    """

    def __init__(self, weights: Path, conf: float = 0.4, iou: float = 0.5) -> None:
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"CTD weights not found at {weights}. Download {CTD_URL} and run "
                "training/prune_ctd.py (see weights/README.md)."
            )
        self.session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 1024
        self.conf = conf
        self.iou = iou

    def detect(self, image_bgr: np.ndarray) -> list[TextRegion]:
        tensor, scale, pad = letterbox(image_bgr, self.size)
        outputs = self.session.run(None, {self.input_name: tensor})
        blks = next(o for o in outputs if o.ndim == 3 and o.shape[-1] >= 6)[0]
        boxes = blks[:, :4]
        obj = blks[:, 4]
        cls = blks[:, 5:].max(axis=1) if blks.shape[1] > 5 else np.ones_like(obj)
        scores = obj * cls
        return xywh_to_regions(boxes, scores, scale, pad, image_bgr.shape[1::-1], self.conf, self.iou)
