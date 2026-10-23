from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..models import TextRegion
from ..registry import DETECTORS
from .base import Detection
from .onnx_common import letterbox, xywh_to_regions


class YoloDetector:
    """Single-class text-region detector exported from Ultralytics YOLO (v8/v11) to ONNX.

    Expects the default export layout: input NCHW float [0,1], output (1, 4 + num_classes, N)
    with xywh in letterboxed pixel space.
    """

    def __init__(self, weights: Path, conf: float = 0.4, iou: float = 0.5) -> None:
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {weights}. Train one with training/train.py "
                "or set DETECTOR=ctd."
            )
        self.session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 1024
        self.conf = conf
        self.iou = iou

    def detect(self, image_bgr: np.ndarray) -> Detection:
        tensor, scale, pad = letterbox(image_bgr, self.size)
        (out,) = self.session.run(None, {self.input_name: tensor})
        return Detection(
            decode_ultralytics(out, scale, pad, image_bgr.shape[1::-1], self.conf, self.iou)
        )


@DETECTORS.register(
    "yolo",
    description="YOLO11n fine-tuned on Manga109 / pseudo-labels (training/train.py). Boxes only.",
)
def _build(settings, conf: float | None = None, iou: float | None = None, weights=None):
    return YoloDetector(
        Path(weights or settings.yolo_weights),
        settings.detect_conf if conf is None else conf,
        settings.detect_iou if iou is None else iou,
    )


def decode_ultralytics(
    out: np.ndarray,
    scale: float,
    pad: tuple[int, int],
    orig_wh: tuple[int, int],
    conf: float,
    iou: float,
) -> list[TextRegion]:
    pred = out[0].T  # (4+nc, N) -> (N, 4+nc)
    boxes = pred[:, :4]
    scores = pred[:, 4:].max(axis=1)
    return xywh_to_regions(boxes, scores, scale, pad, orig_wh, conf, iou)
