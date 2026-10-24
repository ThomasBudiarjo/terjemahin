from __future__ import annotations

from pathlib import Path

import numpy as np

from ..models import TextRegion
from ..registry import DETECTORS
from .base import Detection
from .ctd_full import CtdFullDetector

CTD_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/"
    "comictextdetector.pt.onnx"
)


class CtdDetector(CtdFullDetector):
    """comic-text-detector's YOLOv5 box head only (blocks, no lines, no glyph mask).

    Runs `comictextdetector-blk.onnx` from training/prune_ctd.py (a full export also works: the
    other heads are cut out at load time when onnx is installed). Pre/post-processing, tiling of
    tall pages and NMS (iou 0.35) are the `ctd` method's, so the boxes equal ctd's raw blocks
    before its text lines refine them. Regions are bare boxes + score: without the DB head the
    line quads, direction and font size ctd would infer are guesses from the box alone, and
    downstream stages (48px OCR line splitting, mask refine) do better estimating them from
    pixels themselves.
    """

    def __init__(self, weights: Path, conf: float = 0.4, iou: float = 0.35, **kw) -> None:
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"CTD weights not found at {weights}. Run training/prune_ctd.py on "
                f"weights/koharu-comictextdetector.onnx (or {CTD_URL}); see weights/README.md."
            )
        super().__init__(Path(weights), conf=conf, iou=iou, heads=("blk",), **kw)

    def detect(self, image_bgr: np.ndarray) -> Detection:
        det = super().detect(image_bgr)
        return Detection(
            [TextRegion(id=i, bbox=r.bbox, score=r.score) for i, r in enumerate(det.regions)]
        )


@DETECTORS.register(
    "ctd_blk",
    description="comic-text-detector pruned to the box head (training/prune_ctd.py). Boxes only.",
)
def _build(
    settings,
    conf: float | None = None,
    iou: float = 0.35,
    weights=None,
    threads: int = 0,
    tile_aspect: float = 2.5,
    **extra,
):
    from ..opts import warn_unknown

    warn_unknown("detector 'ctd_blk'", extra)
    return CtdDetector(
        Path(weights or settings.ctd_weights),
        settings.detect_conf if conf is None else conf,
        iou,
        threads=threads,
        tile_aspect=tile_aspect,
    )
