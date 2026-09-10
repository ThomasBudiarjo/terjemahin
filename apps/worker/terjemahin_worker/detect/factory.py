from __future__ import annotations

from ..config import Settings
from .base import Detector


def build_detector(settings: Settings) -> Detector:
    if settings.detector == "yolo":
        from .yolo import YoloDetector

        return YoloDetector(settings.yolo_weights, settings.detect_conf, settings.detect_iou)
    if settings.detector == "ctd":
        from .ctd import CtdDetector

        return CtdDetector(settings.ctd_weights, settings.detect_conf, settings.detect_iou)
    raise ValueError(f"unknown detector: {settings.detector}")
