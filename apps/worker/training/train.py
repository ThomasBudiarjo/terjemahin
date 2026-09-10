"""Fine-tune YOLO11n on the Manga109 text dataset and export ONNX for the worker.

    uv sync --extra train
    python training/train.py --data data/manga109-text/dataset.yaml --epochs 60

Text in manga is small relative to the page, so we train at imgsz=1024 rather than the
default 640. On an RTX 4060 (8GB) batch 8 at 1024 fits for the nano model.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="manga-text-yolo11n")
    ap.add_argument("--out", type=Path, default=Path("weights/manga-text-yolo11n.onnx"))
    a = ap.parse_args()

    model = YOLO(a.model)
    model.train(
        data=str(a.data),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        project="training/runs",
        name=a.name,
        exist_ok=True,
        # manga pages are not photos: no colour jitter, mild geometric aug only
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
        fliplr=0.0,  # flipping breaks right-to-left reading cues in text shapes
        mosaic=0.5,
        degrees=2.0,
    )

    best = Path("training/runs") / a.name / "weights" / "best.pt"
    onnx_path = YOLO(str(best)).export(format="onnx", imgsz=a.imgsz, opset=17, simplify=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, a.out)
    print(f"exported -> {a.out}")


if __name__ == "__main__":
    main()
