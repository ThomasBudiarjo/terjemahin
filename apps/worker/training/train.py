"""Fine-tune YOLO11n as a one-class text-block detector and export ONNX for the worker (`-d yolo`).

    uv sync --extra train
    python training/train.py --data data/manga109-text/dataset.yaml --epochs 60
    # several datasets (e.g. synth + pseudo-labels) are merged into one yaml:
    python training/train.py --data data/synth/det/dataset.yaml data/pseudo/dataset.yaml \
        --imgsz 640 --epochs 30 --time 0.6

Text in manga is small relative to the page, so imgsz=1024 is the default rather than 640; on
an RTX 4060 (8GB) batch 8 at 1024 fits for the nano model. On CPU, 640 is ~2.5x faster.

The pretrained checkpoint (`yolo11n.pt`) is fetched from the Hugging Face mirror
`Ultralytics/YOLO11` when it is not a local file (Ultralytics' own GitHub download is often
blocked). The ONNX is exported at the training size with a fixed 1x3xSxS input, which
`detect/yolo.py` reads back to letterbox pages the same way.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HF_MIRROR = "Ultralytics/YOLO11"


def resolve_model(name: str) -> str:
    """Local path as is; a bare `yolo11*.pt` name is downloaded from the HF mirror."""
    p = Path(name)
    if p.exists() or p.suffix != ".pt" or p.parent != Path("."):
        return name
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return hf_hub_download(HF_MIRROR, p.name)
    except (OSError, EntryNotFoundError) as e:  # not on the mirror / offline: ultralytics' source
        print(f"note: {p.name} not on hf:{HF_MIRROR} ({type(e).__name__}); trying ultralytics")
        return name


def merge_datasets(paths: list[Path], out: Path) -> Path:
    """One yaml whose train/val lists every dataset's split dirs (single class 'text')."""
    if len(paths) == 1:
        return paths[0]
    train, val = [], []
    for p in paths:
        cfg = yaml.safe_load(p.read_text())
        base = Path(cfg.get("path") or p.parent)
        if not base.is_absolute():
            base = (p.parent / base).resolve()
        for key, acc in (("train", train), ("val", val)):
            items = cfg.get(key) or []
            for item in [items] if isinstance(items, str) else items:
                d = base / item
                if d.exists() and any(d.iterdir()):
                    acc.append(str(d))
    if not train or not val:
        raise SystemExit(f"merged datasets have no train or val images: {paths}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({"train": train, "val": val, "names": {0: "text"}}))
    return out


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="YOLO dataset yaml(s)")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--time", type=float, default=None,
                    help="wall-clock budget in hours (overrides --epochs, LR schedule fits it)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cpu, 0, ... (default: ultralytics picks)")
    ap.add_argument("--workers", type=int, default=2, help="dataloader workers")
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plots", action="store_true",
                    help="write ultralytics plots (downloads a font from GitHub on first use)")
    ap.add_argument("--project", type=Path, default=ROOT / "training" / "runs")
    ap.add_argument("--name", default="manga-text-yolo11n")
    ap.add_argument("--out", type=Path, default=ROOT / "weights" / "manga-text-yolo11n.onnx")
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    # ultralytics nests a relative project under its global runs_dir/<task>/
    a.project = a.project.resolve()
    data = merge_datasets(a.data, a.project / a.name / "data.yaml")
    model = YOLO(resolve_model(a.model))
    extra = {"time": a.time} if a.time else {}
    if a.device is not None:
        extra["device"] = a.device
    model.train(
        data=str(data),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        workers=a.workers,
        patience=a.patience,
        seed=a.seed,
        project=str(a.project),
        name=a.name,
        exist_ok=True,
        plots=a.plots,
        single_cls=True,
        # manga pages are not photos: no colour jitter, mild geometric aug only
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
        fliplr=0.0,  # flipping breaks right-to-left reading cues in text shapes
        mosaic=0.5,
        degrees=2.0,
        **extra,
    )

    best = Path(model.trainer.best)
    if not best.exists():
        raise SystemExit(f"training produced no checkpoint at {best}")
    print(f"best checkpoint: {best}")
    simplify = importlib.util.find_spec("onnxslim") is not None
    onnx_path = YOLO(str(best)).export(
        format="onnx", imgsz=a.imgsz, opset=17, simplify=simplify, dynamic=False
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, a.out)
    print(f"exported -> {a.out}")
    return a.out


if __name__ == "__main__":
    main()
