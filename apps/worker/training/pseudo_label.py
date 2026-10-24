"""Pseudo-label a pool of manga pages with an existing detector and write a YOLO dataset.

This distils a heavier teacher (comic-text-detector by default) into the small YOLO student
that the worker runs on CPU. Review the `review/` visualisations and delete or fix bad
labels before training; pages with zero detections are skipped.

`--augment N` adds N random views of every page (crop, rescale, contrast, blur, noise, JPEG)
and runs the teacher on each view, so the labels match what the student sees. That stretches a
handful of real pages into a usable dataset. Views of one page go to one split only.

Usage:
    python training/pseudo_label.py --pool data/pool --out data/pseudo-text
    python training/pseudo_label.py --pool page.png --augment 60 --out data/pseudo-page
    python training/train.py --data data/pseudo-text/dataset.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terjemahin_worker.config import get_settings
from terjemahin_worker.detect import DETECTORS

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(pool: Path) -> list[Path]:
    if pool.is_file():
        return [pool]
    return sorted(p for p in pool.rglob("*") if p.suffix.lower() in IMG_EXT)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random view of a page: a crop (at least ~1/3 of each side), rescale, photometric."""
    H, W = img.shape[:2]
    if rng.random() < 0.2:
        crop = img
    else:
        s = rng.uniform(0.35, 0.9)
        ch = int(H * s)
        cw = int(min(W, ch * rng.uniform(0.6, 1.0) * W / H))
        y, x = int(rng.integers(0, H - ch + 1)), int(rng.integers(0, W - cw + 1))
        crop = img[y : y + ch, x : x + cw]
    # long side 640..1600 px: covers scans from phone screenshots to print resolution
    long = rng.uniform(640, 1600)
    f = long / max(crop.shape[:2])
    out = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    out = out.astype(np.float32)
    out = (out - 128) * rng.uniform(0.7, 1.2) + 128 + rng.uniform(-30, 30)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.4))
    if rng.random() < 0.3:
        out += rng.normal(0, rng.uniform(2, 10), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        q = int(rng.integers(40, 90))
        out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True, help="folder of page images, or one image")
    ap.add_argument("--out", type=Path, default=Path("data/pseudo-text"))
    ap.add_argument("--detector", default="ctd", help="teacher: any worker detector method")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--conf", type=float, default=0.5, help="teacher confidence threshold")
    ap.add_argument("--min-boxes", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--augment", type=int, default=0, help="extra random views per page")
    ap.add_argument("--no-original", action="store_true", help="only write the augmented views")
    ap.add_argument("--val-split", choices=("auto", "train", "val"), default="auto",
                    help="force every page into one split (e.g. keep an eval page out of val)")
    ap.add_argument("--prefix", default="", help="prepended to output file names")
    ap.add_argument("--review", type=int, default=50, help="how many images to visualise")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    settings = get_settings()
    try:
        detector = DETECTORS.build(a.detector, settings, conf=a.conf)
    except TypeError:  # teacher without a confidence option (e.g. classical)
        print(f"note: detector '{a.detector}' takes no conf, --conf ignored", file=sys.stderr)
        detector = DETECTORS.build(a.detector, settings)

    images = list_images(a.pool)
    if a.limit:
        images = images[: a.limit]
    random.Random(a.seed).shuffle(images)

    for s in ("train", "val"):
        (a.out / "images" / s).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / s).mkdir(parents=True, exist_ok=True)
    review_dir = a.out / "review"
    review_dir.mkdir(exist_ok=True)

    n_val = round(len(images) * a.val_ratio)
    kept = boxes_total = seen = 0
    for i, src in enumerate(images):
        page = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if page is None:
            continue
        split = a.val_split if a.val_split != "auto" else ("val" if i < n_val else "train")
        rng = np.random.default_rng([a.seed, i])
        views = [] if a.no_original else [("", page)]
        views += [(f"_a{k:03d}", augment(page, rng)) for k in range(a.augment)]
        for suffix, img in views:
            seen += 1
            H, W = img.shape[:2]
            regions = detector.detect(img).regions
            if len(regions) < a.min_boxes:
                continue
            stem = f"{a.prefix}{src.stem}{suffix}"
            lines = [
                f"0 {r.bbox.cx / W:.6f} {r.bbox.cy / H:.6f} {r.bbox.w / W:.6f} {r.bbox.h / H:.6f}"
                for r in regions
            ]
            cv2.imwrite(str(a.out / "images" / split / f"{stem}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (a.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")

            if kept < a.review:
                vis = img.copy()
                for r in regions:
                    b = r.bbox
                    cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), 2)
                    cv2.putText(vis, f"{r.score:.2f}", (b.x1, max(12, b.y1 - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imwrite(str(review_dir / f"{stem}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])

            kept += 1
            boxes_total += len(regions)
            if kept % 50 == 0:
                print(f"{kept} images, {boxes_total} boxes", flush=True)

    (a.out / "dataset.yaml").write_text(
        f"path: {a.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: text\n"
    )
    print(f"done: {kept}/{seen} images kept ({len(images)} pages), {boxes_total} boxes, "
          f"avg {boxes_total / max(kept, 1):.1f}/image")
    print(f"review visualisations: {review_dir}")
    print(f"dataset yaml: {a.out / 'dataset.yaml'}")
    return {"images": kept, "boxes": boxes_total, "pages": len(images)}


if __name__ == "__main__":
    main()
