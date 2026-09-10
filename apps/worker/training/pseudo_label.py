"""Pseudo-label a pool of manga pages with an existing detector and write a YOLO dataset.

This distils a heavier teacher (comic-text-detector by default) into the small YOLO student
that the worker runs on CPU. Review the `review/` visualisations and delete or fix bad
labels before training; pages with zero detections are skipped.

Usage:
    DETECTOR=ctd python training/pseudo_label.py --pool data/pool --out data/pseudo-text
    python training/train.py --data data/pseudo-text/dataset.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terjemahin_worker.config import get_settings  # noqa: E402
from terjemahin_worker.detect import build_detector  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True, help="folder of page images")
    ap.add_argument("--out", type=Path, default=Path("data/pseudo-text"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--conf", type=float, default=0.5, help="teacher confidence threshold")
    ap.add_argument("--min-boxes", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--review", type=int, default=50, help="how many pages to visualise")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    settings = get_settings()
    settings.detect_conf = a.conf
    detector = build_detector(settings)

    images = sorted(p for p in a.pool.iterdir() if p.suffix.lower() in IMG_EXT)
    if a.limit:
        images = images[: a.limit]
    rng = random.Random(a.seed)
    rng.shuffle(images)

    for s in ("train", "val"):
        (a.out / "images" / s).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / s).mkdir(parents=True, exist_ok=True)
    review_dir = a.out / "review"
    review_dir.mkdir(exist_ok=True)

    n_val = round(len(images) * a.val_ratio)
    kept = boxes_total = 0
    for i, src in enumerate(images):
        img = cv2.imread(str(src))
        if img is None:
            continue
        H, W = img.shape[:2]
        regions = detector.detect(img)
        if len(regions) < a.min_boxes:
            continue

        split = "val" if i < n_val else "train"
        lines = []
        for r in regions:
            b = r.bbox
            lines.append(f"0 {b.cx / W:.6f} {b.cy / H:.6f} {b.w / W:.6f} {b.h / H:.6f}")
        dst = a.out / "images" / split / (src.stem + ".jpg")
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (a.out / "labels" / split / (src.stem + ".txt")).write_text("\n".join(lines) + "\n")

        if kept < a.review:
            vis = img.copy()
            for r in regions:
                b = r.bbox
                cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), 2)
                cv2.putText(vis, f"{r.score:.2f}", (b.x1, max(12, b.y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(review_dir / (src.stem + ".jpg")), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])

        kept += 1
        boxes_total += len(regions)
        if kept % 50 == 0:
            print(f"{kept} pages, {boxes_total} boxes", flush=True)

    (a.out / "dataset.yaml").write_text(
        f"path: {a.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: text\n"
    )
    print(f"done: {kept}/{len(images)} pages kept, {boxes_total} boxes, "
          f"avg {boxes_total / max(kept, 1):.1f}/page")
    print(f"review visualisations: {review_dir}")
    print(f"dataset yaml: {a.out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
