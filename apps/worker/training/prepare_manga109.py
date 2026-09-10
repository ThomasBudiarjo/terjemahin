"""Convert Manga109 annotations to a single-class Ultralytics YOLO dataset.

Manga109 layout (after you get access at http://www.manga109.org/):
    Manga109/
      annotations/<Title>.xml
      images/<Title>/<page:03d>.jpg

Only <text> boxes are used (one class: "text"). Pages are double-page spreads; we keep
them as-is because the detector will see whole pages at inference too.

Usage:
    python prepare_manga109.py --root /path/to/Manga109 --out data/manga109-text --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path


def convert(root: Path, out: Path, val_ratio: float, seed: int, link: bool) -> None:
    ann_dir, img_dir = root / "annotations", root / "images"
    titles = sorted(p.stem for p in ann_dir.glob("*.xml"))
    if not titles:
        raise SystemExit(f"no annotations found in {ann_dir}")

    # split by title so validation pages come from unseen series
    rng = random.Random(seed)
    rng.shuffle(titles)
    n_val = max(1, round(len(titles) * val_ratio))
    split = {t: "val" for t in titles[:n_val]} | {t: "train" for t in titles[n_val:]}

    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    n_pages = n_boxes = 0
    for title in titles:
        tree = ET.parse(ann_dir / f"{title}.xml")
        for page in tree.getroot().iter("page"):
            idx = int(page.get("index"))
            W, H = int(page.get("width")), int(page.get("height"))
            src = img_dir / title / f"{idx:03d}.jpg"
            if not src.exists():
                continue
            lines = []
            for t in page.iter("text"):
                x1, y1 = int(t.get("xmin")), int(t.get("ymin"))
                x2, y2 = int(t.get("xmax")), int(t.get("ymax"))
                if x2 <= x1 or y2 <= y1:
                    continue
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if not lines:
                continue  # skip text-free pages; they add little for a single-class detector
            s = split[title]
            stem = f"{title}_{idx:03d}"
            dst = out / "images" / s / f"{stem}.jpg"
            if not dst.exists():
                if link:
                    os.symlink(src.resolve(), dst)
                else:
                    dst.write_bytes(src.read_bytes())
            (out / "labels" / s / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            n_pages += 1
            n_boxes += len(lines)

    (out / "dataset.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: text\n"
    )
    print(f"pages={n_pages} boxes={n_boxes} val_titles={n_val}/{len(titles)}")
    print(f"dataset yaml: {out / 'dataset.yaml'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/manga109-text"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    a = ap.parse_args()
    convert(a.root, a.out, a.val_ratio, a.seed, link=not a.copy)
