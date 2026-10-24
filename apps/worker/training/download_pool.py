"""Download an unlabeled pool of manga pages for pseudo-labelling.

Default source: PSImera/manga_bubbles_detect on Hugging Face (2,671 pages, CC BY-NC 4.0, no
gating). Only the images are used; its labels mark whole speech bubbles, not text, so they
do not match our target class.

Usage:
    python training/download_pool.py --out data/pool
    python training/download_pool.py --out data/pool --limit 300   # quick experiment
    python training/download_pool.py --out data/pool --split test  # 211 pages, ~70MB download
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "PSImera/manga_bubbles_detect"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/pool"))
    ap.add_argument("--limit", type=int, default=0, help="copy at most N images (0 = all)")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--split", default="", help="only download images/<split>/ (train/val/test)")
    a = ap.parse_args()

    pattern = f"images/{a.split}/**" if a.split else "images/**"
    cache = snapshot_download(a.repo, repo_type="dataset", allow_patterns=[pattern])
    root = Path(cache) / "images" / a.split if a.split else Path(cache) / "images"
    imgs = sorted(root.glob("**/*.jp*g")) + sorted(root.glob("**/*.png"))
    if a.limit:
        imgs = imgs[: a.limit]

    a.out.mkdir(parents=True, exist_ok=True)
    for p in imgs:
        dst = a.out / f"{p.parent.name}_{p.name}"
        if not dst.exists():
            shutil.copy(p, dst)
    print(f"{len(imgs)} images -> {a.out}")


if __name__ == "__main__":
    main()
