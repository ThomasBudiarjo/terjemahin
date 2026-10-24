"""Proof image helpers: downscaled JPEGs under a size budget, side-by-sides, mask overlays."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

MAX_BYTES = 390 * 1024


def read(p) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(p)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def fit_w(img: np.ndarray, w: int) -> np.ndarray:
    if img.shape[1] <= w:
        return img
    h = round(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def save_jpg(img: np.ndarray, out, max_w: int = 1400, max_bytes: int = MAX_BYTES) -> int:
    """Write a JPEG <= max_bytes: lower quality first, then shrink."""
    img = fit_w(img, max_w)
    while True:
        for q in (88, 82, 76, 70, 62):
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok and len(buf) <= max_bytes:
                Path(out).write_bytes(buf.tobytes())
                return len(buf)
        img = fit_w(img, int(img.shape[1] * 0.85))


def label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((44, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def side(images: list[np.ndarray], labels: list[str], h: int = 1100, gap: int = 12) -> np.ndarray:
    tiles = []
    for im, lb in zip(images, labels):
        w = round(im.shape[1] * h / im.shape[0])
        tiles.append(label(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA), lb))
    sep = np.full((tiles[0].shape[0], gap, 3), 200, np.uint8)
    row = [tiles[0]]
    for t in tiles[1:]:
        row += [sep, t]
    return np.hstack(row)


def overlay(page: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask if mask.ndim == 2 else mask[..., 0]
    out = page.copy()
    red = np.zeros_like(page)
    red[..., 2] = 255
    sel = m > 0
    out[sel] = (0.45 * page[sel] + 0.55 * red[sel]).astype(np.uint8)
    return out


if __name__ == "__main__":
    # usage: mk.py jpg IN OUT [max_w]
    cmd = sys.argv[1]
    if cmd == "jpg":
        n = save_jpg(read(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1400)
        print(sys.argv[3], n)
