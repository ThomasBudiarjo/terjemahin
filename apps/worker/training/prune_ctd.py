"""Strip the mask/line heads from comic-text-detector so only the box head remains.

The worker only reads the `blk` output, but onnxruntime still executes the `seg` and `det`
decoders, whose ConvTranspose layers dominate CPU time (~29s vs ~0.15s per 1024px page).
Cutting the graph at `blk` also shrinks the file from ~90MB to ~27MB.

Usage:
    python training/prune_ctd.py                      # weights/comictextdetector.pt.onnx -> weights/comictextdetector-blk.onnx
                                                      # (or weights/koharu-comictextdetector.onnx, the
                                                      # file the `ctd` method downloads, if that is absent)
    python training/prune_ctd.py --src a.onnx --out b.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
SOURCES = ("comictextdetector.pt.onnx", "koharu-comictextdetector.onnx")


def default_src() -> Path:
    """First full export present in weights/ (both have the same `images` -> `blk` head)."""
    for name in SOURCES:
        if (WEIGHTS / name).exists():
            return WEIGHTS / name
    return WEIGHTS / SOURCES[0]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None, help=f"default: first of {SOURCES} in weights/")
    ap.add_argument("--out", type=Path, default=WEIGHTS / "comictextdetector-blk.onnx")
    a = ap.parse_args(argv)
    a.src = a.src or default_src()

    if not a.src.exists():
        raise SystemExit(f"{a.src} not found; see weights/README.md for the download URL")
    onnx.utils.extract_model(str(a.src), str(a.out), ["images"], ["blk"])
    onnx.checker.check_model(str(a.out))
    print(f"{a.src} ({a.src.stat().st_size >> 20}MB) -> {a.out} ({a.out.stat().st_size >> 20}MB)")


if __name__ == "__main__":
    main()
