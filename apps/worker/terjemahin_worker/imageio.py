"""Reading pages: every entry point (CLI, web, queue consumer, Pipeline API) ends up with an
HxWx3 uint8 BGR array.

cv2.imread(IMREAD_COLOR) drops an alpha channel without compositing, so the transparent
pixels of a PNG take whatever RGB they store (usually black) and white balloons turn black.
Pages with alpha are composited over white here instead. Sizes are checked from the header
before decoding, so a small file that decodes to a huge page is refused before it takes
gigabytes of memory.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np


class ImageError(ValueError):
    """The data is not a readable image, or it is too large to process."""


def composite_white(img: np.ndarray) -> np.ndarray:
    """BGRA (or grey+alpha) uint8 -> BGR over a white background."""
    color, alpha = img[..., :-1], img[..., -1:].astype(np.float32) / 255.0
    if color.shape[-1] == 1:
        color = np.repeat(color, 3, axis=-1)
    out = color.astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def to_bgr8(img: np.ndarray) -> np.ndarray:
    """Normalise an array to HxWx3 uint8 BGR: greyscale (HxW, HxWx1), BGRA / grey+alpha
    (alpha composited over white), 16-bit, bool, and float pages (0..1 or 0..255)."""
    a = np.asarray(img)
    if a.ndim not in (2, 3) or a.size == 0 or (a.ndim == 3 and a.shape[2] not in (1, 2, 3, 4)):
        raise ImageError(f"expected an HxW or HxWxC image (C = 1..4), got shape {a.shape}")
    if a.dtype == np.uint8:
        pass
    elif a.dtype == bool:
        a = a.astype(np.uint8) * 255
    elif a.dtype == np.uint16:
        a = (a >> 8).astype(np.uint8)
    elif np.issubdtype(a.dtype, np.floating):
        finite = a[np.isfinite(a)]
        scale = 255.0 if finite.size and float(finite.max()) <= 1.0 else 1.0
        a = np.clip(np.nan_to_num(a.astype(np.float32)) * scale + 0.5, 0, 255).astype(np.uint8)
    elif np.issubdtype(a.dtype, np.integer):
        a = np.clip(a, 0, 255).astype(np.uint8)
    else:
        raise ImageError(f"unsupported image dtype {a.dtype}")
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    elif a.shape[2] == 1:
        a = cv2.cvtColor(a[..., 0], cv2.COLOR_GRAY2BGR)
    elif a.shape[2] in (2, 4):
        a = composite_white(a)
    return np.ascontiguousarray(a)


def _header(data: bytes) -> tuple[tuple[int, int] | None, bool]:
    """((width, height) or None, has transparency) from the header, without decoding."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            alpha = im.mode in ("RGBA", "LA", "PA", "La", "RGBa") or (
                im.mode == "P" and "transparency" in im.info
            )
            return im.size, alpha
    except Exception:  # noqa: BLE001 - PIL cannot read it: let OpenCV try
        return None, False


def decode_image(data: bytes, max_pixels: int | None = None, name: str = "image") -> np.ndarray:
    """Encoded bytes -> HxWx3 uint8 BGR (alpha composited over white, EXIF rotation applied
    as cv2.imread does). Raises ImageError for empty, unreadable or oversized data."""
    if not data:
        raise ImageError(f"{name} is empty")
    size, alpha = _header(data)
    if size and max_pixels and size[0] * size[1] > max_pixels:
        raise ImageError(
            f"{name} is {size[0]}x{size[1]} = {size[0] * size[1] / 1e6:.0f} MP, more than the "
            f"{max_pixels / 1e6:.0f} MP limit (MAX_PIXELS)"
        )
    buf = np.frombuffer(data, np.uint8)
    flag = cv2.IMREAD_UNCHANGED if alpha else cv2.IMREAD_COLOR
    try:
        img = cv2.imdecode(buf, flag)
    except cv2.error:
        img = None
    if img is None:
        raise ImageError(f"{name} is not a readable image")
    if max_pixels and img.shape[0] * img.shape[1] > max_pixels:
        raise ImageError(f"{name} is {img.shape[1]}x{img.shape[0]}, more than the limit (MAX_PIXELS)")
    return to_bgr8(img)


def read_image(path: str | Path, max_pixels: int | None = None) -> np.ndarray:
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as e:
        raise ImageError(f"cannot read {p}: {e.strerror or e}") from None
    return decode_image(data, max_pixels, name=str(p))
