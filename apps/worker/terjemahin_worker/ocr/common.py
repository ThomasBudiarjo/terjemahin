"""Crop, deskew, colour and text-normalisation helpers shared by OCR methods."""

from __future__ import annotations

import itertools
import re
import unicodedata

import cv2
import numpy as np

from ..models import RGB, BBox, TextRegion


def padded_box(
    region: TextRegion, width: int, height: int, pad: float = 0.12, min_pad: int = 4
) -> BBox:
    """Region box grown by `pad` x the glyph size (or short side), clamped to the page.

    Detector boxes hug the ink; recognisers read better with a little margin around the glyphs.
    """
    b = region.bbox
    size = region.font_size or min(b.w, b.h)
    px = max(min_pad, round(pad * size))
    return b.pad(px).clamp(width, height)


def crop_region(
    image_bgr: np.ndarray, region: TextRegion, pad: float = 0.12, min_pad: int = 4
) -> np.ndarray:
    """Padded crop of a region, rotated upright when the detector reports a tilt."""
    H, W = image_bgr.shape[:2]
    if abs(region.angle) < 1.0:
        b = padded_box(region, W, H, pad, min_pad)
        return image_bgr[b.y1 : b.y2, b.x1 : b.x2]
    return _deskew(image_bgr, region, pad, min_pad)


def _deskew(image_bgr: np.ndarray, region: TextRegion, pad: float, min_pad: int) -> np.ndarray:
    """Rotate the page around the region centre by -angle and cut the upright extent.

    `angle` is the counter-clockwise tilt of the text (see detect/ctd_utils.line_geometry), so a
    clockwise rotation of the same amount makes it upright. The upright extent comes from the
    rotated line quads when available, else from the rotated bbox corners.
    """
    b = region.bbox
    cx, cy = b.cx, b.cy
    M = cv2.getRotationMatrix2D((cx, cy), -region.angle, 1.0)
    pts = (
        np.array([p for q in region.lines for p in q], np.float32)
        if region.lines
        else np.array([[b.x1, b.y1], [b.x2, b.y1], [b.x2, b.y2], [b.x1, b.y2]], np.float32)
    )
    rot = pts @ M[:, :2].T + M[:, 2]
    x1, y1 = rot.min(0)
    x2, y2 = rot.max(0)
    size = region.font_size or min(x2 - x1, y2 - y1)
    px = max(min_pad, pad * size)
    x1, y1, x2, y2 = x1 - px, y1 - px, x2 + px, y2 + px
    w, h = max(1, round(x2 - x1)), max(1, round(y2 - y1))
    M[:, 2] -= (x1, y1)
    border = tuple(int(v) for v in np.median(image_bgr[b.y1 : b.y2, b.x1 : b.x2].reshape(-1, 3), 0))
    return cv2.warpAffine(
        image_bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def estimate_colors(crop_bgr: np.ndarray) -> tuple[RGB, RGB]:
    """(glyph, background) colours of a text crop, as RGB.

    Otsu splits luminance into two classes; glyph strokes cover less area than the paper, so
    the minority class is the glyph. Pixels near the threshold (anti-aliased edges, outlines)
    are dropped so the medians reflect solid stroke and paper colour.
    """
    if crop_bgr.size == 0:
        return (0, 0, 0), (255, 255, 255)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = gray <= t  # cv2 marks > t as foreground; on two-level images t equals the dark level
    light = ~dark
    if not dark.any() or not light.any():
        c = _rgb(np.median(crop_bgr.reshape(-1, 3), 0))
        return c, c
    fg_is_dark = dark.sum() <= light.sum()
    fg, bg = (dark, light) if fg_is_dark else (light, dark)
    spread = max(8.0, 0.25 * float(abs(np.median(gray[light]) - np.median(gray[dark]))))
    core_fg = fg & (np.abs(gray.astype(np.float32) - t) > spread)
    core_bg = bg & (np.abs(gray.astype(np.float32) - t) > spread)
    fg = core_fg if core_fg.sum() >= 8 else fg
    bg = core_bg if core_bg.sum() >= 8 else bg
    return _rgb(np.median(crop_bgr[fg], 0)), _rgb(np.median(crop_bgr[bg], 0))


def _rgb(bgr: np.ndarray) -> RGB:
    b, g, r = (round(float(v)) for v in bgr)
    return (r, g, b)


_HALF_KANA = re.compile("[\\uff61-\\uff9f]+")


def to_fullwidth(text: str) -> str:
    """Half-width ASCII, digits and katakana -> full width (what jaconv.h2z does for manga-ocr)."""
    text = _HALF_KANA.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)
    return "".join(
        chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else ("　" if c == " " else c) for c in text
    )


def normalize_ja(text: str) -> str:
    """manga-ocr's post-processing: drop whitespace, unify ellipses, widen half-width chars."""
    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda m: (m.end() - m.start()) * ".", text)
    return to_fullwidth(text)


def ink_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Boolean glyph mask: the minority Otsu class of the grey crop."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = binary == 0
    return dark if dark.sum() <= dark.size / 2 else ~dark


def split_long(crop_bgr: np.ndarray, vertical: bool, chunk_aspect: float = 4.0) -> list[np.ndarray]:
    """Cut a very long single line into pieces of about `chunk_aspect`, between glyphs.

    Each cut lands on the least-inked row (column for horizontal text) within half a glyph of
    the even split point, so no character is sliced in two.
    """
    img = crop_bgr if vertical else crop_bgr.transpose(1, 0, 2)
    h, w = img.shape[:2]
    n = int(np.ceil(h / (chunk_aspect * w)))
    if n <= 1:
        return [crop_bgr]
    profile = ink_mask(img).sum(1).astype(np.float32)
    cuts, prev = [0], 0
    for k in range(1, n):
        lo = max(prev + w // 2, int(k * h / n - w / 2))
        hi = min(h - w // 2, int(k * h / n + w / 2))
        if hi <= lo:
            continue
        prev = lo + int(np.argmin(profile[lo:hi]))
        cuts.append(prev)
    cuts.append(h)
    pieces = [img[a:b] for a, b in itertools.pairwise(cuts) if b - a > 2]
    return pieces if vertical else [p.transpose(1, 0, 2) for p in pieces]


def text_pieces(
    image_bgr: np.ndarray,
    region: TextRegion,
    pad: float = 0.12,
    split_aspect: float = 10.0,
    chunk_aspect: float = 4.0,
) -> list[np.ndarray]:
    """Crops to recognise for one region, in reading order; usually just the padded block.

    Whole-block recognisers (manga-ocr) read a full bubble best, but fail on extremely long
    blocks (a 1:15 column comes out as garbage). Those are read line by line, and each overlong
    line is cut into chunks; the caller joins the texts.
    """
    crop = crop_region(image_bgr, region, pad)
    h, w = crop.shape[:2]
    vertical = region.direction == "v"
    long_side, short_side = (h, w) if vertical else (w, h)
    if split_aspect <= 0 or long_side <= split_aspect * short_side:
        return [crop]
    lines = [crop]
    if len(region.lines) > 1 and abs(region.angle) < 1.0:
        order = sorted(
            region.lines,
            key=lambda q: -sum(p[0] for p in q) if vertical else sum(p[1] for p in q),
        )
        lines = [crop_region(image_bgr, _line_region(region, q), pad) for q in order]
    return [p for line in lines for p in split_long(line, vertical, chunk_aspect)]


def _line_region(region: TextRegion, quad: list[tuple[int, int]]) -> TextRegion:
    xs, ys = [p[0] for p in quad], [p[1] for p in quad]
    return region.model_copy(
        update={
            "bbox": BBox(x1=min(xs), y1=min(ys), x2=max(xs) + 1, y2=max(ys) + 1),
            "lines": [quad],
        }
    )
