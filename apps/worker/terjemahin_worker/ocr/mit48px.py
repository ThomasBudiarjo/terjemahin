"""manga-image-translator's 48px autoregressive OCR: per-text-line reading with colour prediction.

Every region is cut into text lines (the detector's line quads, or a projection-profile split
of the box when it gave none), each line is warped to 48px high (vertical columns are rotated
so they read left to right), lines are batched by width and beam-searched, and the line texts
are joined in reading order. The model also predicts each character's glyph colour and outline
colour; those are averaged per region and returned for the renderer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..models import RGB, TextRegion
from ..registry import OCRS
from ..weights import ensure
from .base import OcrResult

TEXT_HEIGHT = 48
MAX_LINE_WIDTH = TEXT_HEIGHT * 256  # warp cap; longer lines are squeezed
# The model collapses (stops early or loops, still confident) on lines wider than ~2100px, so
# longer lines are cut at blank columns into pieces of at most PIECE_WIDTH.
SPLIT_WIDTH = TEXT_HEIGHT * 32
PIECE_WIDTH = TEXT_HEIGHT * 26
CLOSERS = ".,!?;:)]}\"'%…"
MAX_GAP = 3.0  # in line thicknesses: wider blank stretches along a line end it


def order_quad(q: np.ndarray) -> np.ndarray:
    """Points as tl, tr, br, bl (clockwise in image coords) whatever order they came in."""
    q = np.asarray(q, np.float32).reshape(4, 2)
    c = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
    return np.roll(q, -int(np.argmin(q.sum(1))), 0)


def line_direction(q: np.ndarray, default: str) -> str:
    """Clearly elongated quads say their own direction; near-square ones follow the block."""
    along_x = np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])
    along_y = np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])
    if along_y > 1.3 * along_x:
        return "v"
    if along_x > 1.3 * along_y:
        return "h"
    return default


def warp_line(image_rgb: np.ndarray, q: np.ndarray, direction: str, pad: float = 0.0) -> np.ndarray:
    """Rectify one line quad to TEXT_HEIGHT px high; vertical lines are rotated 90 deg CCW so
    the first character ends up on the left, as the model was trained. `pad` grows the quad by
    that fraction of the line thickness on every side."""
    q = order_quad(q)
    ex = ((q[1] - q[0]) + (q[2] - q[3])) / 2  # along the x-ish edges
    ey = ((q[3] - q[0]) + (q[2] - q[1])) / 2
    nx, ny = max(float(np.linalg.norm(ex)), 1.0), max(float(np.linalg.norm(ey)), 1.0)
    if pad:
        m = pad * (nx if direction == "v" else ny)
        ux, uy = ex / nx * m, ey / ny * m
        q = q + np.stack([-ux - uy, ux - uy, ux + uy, uy - ux])
        nx, ny = nx + 2 * m, ny + 2 * m
    across, along = (nx, ny) if direction == "v" else (ny, nx)
    length = int(np.clip(round(TEXT_HEIGHT * along / across), 2, MAX_LINE_WIDTH))
    w, h = (TEXT_HEIGHT, length) if direction == "v" else (length, TEXT_HEIGHT)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    out = cv2.warpPerspective(
        image_rgb,
        M,
        (w, h),
        flags=cv2.INTER_AREA if across > TEXT_HEIGHT else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE) if direction == "v" else out


def _runs(on: np.ndarray) -> list[tuple[int, int]]:
    d = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(np.int8), [0]])))
    return list(zip(d[::2].tolist(), d[1::2].tolist()))


def glyph_mask(
    crop: np.ndarray, font_size: int | None = None, vertical: bool = False
) -> np.ndarray:
    """0/1 mask of lettering in a grey crop: Otsu ink of the minority polarity, minus connected
    components too thick ACROSS the line direction to be lettering (balloon edges, tails,
    screentone, art). Extent along the line is unbounded: dash runs (――), kerned or touching
    Latin pairs and bold glyphs fused into one blob are all still text."""
    _, ink = cv2.threshold(crop, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([ink[0], ink[-1], ink[:, 0], ink[:, -1]])
    if border.mean() > 0.5:  # background is the bright class: ink is the dark one
        ink = 1 - ink
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    if n <= 1:
        return ink
    across = stats[1:, cv2.CC_STAT_WIDTH if vertical else cv2.CC_STAT_HEIGHT]
    # glyph size from each component's larger side: narrow marks (!, dashes, kana strokes) may
    # outnumber full glyphs, and their across extent alone would make real glyphs look "big"
    size = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
    big = size[stats[1:, cv2.CC_STAT_AREA] >= 12]
    fs = font_size or (float(np.percentile(big, 75)) if len(big) else min(crop.shape))
    keep = np.concatenate([[0], (across <= 1.6 * fs).astype(np.uint8)])
    return keep[labels]


def split_lines(gray: np.ndarray, region: TextRegion) -> list[np.ndarray]:
    """Line quads from ink projection profiles inside the region box (columns for vertical
    text, rows for horizontal), for detectors that only give blocks."""
    b = region.bbox.clamp(gray.shape[1], gray.shape[0])
    crop = gray[b.y1 : b.y2, b.x1 : b.x2]
    if crop.size == 0:
        return []
    vertical = region.direction == "v"
    ink = glyph_mask(crop, region.font_size, vertical)
    prof = ink.sum(0 if vertical else 1).astype(np.float32)
    runs = _runs(prof > max(1.0, 0.02 * ink.shape[0 if vertical else 1]))
    if not runs:
        return [_rect(b.x1, b.y1, b.x2, b.y2)]
    # glue fragments: thin runs (punctuation, split strokes) join their nearest neighbour
    widths = np.array([e - s for s, e in runs])
    thick = float(np.median(widths[widths >= 0.5 * widths.max()]))
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if s - pe < 0.25 * thick and (e - s < 0.6 * thick or pe - ps < 0.6 * thick):
            merged[-1][1] = e
        else:
            merged.append([s, e])
    # the strong-profile runs stop at the baseline: grow each over its faint ink (descenders,
    # commas, accents) up to 0.35 thickness, never past halfway to its neighbour
    merged = [m for m in merged if m[1] - m[0] >= 0.3 * thick]
    grow = int(0.35 * thick)
    for k, (s, e) in enumerate(merged):
        lo = max(s - grow, (merged[k - 1][1] + s) // 2 if k else 0)
        hi = min(e + grow, (e + merged[k + 1][0] + 1) // 2 if k + 1 < len(merged) else len(prof))
        while s > lo and prof[s - 1] > 0:
            s -= 1
        while e < hi and prof[e] > 0:
            e += 1
        merged[k] = [s, e]
    quads = []
    for s, e in merged:
        span = _main_span(ink[:, s:e] if vertical else ink[s:e].T, thick)
        if span is None:
            continue
        a, z = span
        if vertical:
            quads.append(_rect(b.x1 + s, b.y1 + a, b.x1 + e, b.y1 + z))
        else:
            quads.append(_rect(b.x1 + a, b.y1 + s, b.x1 + z, b.y1 + e))
    return quads


def _main_span(slab: np.ndarray, thick: float) -> tuple[int, int] | None:
    """Extent of the text along one line's ink slab (rows = along the line). Ink stretches
    less than one line thickness apart are one group. From the inkiest group the line grows
    over gaps of up to MAX_GAP thicknesses (word spaces, full-width spaces, ", " in monospaced
    fonts) to the next group that fills at least half the line's thickness like a glyph, taking
    small marks in between along; a lone stray stroke past the text (balloon edge, speed line
    clipped by the box) has no glyph beyond it and is left out."""
    profile = slab.sum(1)
    runs = _runs(profile > 0)
    if not runs:
        return None
    groups = [list(runs[0])]
    for s, e in runs[1:]:
        if s - groups[-1][1] <= thick:
            groups[-1][1] = e
        else:
            groups.append([s, e])
    glyph = []
    for s, e in groups:
        cross = np.flatnonzero(slab[s:e].any(0))
        glyph.append(cross[-1] - cross[0] + 1 >= 0.5 * thick)
    gap = [groups[i + 1][0] - groups[i][1] for i in range(len(groups) - 1)]

    def reach(k: int, step: int) -> int:
        end, i = k, k
        while 0 <= i + step < len(groups) and gap[min(i, i + step)] <= MAX_GAP * thick:
            i += step
            if glyph[i]:
                end = i
        return end

    k = int(np.argmax([profile[s:e].sum() for s, e in groups]))
    return groups[reach(k, -1)][0], groups[reach(k, 1)][1]


def _rect(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


def reading_order(quads: list[np.ndarray], direction: str) -> list[int]:
    """Japanese: columns right to left, rows top to bottom."""
    cs = [order_quad(q).mean(0) for q in quads]
    if direction == "v":
        return sorted(range(len(quads)), key=lambda i: -cs[i][0])
    return sorted(range(len(quads)), key=lambda i: cs[i][1])


def join_lines(texts: list[str]) -> str:
    """Concatenate line texts; a space only where a Latin word or sentence breaks across lines."""
    out = ""
    for t in texts:
        if out and t and word_break(out, t):
            out += " "
        out += t
    return out


def word_break(prev: str, nxt: str) -> bool:
    """Whether Latin text `prev` continued by `nxt` needs a space between them: after a word
    or its closing punctuation ("THERE," + "HOW"), before a word or an opening quote. A trailing
    hyphen glues ("WON-" + "DERFUL"); CJK ("あ!" + "10") never gets spaces."""
    if not prev or not nxt or not (_latin(nxt[0]) or nxt[0] in "([{\"'"):
        return False
    word = prev.rstrip(CLOSERS)
    return len(prev) - len(word) <= 3 and bool(word) and _latin(word[-1])


def _latin(c: str) -> bool:
    return c.isascii() and c.isalnum()


def vertical_forms(text: str) -> str:
    """In a rotated column the vertical two-dot leader (looks like a colon) reads as "："; a
    colon between Latin letters or digits (e.g. "10:30") is a real one and is kept."""
    return "".join(
        "‥"
        if c in "：:"
        and not (0 < i < len(text) - 1 and _latin(text[i - 1]) and _latin(text[i + 1]))
        else c
        for i, c in enumerate(text)
    )


def line_pieces(crop: np.ndarray) -> list[tuple[int, int, bool]]:
    """Column spans (start, end, starts_after_space) cutting a long line crop at blank columns,
    so that no piece exceeds PIECE_WIDTH. Short lines come back whole."""
    w = crop.shape[1]
    if w <= SPLIT_WIDTH:
        return [(0, w, False)]
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.int16)
    bg = np.median(np.concatenate([g[:2], g[-2:]]))  # the margin rows are background
    ink = (np.abs(g - bg) > 48).sum(0)
    # a cut gap reads as a word space when it is clearly wider than the usual letter gap
    gaps = [e - s for s, e in _runs(ink == 0) if 0 < s and e < w]
    space = max(2 * float(np.median(gaps)) if gaps else 0.0, TEXT_HEIGHT / 5)
    pieces, start, gap = [], 0, False
    while w - start > PIECE_WIDTH:
        lo, hi = start + PIECE_WIDTH // 2, start + PIECE_WIDTH
        win = ink[lo:hi]
        runs = _runs(win <= win.min())
        a, b = max(runs, key=lambda r: r[1] - r[0])
        cut = lo + (a + b) // 2
        pieces.append((start, cut, gap))
        start, gap = cut, win.min() == 0 and b - a >= space
    pieces.append((start, w, gap))
    return pieces


def _inside(q: np.ndarray, w: int, h: int) -> float:
    """Fraction of the quad's bounding box that lies on the image."""
    (x0, y0), (x1, y1) = q.min(0), q.max(0)
    ix = max(0.0, min(x1, w) - max(x0, 0))
    iy = max(0.0, min(y1, h) - max(y0, 0))
    return ix * iy / max((x1 - x0) * (y1 - y0), 1e-6)


def as_bgr8(img: np.ndarray) -> np.ndarray:
    """Any grey/BGR/BGRA image, (h,w) or (h,w,c), 8/16-bit or float (0-1 or 0-255) as 8-bit BGR."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        if img.dtype == np.uint16:
            img = img >> 8
        elif img.dtype.kind == "f" and img.size and float(np.nanmax(img)) <= 1.0:
            img = img * 255.0
        img = np.clip(np.nan_to_num(img), 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if img.shape[2] == 4 else img


def default_threads() -> int:
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        cores = os.cpu_count() or 2
    return max(1, cores // 2)


def _rgb(v: np.ndarray) -> RGB:
    r, g, b = np.clip(np.round(v * 255), 0, 255).astype(int).tolist()
    return r, g, b


class Mit48pxOCR:
    """48px OCR.

    Options: beams (1 = greedy; same text here at ~same speed), batch (lines per forward),
    threads (torch intra-op threads while reading; default half the usable cores, because
    a fully subscribed CPU shared with other stages or workers runs 4-15x slower), min_line_conf
    (lines below are dropped),
    margin (quad growth as a fraction of line thickness; 0.1 keeps edge punctuation), device.
    """

    def __init__(
        self,
        model_path: Path,
        alphabet_path: Path,
        *,
        beams: int = 5,
        batch: int = 16,
        threads: int | None = None,
        min_line_conf: float = 0.2,
        margin: float = 0.1,
        device: str = "cpu",
    ) -> None:
        import torch

        from ._mit48px_model import OCR48px

        self.threads = int(threads) if threads else default_threads()
        with open(alphabet_path, encoding="utf-8") as f:
            self.alphabet = [s.rstrip("\n") for s in f]
        self.model = OCR48px(len(self.alphabet))
        sd = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(sd, strict=False)  # inv_freq is a derived, unsaved buffer
        self.model.eval().to(device)
        self.device = device
        self.beams, self.batch, self.min_line_conf = int(beams), int(batch), float(min_line_conf)
        self.margin = float(margin)
        self.text = [
            " " if s == "<SP>" else "" if s.startswith("<") and s.endswith(">") else s
            for s in self.alphabet
        ]

    def lines(self, gray: np.ndarray, region: TextRegion) -> list[np.ndarray]:
        """Line quads of a region in reading order; parts lying off the image are ignored."""
        h, w = gray.shape
        b = region.bbox
        if min(b.x2, w) - max(b.x1, 0) < 2 or min(b.y2, h) - max(b.y1, 0) < 2:
            return []
        quads = [np.asarray(q, np.float32) for q in region.lines if len(q) == 4]
        quads = [q for q in quads if np.ptp(q, 0).min() >= 3 and _inside(q, w, h) >= 0.5]
        quads = quads or split_lines(gray, region)
        return [quads[i] for i in reading_order(quads, region.direction)]

    def read_lines(self, crops: list[np.ndarray]) -> list[tuple[str, float, np.ndarray]]:
        """Recognise 48px-high RGB line crops: (text, confidence, per-char colours (n, 10))."""
        spans = [line_pieces(c) for c in crops]
        pieces = [c[:, a:b] for c, sp in zip(crops, spans) for a, b, _ in sp]
        read = self._read(pieces)
        out, k = [], 0
        for sp in spans:
            part = read[k : k + len(sp)]
            k += len(sp)
            if len(part) == 1:
                out.append(part[0])
                continue
            text = ""
            for (t, _, _), (_, _, space) in zip(part, sp):
                text += " " + t if space and text and t and word_break(text, t) else t
            n = np.array([max(len(t), 1) for t, _, _ in part], np.float32)
            conf = float(np.exp((np.log([max(c, 1e-9) for _, c, _ in part]) * n).sum() / n.sum()))
            out.append((text, conf, np.concatenate([c for _, _, c in part])))
        return out

    def _read(self, crops: list[np.ndarray]) -> list[tuple[str, float, np.ndarray]]:
        import torch

        out: list[Any] = [None] * len(crops)
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1])
        prev = torch.get_num_threads()
        torch.set_num_threads(self.threads)  # scoped, so other torch stages keep their own
        try:
            for s in range(0, len(order), self.batch):
                idx = order[s : s + self.batch]
                widths = [crops[i].shape[1] for i in idx]
                # black right padding, 7px at least, as the reference (white/edge reads worse)
                arr = np.zeros((len(idx), TEXT_HEIGHT, max(widths) + 7, 3), np.uint8)
                for j, i in enumerate(idx):
                    arr[j, :, : widths[j]] = crops[i]
                x = torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2).float() / 127.5 - 1
                for i, d in zip(idx, self.model.decode(x, widths, beams=self.beams)):
                    text = "".join(self.text[t] for t in d.tokens).strip()
                    out[i] = (text, float(np.exp(d.logprob)), d.colors.cpu().numpy())
        finally:
            torch.set_num_threads(prev)
        return out

    def recognize(self, image_bgr: np.ndarray, regions: list[TextRegion]) -> list[OcrResult]:
        image_bgr = as_bgr8(image_bgr)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        owner, dirs, crops = [], [], []
        for ri, r in enumerate(regions):
            for q in self.lines(gray, r):
                d = line_direction(order_quad(q), r.direction)
                owner.append(ri)
                dirs.append(d)
                crops.append(warp_line(rgb, q, d, self.margin))
        read = self.read_lines(crops) if crops else []
        per: list[list[tuple[str, float, np.ndarray]]] = [[] for _ in regions]
        for ri, d, (text, conf, cols) in zip(owner, dirs, read):
            per[ri].append((vertical_forms(text) if d == "v" else text, conf, cols))
        return [self._merge(lines) for lines in per]

    def _merge(self, lines: list[tuple[str, float, np.ndarray]]) -> OcrResult:
        kept = [ln for ln in lines if ln[0] and ln[1] >= self.min_line_conf]
        if not kept:
            return OcrResult(text="", confidence=max((ln[1] for ln in lines), default=0.0))
        n = np.array([max(len(t), 1) for t, _, _ in kept], np.float32)
        conf = float(np.exp((np.log([c for _, c, _ in kept]) * n).sum() / n.sum()))
        cols = np.concatenate([c for _, _, c in kept])
        has_fg = cols[:, 7] > cols[:, 6]
        has_bg = cols[:, 9] > cols[:, 8]
        fg = _rgb(cols[has_fg, 0:3].mean(0)) if has_fg.any() else None
        # outline colour only when most glyphs are predicted to have one
        bg = _rgb(cols[has_bg, 3:6].mean(0)) if has_bg.sum() * 2 > len(cols) else None
        text = join_lines([t for t, _, _ in kept])
        return OcrResult(text=text, confidence=conf, fg_color=fg, bg_color=bg)


@OCRS.register(
    "48px",
    description=(
        "manga-image-translator 48px autoregressive OCR, per text line, beam search; "
        "predicts glyph/outline colours (torch, ~1-2s/page CPU)"
    ),
    requires=("torch",),
    weights=("ocr_48px", "ocr_48px_alphabet"),
)
def _build(settings, **opts):
    opts.setdefault("device", getattr(settings, "device", "cpu"))
    return Mit48pxOCR(ensure("ocr_48px", settings), ensure("ocr_48px_alphabet", settings), **opts)
