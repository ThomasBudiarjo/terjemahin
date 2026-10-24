"""LaMa inpainting (ONNX, fixed 512x512) run per text cluster instead of on a downscaled page.

manga-image-translator shrinks the whole page to `inpainting_size` and pastes the masked pixels
back, which blurs every repaired area on a large page. Here the hole mask is split into clusters
(nearby components merged), each cluster gets a square window with real context around it
(`margin` x its size, never smaller than the model input so small bubbles run at native
resolution), windows that would need more than `max_down` downscaling are cut into cells, and
only the hole pixels (plus a short feathered ring) are pasted back. Holes surrounded by a flat
colour (text inside a plain white balloon) are filled with that colour directly: exact, instant,
and the model is only loaded when some hole actually sits on artwork.

Both graphs take RGB in 0..1 and a 0/1 mask (1 = hole) and zero the hole themselves; koharu's
lama-manga returns the raw sigmoid image in 0..1, Carve's big-lama returns 0..255 already
composited with the input.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..registry import INPAINTERS
from ..weights import ensure

_SESSIONS: dict[tuple[str, int, str], Any] = {}
_LOCK = threading.Lock()


def _session(path: Path, threads: int, device: str) -> Any:
    """One onnxruntime session per (file, threads, device), shared by every instance."""
    import onnxruntime as ort

    key = (str(path), threads, device)
    with _LOCK:
        if key not in _SESSIONS:
            so = ort.SessionOptions()
            if threads > 0:
                so.intra_op_num_threads = threads
            providers = ["CPUExecutionProvider"]
            if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            _SESSIONS[key] = ort.InferenceSession(str(path), so, providers=providers)
        return _SESSIONS[key]


@dataclass(frozen=True)
class Tile:
    """`win` is the page window fed to the model; only hole pixels inside `own` are pasted."""

    win: tuple[int, int, int, int]
    own: tuple[int, int, int, int]


FLAT_INLIERS = 0.8  # share of the ring that must sit within 4 x tol of its median colour


def flat_fill(
    img: np.ndarray, hole: np.ndarray, tol: float, ring: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Fill hole components whose surrounding ring is one flat colour; return (image, rest).

    Flat means: the ring is uniform (std <= tol, 2-98% range <= 4 tol), or, on a noisy scan,
    nearly all of it (FLAT_INLIERS) lies within 4 tol of its median and those pixels are
    uniform (std <= 2 tol): scan grain, JPEG noise and the stretch of balloon outline a ring
    brushes past (up to 1 - FLAT_INLIERS of it). Such holes went to LaMa before, which on B/W scans rebuilt dotted
    ghosts of furigana and glyph fringes in plain white balloons (Arisa_010)."""
    out = img.copy()
    rest = hole.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
    H, W = hole.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        x1, y1 = max(0, x - ring), max(0, y - ring)
        x2, y2 = min(W, x + w + ring), min(H, y + h + ring)
        comp = labels[y1:y2, x1:x2] == i
        band = cv2.dilate(comp.astype(np.uint8), k).astype(bool) & ~hole[y1:y2, x1:x2]
        px = img[y1:y2, x1:x2][band].astype(np.float32)
        if len(px) < 8:
            continue
        lo, hi = np.percentile(px, (2, 98), axis=0)
        fill = None
        if px.std(axis=0).max() <= tol and (hi - lo).max() <= 4 * tol:
            fill = np.median(px, axis=0)
        else:
            med = np.median(px, axis=0)
            inl = np.abs(px - med).max(axis=1) <= 4 * tol
            if inl.mean() >= FLAT_INLIERS and px[inl].std(axis=0).max() <= 2 * tol:
                fill = np.median(px[inl], axis=0)
        if fill is None:
            continue
        out[y1:y2, x1:x2][comp] = fill.round().astype(np.uint8)
        rest[y1:y2, x1:x2][comp] = False
    return out, rest


def plan_tiles(
    hole: np.ndarray, size: int = 512, margin: float = 2.0, merge: int = 48, max_down: float = 2.0
) -> list[Tile]:
    """Cluster the hole mask and give every cluster (or cell of a huge one) a context window."""
    H, W = hole.shape
    m = hole.astype(np.uint8)
    if merge > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (merge, merge)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        ys, xs = np.nonzero(hole[y : y + h, x : x + w] & (labels[y : y + h, x : x + w] == i))
        if len(xs):
            boxes.append((x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1))
    cell = max(16, int(size * max_down / max(margin, 1.0)))
    tiles: list[Tile] = []
    for gx1, gy1, gx2, gy2 in _pack(boxes, size / max(margin, 1.0)):
        for cy in range(gy1, gy2, cell):
            for cx in range(gx1, gx2, cell):
                own = (cx, cy, min(cx + cell, gx2), min(cy + cell, gy2))
                if hole[own[1] : own[3], own[0] : own[2]].any():
                    tiles.append(Tile(_window(own, W, H, size, margin), own))
    return tiles


def _pack(boxes: list[tuple[int, int, int, int]], fit: float) -> list[tuple[int, int, int, int]]:
    """Greedily union clusters whose joint box still fits one native-resolution window:
    one model call instead of several, with the same context per cluster."""
    out: list[tuple[int, int, int, int]] = []
    for b in sorted(boxes, key=lambda b: (b[1], b[0])):
        for k, a in enumerate(out):
            u = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
            if max(u[2] - u[0], u[3] - u[1]) <= fit:
                out[k] = u
                break
        else:
            out.append(b)
    return out


def _window(own: tuple[int, int, int, int], W: int, H: int, size: int, margin: float):
    x1, y1, x2, y2 = own
    side = max(size, math.ceil(max(x2 - x1, y2 - y1) * max(margin, 1.0)))
    ww, wh = min(side, W), min(side, H)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    wx = min(max(0, cx - ww // 2), W - ww)
    wy = min(max(0, cy - wh // 2), H - wh)
    return wx, wy, wx + ww, wy + wh


class LamaInpainter:
    def __init__(
        self,
        model: Path,
        *,
        out_scale: float,
        size: int = 512,
        margin: float = 2.0,
        merge: int = 48,
        max_down: float = 2.0,
        batch: int = 1,
        feather: int = 2,
        flat_tol: float = 3.0,
        dilate: int = 2,
        threads: int = 0,
        device: str = "cpu",
    ) -> None:
        self.model, self.out_scale = model, out_scale
        self.size, self.margin, self.merge, self.max_down = size, margin, merge, max_down
        self.batch, self.feather, self.flat_tol = max(1, batch), feather, flat_tol
        self.dilate = int(dilate)
        self.threads, self.device = threads, device
        self.stats: dict[str, int] = {}

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hole = (mask.max(axis=2) if mask.ndim == 3 else mask) > 0
        out = image_bgr.copy()
        if not hole.any():
            return out
        if self.dilate > 0:
            k = np.ones((2 * self.dilate + 1,) * 2, np.uint8)
            hole = cv2.dilate(hole.astype(np.uint8), k).astype(bool)
        if self.flat_tol > 0:
            out, hole = flat_fill(out, hole, self.flat_tol)
        tiles = plan_tiles(hole, self.size, self.margin, self.merge, self.max_down)
        self.stats = {"tiles": len(tiles), "model_px": int(hole.sum())}
        for b in range(0, len(tiles), self.batch):
            chunk = tiles[b : b + self.batch]
            preps = [self._prep(out, hole, t) for t in chunk]
            res = self._run(np.stack([p[0] for p in preps]), np.stack([p[1] for p in preps]))
            for t, (_, _, rw, rh), r in zip(chunk, preps, res):
                self._paste(out, hole, t, r[:rh, :rw])
        return out

    def _prep(self, img: np.ndarray, hole: np.ndarray, t: Tile):
        x1, y1, x2, y2 = t.win
        w, h, S = x2 - x1, y2 - y1, self.size
        s = min(1.0, S / max(w, h))
        rw, rh = max(1, round(w * s)), max(1, round(h * s))
        crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        m = hole[y1:y2, x1:x2].astype(np.uint8) * 255
        if s < 1.0:
            crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA)
            # one extra model pixel so upsampled hole edges interpolate only inpainted pixels
            m = cv2.dilate(cv2.resize(m, (rw, rh), interpolation=cv2.INTER_AREA), np.ones((3, 3)))
        pad = ((0, S - rh), (0, S - rw))
        crop = np.pad(crop, (*pad, (0, 0)), mode="reflect" if min(rw, rh) > 1 else "edge")
        m = np.pad(m, pad, mode="reflect" if min(rw, rh) > 1 else "edge")
        x = crop.astype(np.float32).transpose(2, 0, 1) / 255.0
        return x, (m > 0).astype(np.float32)[None], rw, rh

    def _run(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        sess = _session(self.model, self.threads, self.device)
        y = sess.run(None, {"image": x, "mask": m})[0]
        y = y.transpose(0, 2, 3, 1) * (255.0 / self.out_scale)
        return np.clip(y, 0, 255).round().astype(np.uint8)[..., ::-1]

    def _paste(self, out: np.ndarray, hole: np.ndarray, t: Tile, res: np.ndarray) -> None:
        x1, y1, x2, y2 = t.win
        w, h = x2 - x1, y2 - y1
        if res.shape[:2] != (h, w):
            res = cv2.resize(res, (w, h), interpolation=cv2.INTER_LINEAR)
        own = np.zeros((h, w), np.float32)
        ox1, oy1, ox2, oy2 = t.own
        own[oy1 - y1 : oy2 - y1, ox1 - x1 : ox2 - x1] = hole[oy1:oy2, ox1:ox2]
        a = own
        if self.feather > 0:
            k = 2 * self.feather + 1
            ring = cv2.GaussianBlur(cv2.dilate(own, np.ones((k, k), np.uint8)), (k, k), 0)
            a = np.maximum(own, ring)
        a = a[..., None]
        dst = out[y1:y2, x1:x2]
        dst[:] = (res * a + dst * (1.0 - a)).round().astype(np.uint8)


# name: (kind, lowest, highest). max_down below 1 shrank the tile cell to 16 px: page01 then
# planned 1086 model calls (~3 s each) instead of 13.
_RANGES: dict[str, tuple[type, float, float | None]] = {
    "size": (int, 64, 4096), "merge": (int, 0, 1000), "batch": (int, 1, 64),
    "feather": (int, 0, 64), "dilate": (int, 0, 64), "threads": (int, 0, 1024),
    "margin": (float, 1.0, 16.0), "max_down": (float, 1.0, 16.0), "flat_tol": (float, 0.0, 255.0),
}


def _opts(settings, name: str = "lama", **o: Any) -> dict[str, Any]:
    from ..opts import as_float, as_int, warn_unknown

    warn_unknown(f"inpainter '{name}'", {k: v for k, v in o.items() if k not in _RANGES and k != "weights"})
    out: dict[str, Any] = {"device": settings.device}
    for k, v in o.items():
        if k in _RANGES:
            kind, lo, hi = _RANGES[k]
            out[k] = (as_int if kind is int else as_float)(v, f"inpainter {k}", lo, hi)
    return out


def _check_size(path: Path, opts: dict[str, Any], given: bool) -> None:
    """The ONNX graphs have a static input size (512 for both shipped ones). A different
    `size` only failed inside onnxruntime once a hole reached the model, i.e. on some pages
    only; the graph is opened here when the size or the graph is not the default."""
    size = opts.get("size", 512)
    if size == 512 and not given:
        return
    sess = _session(path, opts.get("threads", 0), opts.get("device", "cpu"))
    shape = next((i.shape for i in sess.get_inputs() if i.name == "image"), None)
    fixed = [d for d in (shape or [])[2:] if isinstance(d, int)]
    if fixed and any(d != size for d in fixed):
        raise ValueError(f"inpainter size must be {fixed[0]} for {path.name} (its input is fixed), not {size}")


def _graph(settings, key: str, weights: Any = None) -> Path:
    """`weights=<onnx>` swaps in another graph with the same interface (e.g. a trained AOT export)."""
    if not weights:
        return ensure(key, settings)
    path = Path(weights).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"inpainter weights not found: {path}")
    return path


@INPAINTERS.register(
    "lama_manga",
    description="LaMa fine-tuned on manga (ONNX 512), per text cluster at native resolution; "
    "flat-background holes filled exactly without the model (~3s per 512 tile CPU).",
    requires=("onnxruntime",),
    weights=("lama_manga",),
)
def _build_manga(settings, **opts: Any) -> LamaInpainter:
    path = _graph(settings, "lama_manga", opts.get("weights"))
    o = _opts(settings, "lama_manga", **opts)
    _check_size(path, o, bool(opts.get("weights")))
    return LamaInpainter(path, out_scale=1.0, **o)


@INPAINTERS.register(
    "lama",
    description="big-lama Places2 (Carve ONNX 512), same tiling as lama_manga; better on "
    "photographic backgrounds, worse on line art and screentone.",
    requires=("onnxruntime",),
    weights=("lama",),
)
def _build_big(settings, **opts: Any) -> LamaInpainter:
    path = _graph(settings, "lama", opts.get("weights"))
    o = _opts(settings, "lama", **opts)
    _check_size(path, o, bool(opts.get("weights")))
    return LamaInpainter(path, out_scale=255.0, **o)
