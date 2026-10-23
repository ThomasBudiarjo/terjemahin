"""Fine-tune (or train from scratch) a compact AOT inpainting generator on text-removal triplets.

The network is manga-image-translator's AOTGenerator (gated, scaled-weight-standardised convs +
10 AOT dilated blocks, ~5.8M params, 32 base channels), rewritten here with the same parameter
names so the released `inpainting.ckpt` loads as the starting point (`--init aot`, fetched from
the MIT-licensed Hugging Face mirror ogkalu/aot-inpainting and checked against the upstream
SHA-256). It is small enough to fine-tune on CPU and fast at inference, unlike big LaMa.

Data: training/synth.py writes `<out>/inpaint/{clean,mask,texted}/NNNNNN.png` (same stem in all
three; mask is uint8 0/255 with 255 = text pixel). Any folder with that layout works, so real
pairs (a cleaned scan + its raw page + a text mask) can be mixed in with more `--data` folders.
`--clean-only DIR` adds text-free pages with random text-like masks.

Training crops are taken around mask components and biased towards holes that sit on artwork
(screentone, lines): the worker's `lama*` inpainters fill holes inside flat balloons exactly
without any model, so those are not what the network needs to learn. Crops are also randomly
downscaled up to 2x, the same range lama.py's tiling feeds the model.

Loss (no perceptual net, cheap on CPU): L1 in the hole (x6), L1 outside (x1), and an L1 on Sobel
gradients in the hole (keeps line art and tone edges sharp instead of smeared).

    python training/train_inpaint.py --data data/synth --steps 2000 --size 256 --batch 8 \
        --out training/runs/inpaint --export weights/aot-ft.onnx
    terjemahin-worker translate page.png -i lama_manga --opt inpainter.weights=weights/aot-ft.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

AOT_REPO = "ogkalu/aot-inpainting"
AOT_FILE = "inpainting.ckpt"
AOT_SHA256 = "878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f"
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")

# --- model -----------------------------------------------------------------------------------


def _ws(weight: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """Scaled weight standardisation (NF-Nets): per-filter zero mean, unit fan-in variance."""
    fan_in = math.prod(weight.shape[1:])
    var, mean = torch.var_mean(weight, dim=(1, 2, 3), keepdim=True)
    scale = torch.rsqrt(torch.clamp(var * fan_in, min=eps)) * gain.view_as(var)
    return weight * scale - mean * scale


class WSConv2d(nn.Conv2d):
    def __init__(self, *a, eps: float = 1e-4, **k) -> None:
        super().__init__(*a, **k)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.eps = eps

    def standardized(self) -> torch.Tensor:
        return _ws(self.weight, self.gain, self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.standardized(), self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class WSConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *a, eps: float = 1e-4, **k) -> None:
        super().__init__(*a, **k)
        self.gain = nn.Parameter(torch.ones(self.in_channels, 1, 1, 1))
        self.eps = eps

    def standardized(self) -> torch.Tensor:
        return _ws(self.weight, self.gain, self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose2d(x, self.standardized(), self.bias, self.stride, self.padding,
                                  self.output_padding, self.groups, self.dilation)


class Gated(nn.Module):
    """signal * sigmoid(gate) * 1.8 (the constant keeps activations unit-scale, as upstream)."""

    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, up: bool = False) -> None:
        super().__init__()
        if up:
            self.padding = nn.Identity()
            self.conv = WSConvTranspose2d(cin, cout, k, stride, padding=(k - 1) // 2)
            self.conv_gate = WSConvTranspose2d(cin, cout, k, stride, padding=(k - 1) // 2)
        else:
            self.padding = nn.ReflectionPad2d((k - 1) // 2)
            self.conv = WSConv2d(cin, cout, k, stride)
            self.conv_gate = WSConv2d(cin, cout, k, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.padding(x)
        return self.conv(x) * torch.sigmoid(self.conv_gate(x)) * 1.8


class ReluNF(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) * 1.7139588594436646


class AOTBlock(nn.Module):
    """Aggregated contextual transformations: 4 dilated branches fused, blended by a gate."""

    def __init__(self, dim: int, rates: tuple[int, ...] = (2, 4, 8, 16)) -> None:
        super().__init__()
        self.rates = rates
        for i, r in enumerate(rates):
            setattr(self, f"block{i:02d}", nn.Sequential(
                nn.ReflectionPad2d(r), nn.Conv2d(dim, dim // 4, 3, dilation=r), nn.ReLU(True)))
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fuse(torch.cat([getattr(self, f"block{i:02d}")(x)
                                   for i in range(len(self.rates))], 1))
        g = self.gate(x)
        g = 2 * (g - g.mean((2, 3), keepdim=True)) / (g.std((2, 3), keepdim=True) + 1e-9) - 1
        m = torch.sigmoid(5 * g)
        return x * (1 - m) + out * m


class AOTGenerator(nn.Module):
    """(img in -1..1 with hole zeroed, mask 0/1) -> img in -1..1; H, W multiples of 4, >= 68."""

    def __init__(self, ch: int = 32, blocks: int = 10) -> None:
        super().__init__()
        self.config = {"ch": ch, "blocks": blocks}
        self.head = nn.Sequential(
            Gated(4, ch, 3), ReluNF(), Gated(ch, ch * 2, 4, 2), ReluNF(),
            Gated(ch * 2, ch * 4, 4, 2))
        self.body_conv = nn.Sequential(*[AOTBlock(ch * 4) for _ in range(blocks)])
        self.tail = nn.Sequential(
            Gated(ch * 4, ch * 4, 3), ReluNF(), Gated(ch * 4, ch * 4, 3), ReluNF(),
            Gated(ch * 4, ch * 2, 4, 2, up=True), ReluNF(),
            Gated(ch * 2, ch, 4, 2, up=True), ReluNF(), Gated(ch, 3, 3))

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.tail(self.body_conv(self.head(torch.cat([mask, img], 1))))
        return x if self.training else x.clamp(-1, 1)


def fetch_aot(weights_dir: Path = ROOT / "weights") -> Path:
    """The released manga-image-translator AOT checkpoint (github release is the upstream)."""
    dst = weights_dir / "aot-inpainting.ckpt"
    if not dst.exists():
        from huggingface_hub import hf_hub_download

        src = Path(hf_hub_download(AOT_REPO, AOT_FILE))
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        if digest != AOT_SHA256:
            raise RuntimeError(f"{AOT_REPO}/{AOT_FILE} sha256 {digest} != upstream {AOT_SHA256}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    return dst


def load_generator(init: str | Path | None, ch: int = 32, blocks: int = 10) -> AOTGenerator:
    """`init`: 'aot' (released weights), a checkpoint saved by this script, or None (scratch)."""
    if init in (None, "", "none"):
        return AOTGenerator(ch, blocks)
    path = fetch_aot() if init == "aot" else Path(init)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    cfg = sd.get("config", {"ch": 32, "blocks": 10}) if isinstance(sd, dict) else {}
    net = AOTGenerator(cfg.get("ch", ch), cfg.get("blocks", blocks))
    net.load_state_dict(sd.get("model", sd))
    return net


# --- data ------------------------------------------------------------------------------------


def find_triplets(root: Path) -> list[tuple[Path, Path, Path]]:
    """(clean, mask, texted) paths sharing a stem; accepts the synth root or its inpaint/ dir."""
    for base in (root / "inpaint", root):
        if (base / "clean").is_dir() and (base / "mask").is_dir():
            break
    else:
        raise FileNotFoundError(f"{root}: expected clean/, mask/ (and texted/) subfolders")
    out = []
    for c in sorted(p for p in (base / "clean").iterdir() if p.suffix.lower() in IMG_EXT):
        m = next((base / "mask" / f"{c.stem}{e}" for e in IMG_EXT
                  if (base / "mask" / f"{c.stem}{e}").exists()), None)
        t = next((base / "texted" / f"{c.stem}{e}" for e in IMG_EXT
                  if (base / "texted" / f"{c.stem}{e}").exists()), c)
        if m is not None:
            out.append((c, m, t))
    return out


def random_text_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Columns/rows of glyph-sized blobs and strokes: the shape of a real text mask."""
    m = np.zeros((h, w), np.uint8)
    for _ in range(rng.randint(1, 4)):
        g = rng.randint(max(6, min(h, w) // 40), max(8, min(h, w) // 12))
        vertical = rng.random() < 0.7
        n_lines, n_chars = rng.randint(1, 5), rng.randint(2, 10)
        x0, y0 = rng.randint(0, max(1, w - g)), rng.randint(0, max(1, h - g))
        for li in range(n_lines):
            for ci in range(n_chars):
                dx, dy = (-li * g * 1.3, ci * g * 1.1) if vertical else (ci * g * 1.1, li * g * 1.3)
                cx, cy = int(x0 + dx), int(y0 + dy)
                for _ in range(rng.randint(2, 5)):
                    p1 = (cx + rng.randint(0, g), cy + rng.randint(0, g))
                    p2 = (cx + rng.randint(0, g), cy + rng.randint(0, g))
                    cv2.line(m, p1, p2, 255, max(1, g // rng.randint(5, 9)))
    return m


class Pages:
    """Lazy page loader with a small LRU cache and per-page hole components."""

    def __init__(self, triplets, clean_only, flat_tol: float, cache: int = 48) -> None:
        self.items = [("t", *t) for t in triplets] + [("c", p, None, p) for p in clean_only]
        self.flat_tol, self.cache, self._lru = flat_tol, cache, OrderedDict()

    def __len__(self) -> int:
        return len(self.items)

    def get(self, i: int, rng: random.Random):
        if i in self._lru:
            self._lru.move_to_end(i)
            return self._lru[i]
        kind, clean_p, mask_p, texted_p = self.items[i]
        clean = cv2.imread(str(clean_p), cv2.IMREAD_COLOR)
        if kind == "t":
            mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            texted = cv2.imread(str(texted_p), cv2.IMREAD_COLOR)
        else:
            mask = random_text_mask(*clean.shape[:2], rng)
            texted = clean
        hole = mask > 127
        boxes, hard = _components(texted, hole, self.flat_tol)
        entry = (clean, texted, hole, boxes, hard)
        self._lru[i] = entry
        if len(self._lru) > self.cache:
            self._lru.popitem(last=False)
        return entry


def _components(img: np.ndarray, hole: np.ndarray, flat_tol: float):
    """Bounding boxes of text clusters, and the subset that is not on a flat colour."""
    from terjemahin_worker.inpaint.lama import flat_fill

    m = cv2.dilate(hole.astype(np.uint8), np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = [tuple(int(v) for v in stats[k, :4]) for k in range(1, n)]
    _, rest = flat_fill(img, hole, flat_tol) if flat_tol > 0 else (img, hole)
    hard = [b for b in boxes if rest[b[1] : b[1] + b[3], b[0] : b[0] + b[2]].any()]
    return boxes, hard


def sample_crop(pages: Pages, rng: random.Random, size: int, hard_ratio: float,
                max_down: float, max_dilate: int):
    """One (texted, clean, mask) crop of size x size, float32 HWC RGB 0..1 / HW 0..1."""
    clean, texted, hole, boxes, hard = pages.get(rng.randrange(len(pages)), rng)
    H, W = hole.shape
    s = rng.uniform(1.0, max_down)
    win = min(int(size * s), H, W)
    pool = hard if hard and rng.random() < hard_ratio else boxes
    if pool:
        x, y, w, h = rng.choice(pool)
        cx = x + w / 2 + rng.uniform(-0.3, 0.3) * win
        cy = y + h / 2 + rng.uniform(-0.3, 0.3) * win
    else:
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
    x1 = int(min(max(0, cx - win / 2), W - win))
    y1 = int(min(max(0, cy - win / 2), H - win))
    sl = (slice(y1, y1 + win), slice(x1, x1 + win))
    c, t, m = clean[sl], texted[sl], hole[sl].astype(np.uint8) * 255
    if max_dilate > 0:
        d = rng.randint(0, max_dilate)
        if d:
            m = cv2.dilate(m, np.ones((2 * d + 1,) * 2, np.uint8))
    if win != size:
        interp = cv2.INTER_AREA if win > size else cv2.INTER_LINEAR
        c, t = cv2.resize(c, (size, size), interpolation=interp), cv2.resize(t, (size, size), interpolation=interp)
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        c, t, m = c[:, ::-1], t[:, ::-1], m[:, ::-1]
    rgb = lambda a: a[..., ::-1].astype(np.float32) / 255.0
    return rgb(t), rgb(c), (m > 0).astype(np.float32)


def batch_tensors(crops) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.from_numpy(np.stack([c[0] for c in crops])).permute(0, 3, 1, 2)
    c = torch.from_numpy(np.stack([c[1] for c in crops])).permute(0, 3, 1, 2)
    m = torch.from_numpy(np.stack([c[2] for c in crops]))[:, None]
    return t.contiguous(), c.contiguous(), m.contiguous()


# --- training --------------------------------------------------------------------------------

_SOBEL = torch.tensor([[[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]]) / 4


def _grad(x: torch.Tensor) -> torch.Tensor:
    g = x.mean(1, keepdim=True)
    k = _SOBEL.to(x)
    return torch.cat([F.conv2d(g, k[None], padding=1), F.conv2d(g, k.transpose(1, 2)[None], padding=1)], 1)


def run(net: AOTGenerator, texted01: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Model call in the worker's convention: RGB 0..1 in, RGB 0..1 out (hole only differs)."""
    x = (texted01 * 2 - 1) * (1 - mask)
    y = (net(x, mask).clamp(-1, 1) + 1) / 2
    return texted01 * (1 - mask) + y * mask


def loss_fn(net, texted, clean, mask, hole_w: float, valid_w: float, edge_w: float):
    y = net((texted * 2 - 1) * (1 - mask), mask)
    tgt = clean * 2 - 1
    area = mask.sum().clamp(min=1.0) * 3
    l_hole = ((y - tgt).abs() * mask).sum() / area
    l_valid = ((y - tgt).abs() * (1 - mask)).mean()
    l_edge = ((_grad(y) - _grad(tgt)).abs() * mask).sum() / area if edge_w else y.new_zeros(())
    total = hole_w * l_hole + valid_w * l_valid + edge_w * l_edge
    return total, {"hole": round(l_hole.item(), 4), "valid": round(l_valid.item(), 4),
                   "edge": round(l_edge.item(), 4)}


@torch.no_grad()
def evaluate(net: AOTGenerator, val: list) -> dict[str, float]:
    """Hole-only PSNR / MAE (0..255) of the composited output, and the cv2 Telea baseline."""
    net.eval()
    se = ae = n = se_telea = 0.0
    for t, c, m in val:
        tt, _, mm = batch_tensors([(t, c, m)])
        out = run(net, tt, mm)[0].permute(1, 2, 0).numpy()
        k = m > 0
        if not k.any():
            continue
        d = (out - c)[k] * 255
        se, ae, n = se + float((d**2).sum()), ae + float(np.abs(d).sum()), n + d.size
        tel = cv2.inpaint((t * 255).astype(np.uint8), k.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        se_telea += float((((tel.astype(np.float32) / 255 - c)[k] * 255) ** 2).sum())
    net.train()
    psnr = lambda s: 10 * math.log10(255**2 / max(s / max(n, 1), 1e-8))
    return {"psnr_hole": psnr(se), "mae_hole": ae / max(n, 1), "psnr_hole_telea": psnr(se_telea)}


def split(items: list, val_ratio: float) -> tuple[list, list]:
    if len(items) < 2 or val_ratio <= 0:
        return items, items[:0]
    k = max(1, round(len(items) * val_ratio))
    step = len(items) / k
    vi = {int(i * step) for i in range(k)}
    return [x for i, x in enumerate(items) if i not in vi], [x for i, x in enumerate(items) if i in vi]


def save(net: AOTGenerator, path: Path, step: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "config": net.config, "step": step, "metrics": metrics}, path)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, nargs="+", default=[],
                    help="folders with clean/ mask/ texted/ (or a synth.py --out root)")
    ap.add_argument("--clean-only", type=Path, nargs="*", default=[],
                    help="folders of text-free pages; random text-like masks are drawn")
    ap.add_argument("--init", default="aot", help="'aot' (released weights), 'none', or a .pt")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--size", type=int, default=256, help="crop side (multiple of 4)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--hole-weight", type=float, default=6.0)
    ap.add_argument("--valid-weight", type=float, default=1.0)
    ap.add_argument("--edge-weight", type=float, default=1.0)
    ap.add_argument("--hard-ratio", type=float, default=0.8,
                    help="share of crops centred on holes that are not on a flat colour")
    ap.add_argument("--flat-tol", type=float, default=3.0, help="same as lama.py flat_tol")
    ap.add_argument("--max-down", type=float, default=2.0)
    ap.add_argument("--max-dilate", type=int, default=3)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--val-crops", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("training/runs/inpaint"))
    ap.add_argument("--export", type=Path, default=None, help="also write best weights as ONNX")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.size % 4 or a.size < 68:
        ap.error("--size must be a multiple of 4 and >= 68 (the 16-dilated blocks run at 1/4)")

    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    triplets = [t for d in a.data for t in find_triplets(d)]
    clean_only = [p for d in a.clean_only for p in sorted(d.rglob("*")) if p.suffix.lower() in IMG_EXT]
    tr_t, va_t = split(triplets, a.val_ratio)
    tr_c, va_c = split(clean_only, a.val_ratio)
    train = Pages(tr_t, tr_c, a.flat_tol)
    if not len(train):
        ap.error("no training pages: pass --data and/or --clean-only")
    val_pages = Pages(va_t, va_c, a.flat_tol) if va_t or va_c else train
    vr = random.Random(a.seed + 1)
    val = [sample_crop(val_pages, vr, a.size, a.hard_ratio, a.max_down, 0) for _ in range(a.val_crops)]

    net = load_generator(a.init, a.ch, a.blocks).to(a.device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.5, 0.9), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup))
                                              * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, a.steps)))))
    n_params = sum(p.numel() for p in net.parameters())
    print(f"pages train={len(train)} val={len(val_pages) if val_pages is not train else 0} "
          f"params={n_params / 1e6:.2f}M init={a.init}", flush=True)

    history = [{"step": 0, **evaluate(net.cpu(), val)}]
    net.to(a.device)
    print(json.dumps(history[-1]), flush=True)
    best = history[0]["psnr_hole"]
    save(net, a.out / "best.pt", 0, history[0])
    rng, t0 = random.Random(a.seed), time.time()
    for step in range(1, a.steps + 1):
        t, c, m = (x.to(a.device) for x in batch_tensors(
            [sample_crop(train, rng, a.size, a.hard_ratio, a.max_down, a.max_dilate)
             for _ in range(a.batch)]))
        loss, parts = loss_fn(net, t, c, m, a.hole_weight, a.valid_weight, a.edge_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 10 == 0 or step == a.steps:
            print(f"step {step} loss {loss.item():.4f} {parts} {(time.time() - t0) / step:.2f}s/it",
                  flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            history.append({"step": step, **evaluate(net, val)})
            print(json.dumps(history[-1]), flush=True)
            if history[-1]["psnr_hole"] > best:
                best = history[-1]["psnr_hole"]
                save(net, a.out / "best.pt", step, history[-1])
    save(net, a.out / "last.pt", a.steps, history[-1])
    result = {"history": history, "best": str(a.out / "best.pt"), "best_psnr_hole": best}
    (a.out / "metrics.json").write_text(json.dumps(result, indent=2))
    if a.export:
        from export_inpaint_onnx import export  # same folder

        result["onnx"] = str(export(a.out / "best.pt", a.export))
    return result


if __name__ == "__main__":
    main()
