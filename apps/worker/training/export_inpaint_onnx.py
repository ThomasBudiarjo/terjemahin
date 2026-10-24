"""Export an AOT inpainting checkpoint to ONNX with the worker's LaMa interface.

Graph contract (same as koharu-lama-manga.onnx, so lama.py runs it unchanged):
    image [B,3,H,W] float32 RGB 0..1   mask [B,1,H,W] float32, 1 = hole   ->   output [B,3,H,W] 0..1
H and W are dynamic (padded to a multiple of 4 inside the graph, at least 68; the worker feeds 512). The graph zeroes the hole and maps to
the network's -1..1 range itself, and returns the input outside the hole.

Weight standardisation is folded into plain Conv/ConvTranspose weights before tracing: the graph
is smaller and onnxruntime does not recompute mean/var of every filter on each call.

    python training/export_inpaint_onnx.py training/runs/inpaint/best.pt weights/aot-ft.onnx
    python training/export_inpaint_onnx.py aot weights/aot.onnx          # released weights as-is
    terjemahin-worker translate page.png -i lama_manga --opt inpainter.weights=weights/aot-ft.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_inpaint import (
    AOTGenerator,
    WSConv2d,
    WSConvTranspose2d,
    load_generator,
)


def fold_weight_standardization(net: nn.Module) -> nn.Module:
    """Replace every WS conv by an ordinary conv holding the standardised weight (in place)."""
    for name, mod in list(net.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, WSConvTranspose2d):
                new = nn.ConvTranspose2d(child.in_channels, child.out_channels, child.kernel_size,
                                         child.stride, child.padding, child.output_padding,
                                         child.groups, child.bias is not None, child.dilation)
            elif isinstance(child, WSConv2d):
                new = nn.Conv2d(child.in_channels, child.out_channels, child.kernel_size,
                                child.stride, child.padding, child.dilation, child.groups,
                                child.bias is not None)
            else:
                continue
            with torch.no_grad():
                new.weight.copy_(child.standardized())
                if child.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(mod, child_name, new)
    return net


class LamaInterface(nn.Module):
    def __init__(self, gen: AOTGenerator) -> None:
        super().__init__()
        self.gen = gen

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pads H, W up to a multiple of 4 (traced as shape ops, so it stays dynamic)."""
        h, w = image.shape[2], image.shape[3]
        ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
        x = F.pad(image, (0, pw, 0, ph), mode="replicate")
        m = (F.pad(mask, (0, pw, 0, ph), mode="replicate") > 0.5).to(image.dtype)
        y = (self.gen((x * 2 - 1) * (1 - m), m).clamp(-1, 1) + 1) * 0.5
        return (x * (1 - m) + y * m)[:, :, :h, :w]


def export(ckpt: str | Path, out: Path, size: int = 512, opset: int = 17, check: bool = True) -> Path:
    net = load_generator(str(ckpt)).eval()
    ref = LamaInterface(net).eval()
    folded = LamaInterface(fold_weight_standardization(load_generator(str(ckpt)).eval())).eval()
    g = torch.Generator().manual_seed(0)
    img = torch.rand(1, 3, size, size, generator=g)
    mask = torch.zeros(1, 1, size, size)
    mask[..., size // 4 : size // 2, size // 3 : size // 2] = 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            folded, (img, mask), str(out), input_names=["image", "mask"], output_names=["output"],
            dynamic_axes={k: {0: "batch", 2: "height", 3: "width"} for k in ("image", "mask", "output")},
            opset_version=opset, dynamo=False,
        )
        expected = ref(img, mask).numpy()
    if check:  # parity at the trace size and at an odd size (exercises the in-graph padding)
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        err = float(np.abs(sess.run(None, {"image": img.numpy(), "mask": mask.numpy()})[0] - expected).max())
        oi, om = img[..., : size - 3, : size - 1], mask[..., : size - 3, : size - 1]
        with torch.no_grad():
            want = ref(oi, om).numpy()
        err = max(err, float(np.abs(sess.run(None, {"image": oi.numpy(), "mask": om.numpy()})[0] - want).max()))
        if err > 2e-3:
            raise RuntimeError(f"onnx output differs from torch by {err:.2e}")
        print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB), max |onnx - torch| = {err:.1e}")
    return out


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", help="checkpoint from train_inpaint.py, or 'aot' for released weights")
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", type=int, default=512, help="trace / parity-check size")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args(argv)
    return export(a.ckpt, a.out, a.size, a.opset, not a.no_check)


if __name__ == "__main__":
    main()
