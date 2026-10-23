"""Export a (fine-tuned) manga-ocr checkpoint to the worker's single-graph ONNX interface.

The graph matches weights/koharu-manga-ocr.onnx, which the `manga_ocr` OCR method loads:

    inputs   image      float32 [B, 3, 224, 224]   grey->RGB, (x/255 - 0.5) / 0.5
             token_ids  int64   [B, T]             starts with [CLS]=2
    outputs  logits                 float32 [B, T, 6144]
             encoder_hidden_states  float32 [B, 197, 768]

`encoder_hidden_states` is the only other rank-3 output, which is what the method's graph
splitter (ocr/manga_ocr_onnx._split_graph) keys on to cut the encoder from the decoder; node
names keep the `/encoder/...`, `/decoder/cls/predictions/...` scopes of the stock export. The
vocab is copied next to the model as <name>.vocab.txt.

After export the script checks the ONNX logits against torch (random and, with --images, real
crops) and, unless --no-decode-check, that the worker's ONNX decoder reads the same text as
transformers' greedy generate.

Usage:
    python training/export_ocr_onnx.py --model training/runs/ocr-ft/best --out weights/manga-ocr-ft.onnx
    terjemahin-worker translate page.png --ocr manga_ocr --opt ocr.weights=weights/manga-ocr-ft.onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terjemahin_worker.ocr.manga_ocr_onnx import (
    EOS,
    SIZE,
    START,
    MangaOcrOnnx,
    decode_tokens,
    load_vocab,
    preprocess,
)

REPO = "kha-white/manga-ocr-base"


def build_wrapper(model):
    import torch

    class OcrGraph(torch.nn.Module):
        """(image, token_ids) -> (logits, encoder states); attribute names set the ONNX scopes."""

        def __init__(self, m) -> None:
            super().__init__()
            self.encoder = m.encoder
            self.decoder = m.decoder
            self.proj = getattr(m, "enc_to_dec_proj", None)

        def forward(self, image, token_ids):
            hidden = self.encoder(pixel_values=image).last_hidden_state
            if self.proj is not None:
                hidden = self.proj(hidden)
            logits = self.decoder(
                input_ids=token_ids, encoder_hidden_states=hidden, use_cache=False
            ).logits
            return logits, hidden

    return OcrGraph(model).eval()


def export(model_id: str, out: Path, opset: int = 17) -> Path:
    import torch
    from transformers import VisionEncoderDecoderModel

    model = VisionEncoderDecoderModel.from_pretrained(model_id, attn_implementation="eager")
    graph = build_wrapper(model.eval())
    image = torch.zeros(2, 3, SIZE, SIZE)
    tokens = torch.tensor([[START, START, 100, 200], [START, START, 300, 400]], dtype=torch.long)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            graph,
            (image, tokens),
            str(out),
            input_names=["image", "token_ids"],
            output_names=["logits", "encoder_hidden_states"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "token_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
                "encoder_hidden_states": {0: "batch_size"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    vocab = Path(model_id) / "vocab.txt"
    if not vocab.exists():
        from huggingface_hub import hf_hub_download

        vocab = Path(hf_hub_download(model_id, "vocab.txt"))
    shutil.copy(vocab, vocab_path(out))
    return out


def vocab_path(onnx_path: Path) -> Path:
    return onnx_path.with_suffix(".vocab.txt")


def check_logits(model_id: str, onnx_path: Path, crops: list[np.ndarray], seed: int = 0) -> dict:
    """Max |torch - onnxruntime| over logits for random inputs and real crops."""
    import onnxruntime as ort
    import torch
    from transformers import VisionEncoderDecoderModel

    rng = np.random.default_rng(seed)
    model = VisionEncoderDecoderModel.from_pretrained(model_id).eval()
    graph = build_wrapper(model)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    cases = [rng.uniform(-1, 1, (2, 3, SIZE, SIZE)).astype(np.float32)]
    if crops:
        cases.append(preprocess(crops[:4]))
    report = {"max_abs": 0.0, "max_rel": 0.0, "argmax_agree": 1.0}
    for pixels in cases:
        T = int(rng.integers(3, 12))
        ids = np.concatenate(
            [np.full((len(pixels), 1), START), rng.integers(5, 6144, (len(pixels), T))], 1
        ).astype(np.int64)
        with torch.inference_mode():
            ref, _ = graph(torch.from_numpy(pixels), torch.from_numpy(ids))
        ref = ref.numpy()
        got = sess.run(["logits"], {"image": pixels, "token_ids": ids})[0]
        diff = float(np.abs(ref - got).max())
        report["max_abs"] = max(report["max_abs"], diff)
        report["max_rel"] = max(report["max_rel"], diff / float(np.abs(ref).max()))
        agree = float((ref.argmax(-1) == got.argmax(-1)).mean())
        report["argmax_agree"] = min(report["argmax_agree"], agree)
    return report


def check_decode(model_id: str, onnx_path: Path, crops: list[np.ndarray]) -> dict:
    """The worker's split-graph ONNX decoder vs transformers greedy generate, text for text."""
    import torch
    from transformers import VisionEncoderDecoderModel

    model = VisionEncoderDecoderModel.from_pretrained(model_id).eval()
    vocab = load_vocab(vocab_path(onnx_path))
    ocr = MangaOcrOnnx(onnx_path, vocab_path(onnx_path))
    pixels = preprocess(crops)
    with torch.inference_mode():
        ref = model.generate(
            torch.from_numpy(pixels), max_length=ocr.max_len + 1, num_beams=1, do_sample=False,
            no_repeat_ngram_size=ocr.no_repeat_ngram, decoder_start_token_id=START,
            eos_token_id=EOS, pad_token_id=0,
        ).tolist()
    got = ocr.generate(pixels)
    rows = []
    for r, (ids, conf) in zip(ref, got):
        r = r[1:]
        r = r[: r.index(EOS)] if EOS in r else r
        rows.append({"torch": decode_tokens(r, vocab), "onnx": decode_tokens(ids, vocab),
                     "conf": round(conf, 4)})
    same = sum(x["torch"] == x["onnx"] for x in rows)
    return {"split_graph": ocr.decoder is not None, "same": same, "n": len(rows), "rows": rows}


def load_crops(paths: list[Path], limit: int = 8) -> list[np.ndarray]:
    files: list[Path] = []
    for p in paths:
        if p.suffix == ".jsonl":
            rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]
            files += [p.parent / r["image"] for r in rows]
        elif p.is_dir():
            files += sorted(p.glob("*.png")) + sorted(p.glob("*.jpg"))
        else:
            files.append(p)
    crops = [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in files[:limit]]
    return [c for c in crops if c is not None]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=REPO, help="checkpoint folder from train_ocr.py, or HF id")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--images", type=Path, nargs="*", default=[],
                    help="crops (files, folders or a jsonl) for the parity checks")
    ap.add_argument("--tol", type=float, default=1e-3, help="max abs logit difference allowed")
    ap.add_argument("--no-decode-check", action="store_true")
    a = ap.parse_args(argv)

    export(a.model, a.out, a.opset)
    crops = load_crops(a.images)
    report = {"onnx": str(a.out), "vocab": str(vocab_path(a.out)),
              "mb": round(a.out.stat().st_size / 2**20, 1),
              "logits": check_logits(a.model, a.out, crops)}
    if crops and not a.no_decode_check:
        report["decode"] = check_decode(a.model, a.out, crops)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["logits"]["max_abs"] > a.tol:
        raise SystemExit(f"logit mismatch {report['logits']['max_abs']:.2e} > tol {a.tol:.0e}")
    return report


if __name__ == "__main__":
    main()
