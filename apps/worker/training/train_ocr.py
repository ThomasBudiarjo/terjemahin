"""Fine-tune manga-ocr (kha-white/manga-ocr-base) on OCR crops, with CER on a held-out split.

Data is one or more folders/files of jsonl rows {"image": path relative to the jsonl, "text"},
e.g. what training/synth.py writes to out/ocr/{train,val}.jsonl, or real crops labelled by
hand. Crops go through the exact preprocessing the worker uses at inference
(ocr.manga_ocr_onnx.preprocess), and targets are NFKC-normalised like the model's vocab, so
the fine-tuned model is a drop-in replacement for the stock one.

Usage:
    python training/train_ocr.py --data data/synth/ocr --out training/runs/ocr-ft \
        --steps 3000 --batch 16 --lr 2e-5 --eval-every 500
    python training/export_ocr_onnx.py --model training/runs/ocr-ft/best \
        --out weights/manga-ocr-ft.onnx
    terjemahin-worker translate page.png --ocr manga_ocr --opt ocr.weights=weights/manga-ocr-ft.onnx

Everything runs on CPU (slowly); `--freeze-encoder` trains only the 2-layer decoder, which is
~4x faster per step and enough to adapt to new fonts' glyph statistics or a vocabulary shift.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terjemahin_worker.ocr.common import normalize_ja
from terjemahin_worker.ocr.manga_ocr_onnx import (
    EOS,
    START,
    decode_tokens,
    load_vocab,
    preprocess,
)

REPO = "kha-white/manga-ocr-base"
PAD, UNK = 0, 1


@dataclass
class Sample:
    image: Path
    text: str


def read_jsonl(path: Path) -> list[Sample]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.append(Sample(path.parent / r["image"], r["text"]))
    return rows


def load_data(sources: list[Path], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Folders contribute train.jsonl / val.jsonl; bare jsonl files are split by val_ratio."""
    train, val = [], []
    for src in sources:
        if src.is_dir():
            train += read_jsonl(src / "train.jsonl") if (src / "train.jsonl").exists() else []
            val += read_jsonl(src / "val.jsonl") if (src / "val.jsonl").exists() else []
        else:
            rows = read_jsonl(src)
            random.Random(seed).shuffle(rows)
            k = round(len(rows) * val_ratio)
            val += rows[:k]
            train += rows[k:]
    if not train:
        raise SystemExit(f"no training rows in {sources}")
    return train, val


class Tokenizer:
    """manga-ocr's vocab is one token per character, so this is a dict lookup (no MeCab).

    Targets are `[CLS] chars [SEP]` as in manga-ocr's own training (the HF tokenizer adds both),
    so with decoder start [CLS] the model's first prediction is [CLS] again. Dropping that
    leading [CLS] makes the stock model's teacher-forced loss ~30x worse.
    """

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab
        self.ids = {t: i for i, t in enumerate(vocab)}

    def encode(self, text: str, max_len: int) -> list[int]:
        """Over-long targets keep their prefix but lose [SEP]: a [SEP] after a cut would teach
        the model to stop reading early."""
        text = "".join(unicodedata.normalize("NFKC", text).split())
        ids = [START] + [self.ids.get(c, UNK) for c in text]
        return ids[:max_len] if len(ids) >= max_len else ids + [EOS]

    def decode(self, ids: list[int]) -> str:
        return decode_tokens(ids, self.vocab)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild geometric/photometric noise; crops already vary in margin, font and degradation."""
    h, w = img.shape[:2]
    if rng.random() < 0.3:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-3, 3), rng.uniform(0.95, 1.05))
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.2, 1.0))
    if rng.random() < 0.3:
        img = np.clip(img * rng.uniform(0.7, 1.1) + rng.uniform(-20, 30), 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        q = int(rng.integers(40, 90))
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    return img


def batch_tensors(samples: list[Sample], tok: Tokenizer, max_len: int, rng=None):
    import torch

    crops = []
    for s in samples:
        img = cv2.imread(str(s.image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(s.image)
        crops.append(augment(img, rng) if rng is not None else img)
    pixels = torch.from_numpy(preprocess(crops))
    ids = [tok.encode(s.text, max_len) for s in samples]
    labels = torch.full((len(ids), max(map(len, ids))), -100, dtype=torch.long)
    for i, seq in enumerate(ids):
        labels[i, : len(seq)] = torch.tensor(seq)
    return pixels, labels


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate(model, samples: list[Sample], tok: Tokenizer, max_len: int, batch: int) -> dict:
    """Greedy decoding with the worker's settings; CER over normalize_ja'd strings."""
    import torch

    if not samples:
        return {"cer": None, "exact": None, "n": 0}
    model.eval()
    errors = chars = exact = 0
    examples = []
    with torch.inference_mode():
        for i in range(0, len(samples), batch):
            chunk = samples[i : i + batch]
            pixels, _ = batch_tensors(chunk, tok, max_len)
            out = model.generate(
                pixels, max_length=max_len + 1, num_beams=1, do_sample=False,
                no_repeat_ngram_size=3, decoder_start_token_id=START, eos_token_id=EOS,
                pad_token_id=PAD,
            )
            for s, seq in zip(chunk, out.tolist()):
                seq = seq[1:]
                seq = seq[: seq.index(EOS)] if EOS in seq else seq
                pred, ref = tok.decode(seq), normalize_ja(unicodedata.normalize("NFKC", s.text))
                errors += edit_distance(pred, ref)
                chars += max(1, len(ref))
                exact += pred == ref
                if len(examples) < 8:
                    examples.append({"ref": ref, "pred": pred})
    model.train()
    return {"cer": errors / chars, "exact": exact / len(samples), "n": len(samples),
            "examples": examples}


def save(model, vocab_src: Path, out: Path, info: dict) -> None:
    import shutil

    if out.is_symlink():
        out.unlink()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    shutil.copy(vocab_src, out / "vocab.txt")
    (out / "train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


def link(path: Path, target: str) -> None:
    import shutil

    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    path.symlink_to(target, target_is_directory=True)


def resolve_vocab(model_id: str) -> Path:
    local = Path(model_id) / "vocab.txt"
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model_id, "vocab.txt"))


def train(a: argparse.Namespace) -> dict:
    import torch
    from transformers import VisionEncoderDecoderModel, get_linear_schedule_with_warmup

    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    train_rows, val_rows = load_data(a.data, a.val_ratio, a.seed)
    val_rows = val_rows[: a.max_eval]
    vocab_path = resolve_vocab(a.model)
    tok = Tokenizer(load_vocab(vocab_path))
    unk = sum(tok.encode(s.text, 10**6).count(UNK) for s in train_rows)
    print(f"train={len(train_rows)} val={len(val_rows)} unk_chars={unk}", flush=True)

    model = VisionEncoderDecoderModel.from_pretrained(a.model)
    model.config.decoder_start_token_id, model.config.pad_token_id = START, PAD
    if a.freeze_encoder:
        model.encoder.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.steps or math.ceil(len(train_rows) / a.batch) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, min(a.warmup, steps // 10), steps)

    sets = {"val": val_rows} if val_rows else {}
    for src in a.eval or []:
        rows = read_jsonl(src / "val.jsonl" if src.is_dir() else src)
        sets[src.stem if src.is_file() else src.name] = rows[: a.max_eval]

    def run_evals(tag: str) -> dict:
        res = {k: evaluate(model, rows, tok, a.max_len, a.batch) for k, rows in sets.items()}
        if a.freeze_encoder:
            model.encoder.eval()
        print(f"  [{tag}] " + "  ".join(f"{k}: cer={r['cer']:.4f} exact={r['exact']:.3f}"
                                         for k, r in res.items()), flush=True)
        return res

    info: dict = {"args": {k: str(v) for k, v in vars(a).items()}, "train": len(train_rows),
                  "eval_sets": {k: len(v) for k, v in sets.items()}, "history": []}
    evaluating = bool(a.eval_every and sets)
    if evaluating:
        info["baseline"] = run_evals("baseline")
    best = math.inf
    model.train()
    if a.freeze_encoder:
        model.encoder.eval()  # no dropout in a frozen encoder
    order: list[int] = []
    t0 = time.time()
    for step in range(1, steps + 1):
        if len(order) < a.batch:
            order += rng.permutation(len(train_rows)).tolist()
        idx, order = order[: a.batch], order[a.batch :]
        pixels, labels = batch_tensors([train_rows[i] for i in idx], tok, a.max_len, rng)
        loss = model(pixel_values=pixels, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        info["history"].append({"step": step, "loss": round(loss.item(), 5)})
        if step % a.log_every == 0 or step == steps:
            print(f"step {step}/{steps} loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / step:.2f}s/step", flush=True)
        if evaluating and (step % a.eval_every == 0 or step == steps):
            res = run_evals(f"step {step}")
            info["history"][-1]["eval"] = {k: {m: r[m] for m in ("cer", "exact")}
                                           for k, r in res.items()}
            info["final"] = res
            key = "val" if "val" in res else next(iter(res))
            if res[key]["cer"] < best:
                best = res[key]["cer"]
                info["best"] = {"step": step, "selected_on": key, **res}
                save(model, vocab_path, a.out / "best", info)
    info["seconds"] = round(time.time() - t0, 1)
    # one ~440MB copy when best == final: the other name is a symlink
    if info.get("best", {}).get("step") == steps:
        link(a.out / "final", "best")
        (a.out / "best" / "train_info.json").write_text(json.dumps(info, indent=2,
                                                                   ensure_ascii=False))
    else:
        save(model, vocab_path, a.out / "final", info)
        if "best" not in info:
            link(a.out / "best", "final")
    (a.out / "metrics.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return info


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, nargs="+", required=True,
                    help="folders with train/val.jsonl, or jsonl files")
    ap.add_argument("--out", type=Path, default=Path("training/runs/ocr-ft"))
    ap.add_argument("--model", default=REPO, help="HF repo id or a saved checkpoint folder")
    ap.add_argument("--steps", type=int, default=0, help="optimizer steps (0: use --epochs)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--val-ratio", type=float, default=0.1, help="for bare jsonl inputs")
    ap.add_argument("--max-eval", type=int, default=200, help="val crops used for CER")
    ap.add_argument("--eval-every", type=int, default=500, help="0 disables evaluation")
    ap.add_argument("--eval", type=Path, nargs="*", default=[],
                    help="extra held-out sets (folder with val.jsonl, or a jsonl), e.g. real crops")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
