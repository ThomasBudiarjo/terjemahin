"""Fine-tune NLLB-200 / M2M100 on a parallel corpus (e.g. Japanese manga lines -> Indonesian).

Data: one or more `--data` files, either JSONL with {"src": ..., "tgt": ...} per line or TSV
(src<TAB>tgt). A held-out split (`--val-ratio`, or `--val FILE`) is scored with corpus chrF and
a smoothed word BLEU before training and at every eval; the best checkpoint is kept.

The 600M NLLB has a 256k-token shared embedding (~262M params, tied to the output layer). On CPU
or a small GPU the useful knobs are, from cheapest to most expressive:
    --decoder-layers 2      train only the last 2 decoder layers (+ final layer norm)
    --freeze-encoder        encoder fixed, whole decoder trains
    (default)               everything except the shared embedding/lm_head
    --train-embeddings      also train the shared embedding (alone: full fine-tune); combines
                            with either option above
Frozen parts get no gradients or optimizer state, so `--decoder-layers 2` fits in ~4 GB RAM.

The output dir is a normal Hugging Face checkpoint (config, safetensors, tokenizer) that the
worker's offline translators load directly:
    python training/train_translator.py --data data/manga_ja_id.jsonl --out training/runs/nllb-manga
    terjemahin-worker translate page.png -t nllb --opt translator.model=training/runs/nllb-manga
(use `-t m2m100` for an M2M100 base.)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from terjemahin_worker.translate import textproc as tp

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if path.suffix == ".jsonl":
            row = json.loads(line)
            s, t = row["src"], row["tgt"]
        else:
            s, _, t = line.partition("\t")
        s, t = s.strip(), t.strip()
        if s and t:
            pairs.append((s, t))
    return pairs


def split_pairs(pairs: list, val_ratio: float, seed: int) -> tuple[list, list]:
    """Deterministic shuffle split; duplicates of a validation source never leak into train."""
    if val_ratio <= 0 or len(pairs) < 2:
        return pairs, []
    srcs = sorted({s for s, _ in pairs})
    random.Random(seed).shuffle(srcs)
    val_src = set(srcs[: max(1, round(len(srcs) * val_ratio))])
    return [p for p in pairs if p[0] not in val_src], [p for p in pairs if p[0] in val_src]


def bleu(hyps: list[str], refs: list[str], order: int = 4) -> float:
    """Corpus BLEU on lowercased word tokens with add-one smoothing for n > 1 (0..100)."""
    tok = lambda s: re.findall(r"\w+|[^\w\s]", s.lower())
    match, total, hl, rl = [0] * order, [0] * order, 0, 0
    for h, r in zip(hyps, refs, strict=True):
        h, r = tok(h), tok(r)
        hl, rl = hl + len(h), rl + len(r)
        for n in range(1, order + 1):
            hc = Counter(tuple(h[i : i + n]) for i in range(len(h) - n + 1))
            rc = Counter(tuple(r[i : i + n]) for i in range(len(r) - n + 1))
            match[n - 1] += sum((hc & rc).values())
            total[n - 1] += sum(hc.values())
    if not hl or not match[0]:
        return 0.0
    logp = sum(math.log((m + (n > 0)) / (t + (n > 0))) for n, (m, t) in enumerate(zip(match, total)) if t)
    bp = min(0.0, 1 - rl / hl)
    return 100 * math.exp(bp + logp / order)


def family_of(tokenizer) -> str:
    return "nllb" if "nllb" in type(tokenizer).__name__.lower() else "m2m100"


def lang_codes(family: str, src: str, tgt: str) -> tuple[str, str]:
    f = tp.nllb_code if family == "nllb" else tp.m2m_code
    return f(src), f(tgt)


def freeze(model, train_embeddings: bool, freeze_encoder: bool, decoder_layers: int) -> tuple[int, int]:
    """Set requires_grad per the CLI policy; returns (trainable, total) parameter counts."""
    enc, dec = model.get_encoder(), model.get_decoder()
    for p in model.parameters():
        p.requires_grad = decoder_layers < 0
    if decoder_layers >= 0:  # implies a frozen encoder
        layers = dec.layers
        for layer in layers[len(layers) - decoder_layers :] if decoder_layers else []:
            for p in layer.parameters():
                p.requires_grad = True
        if getattr(dec, "layer_norm", None) is not None:
            for p in dec.layer_norm.parameters():
                p.requires_grad = True
    elif freeze_encoder:
        for p in enc.parameters():
            p.requires_grad = False
    for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
        for p in emb.parameters():
            p.requires_grad = train_embeddings
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


class Batcher:
    """Length-bucketed batches (less padding = faster CPU steps), reshuffled every epoch."""

    def __init__(self, pairs, tok, src_code, tgt_code, batch: int, max_len: int, seed: int) -> None:
        tok.src_lang, tok.tgt_lang = src_code, tgt_code
        self.enc = [tok(s, text_target=t, truncation=True, max_length=max_len) for s, t in pairs]
        self.pad = tok.pad_token_id
        self.batch, self.rng = batch, random.Random(seed)

    def epoch(self):
        import torch

        order = sorted(range(len(self.enc)), key=lambda i: len(self.enc[i]["input_ids"]) + self.rng.random())
        chunks = [order[i : i + self.batch] for i in range(0, len(order), self.batch)]
        self.rng.shuffle(chunks)
        for idx in chunks:
            L = max(len(self.enc[i]["input_ids"]) for i in idx)
            T = max(len(self.enc[i]["labels"]) for i in idx)
            ids = torch.full((len(idx), L), self.pad)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            lab = torch.full((len(idx), T), -100)
            for r, i in enumerate(idx):
                a, b = self.enc[i]["input_ids"], self.enc[i]["labels"]
                ids[r, : len(a)], att[r, : len(a)], lab[r, : len(b)] = torch.tensor(a), 1, torch.tensor(b)
            yield {"input_ids": ids, "attention_mask": att, "labels": lab}


def translate(model, tok, texts, src_code, tgt_code, device, beams: int = 4, batch: int = 16,
              no_repeat: int = 3) -> list[str]:
    """Plain beam search with the worker's default n-gram blocking (no textproc pre/post)."""
    import torch

    tok.src_lang = src_code
    bos = tok.convert_tokens_to_ids(tgt_code) if "nllb" in type(tok).__name__.lower() else tok.get_lang_id(tgt_code)
    out = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i : i + batch], return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(device)
            gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=beams,
                                 no_repeat_ngram_size=no_repeat, early_stopping=beams > 1,
                                 max_new_tokens=int(enc["input_ids"].shape[1] * 2.5) + 8)
            out += [s.strip() for s in tok.batch_decode(gen, skip_special_tokens=True)]
    model.train()
    return out


def worker_chrf(model_dir: str | Path, family: str, val, src: str, tgt: str, device: str) -> dict:
    """Score through the worker's own translator (textproc pre/post, retries): what users get."""
    from terjemahin_worker.translate.hf_seq2seq import Seq2SeqTranslator

    tr = Seq2SeqTranslator(str(model_dir), family, device=device)
    hyps = tr.translate_texts([s for s, _ in val], src, tgt)
    return {"chrf": round(tp.chrf(hyps, [t for _, t in val]), 2),
            "bleu": round(bleu(hyps, [t for _, t in val]), 2)}


def evaluate(model, tok, val, src_code, tgt_code, device, beams: int) -> dict:
    if not val:
        return {}
    hyps = translate(model, tok, [s for s, _ in val], src_code, tgt_code, device, beams)
    refs = [t for _, t in val]
    return {"chrf": round(tp.chrf(hyps, refs), 2), "bleu": round(bleu(hyps, refs), 2),
            "samples": [{"src": s, "hyp": h, "ref": r} for (s, r), h in list(zip(val, hyps))[:5]]}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, nargs="+", required=True, help=".jsonl {src,tgt} or .tsv")
    ap.add_argument("--val", type=Path, nargs="*", default=[], help="explicit held-out files (their sources are removed from --data)")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF id or local dir (NLLB or M2M100)")
    ap.add_argument("--src", default="ja")
    ap.add_argument("--tgt", default="id")
    ap.add_argument("--out", type=Path, default=Path("training/runs/translator"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=0, help="optimizer steps (overrides --epochs)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--scheduler", choices=("linear", "cosine", "constant"), default="linear")
    ap.add_argument("--warmup", type=float, default=0.06, help="warmup steps, or a fraction < 1")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--decoder-layers", type=int, default=-1,
                    help="train only the last N decoder layers (-1 = no such limit)")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="encoder fixed (implied by --decoder-layers)")
    ap.add_argument("--train-embeddings", action="store_true",
                    help="also train the shared embedding / lm_head")
    ap.add_argument("--eval-every", type=int, default=0, help="steps between evals (0 = per epoch)")
    ap.add_argument("--eval-beams", type=int, default=4)
    ap.add_argument("--eval-max", type=int, default=200, help="score at most N held-out pairs")
    ap.add_argument("--no-baseline", action="store_true", help="skip the step-0 eval")
    ap.add_argument("--no-normalize", action="store_true",
                    help="keep Japanese sources as-is (default: textproc.normalize_ja, as at inference)")
    ap.add_argument("--eval-worker", action="store_true",
                    help="finally score base and best checkpoint through the worker translator")
    ap.add_argument("--save-dtype", choices=("float16", "bfloat16", "float32"), default="float16",
                    help="on-disk precision (loads back as float32); float16 halves the 2.4 GB")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler

    random.seed(a.seed), torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    norm = (lambda s: s) if a.no_normalize or tp.iso(a.src) != "ja" else tp.normalize_ja
    pairs = [(norm(s), t) for f in a.data for s, t in read_pairs(f)]
    if a.val:
        val = [(norm(s), t) for f in a.val for s, t in read_pairs(f)]
        held = {s for s, _ in val}
        train = [p for p in pairs if p[0] not in held]
        if len(train) < len(pairs):
            print(f"dropped {len(pairs) - len(train)} training pairs whose source is in --val", flush=True)
    else:
        train, val = split_pairs(pairs, a.val_ratio, a.seed)
    val = val[: a.eval_max]
    if not train:
        ap.error("no training pairs")

    def load(cls, **kw):
        try:
            return cls.from_pretrained(a.model, local_files_only=True, **kw)
        except OSError:
            return cls.from_pretrained(a.model, **kw)

    tok = load(AutoTokenizer)
    model = load(AutoModelForSeq2SeqLM).to(a.device)
    model.generation_config.max_length = None  # eval always passes max_new_tokens
    family = family_of(tok)
    src_code, tgt_code = lang_codes(family, a.src, a.tgt)
    trainable, total = freeze(model, a.train_embeddings, a.freeze_encoder, a.decoder_layers)
    batcher = Batcher(train, tok, src_code, tgt_code, a.batch, a.max_len, a.seed)
    n_batches = math.ceil(len(train) / a.batch)
    per_epoch = math.ceil(n_batches / a.grad_accum)  # a short tail of micro-batches still steps
    steps = a.steps or max(1, round(per_epoch * a.epochs))
    warmup = int(a.warmup * steps) if a.warmup < 1 else int(a.warmup)
    eval_every = a.eval_every or per_epoch
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    sched = get_scheduler(a.scheduler if a.scheduler != "constant" else "constant_with_warmup",
                          opt, num_warmup_steps=warmup, num_training_steps=steps)
    loss_f = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=a.label_smoothing)
    print(f"{family} {a.model}: {trainable / 1e6:.1f}M/{total / 1e6:.1f}M trainable, "
          f"train={len(train)} val={len(val)} steps={steps} ({per_epoch}/epoch)", flush=True)

    history: list[dict] = []
    best, best_state, best_metrics = -1.0, None, {}
    if val and not a.no_baseline:
        history.append({"step": 0, **evaluate(model, tok, val, src_code, tgt_code, a.device, a.eval_beams)})
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)

    def save(metrics: dict) -> None:
        """Weights stored in --save-dtype but config says float32, so loaders upcast on load."""
        a.out.mkdir(parents=True, exist_ok=True)
        model.to(getattr(torch, a.save_dtype)).save_pretrained(a.out)
        model.float()
        cfg_path = a.out / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["dtype"] = cfg["torch_dtype"] = "float32"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        tok.save_pretrained(a.out)
        meta = {"base": a.model, "family": family, "src": a.src, "tgt": a.tgt, "metrics": metrics,
                "translator": family, "opt": f"translator.model={a.out}"}
        (a.out / "terjemahin.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    model.train()
    step, epochs, t0, running = 0, 0, time.time(), []
    while step < steps:
        epochs += 1
        for k, batch in enumerate(batcher.epoch()):
            batch = {n: v.to(a.device) for n, v in batch.items()}
            logits = model(**batch).logits
            loss = loss_f(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
            (loss / a.grad_accum).backward()
            running.append(loss.item())
            micro = k % a.grad_accum + 1
            if micro < a.grad_accum and k + 1 < n_batches:
                continue
            if micro < a.grad_accum:  # epoch tail: rescale to a mean over the micro-batches seen
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(a.grad_accum / micro)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == steps:
                print(f"step {step}/{steps} loss {sum(running) / len(running):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/step", flush=True)
                running = []
            if val and (step % eval_every == 0 or step == steps):
                history.append({"step": step, **evaluate(model, tok, val, src_code, tgt_code,
                                                         a.device, a.eval_beams)})
                print(json.dumps(history[-1], ensure_ascii=False), flush=True)
                if history[-1]["chrf"] > best:  # keep only trainable tensors, save once at the end
                    best, best_metrics = history[-1]["chrf"], history[-1]
                    best_state = {n: p.detach().clone() for n, p in model.named_parameters()
                                  if p.requires_grad}
            if step >= steps:
                break
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    save(best_metrics if val else {"step": step})
    result = {"out": str(a.out), "family": family, "steps": step, "epochs": epochs, "trainable": trainable,
              "history": history, "best_chrf": best if val else None}
    if history and history[0]["step"] == 0 and val and best <= history[0]["chrf"]:
        print(f"warning: no eval beat the base model (chrF {history[0]['chrf']}); "
              f"saved checkpoint scores {best}", flush=True)
    if a.eval_worker and val:
        del model
        result["worker"] = {"base": worker_chrf(a.model, family, val, a.src, a.tgt, a.device),
                            "tuned": worker_chrf(a.out, family, val, a.src, a.tgt, a.device)}
        print(json.dumps({"worker": result["worker"]}), flush=True)
    (a.out / "train_history.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
