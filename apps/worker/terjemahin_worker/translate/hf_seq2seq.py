"""Offline seq2seq MT (NLLB-200, M2M100) through transformers, with manga-aware pre/post steps.

Models load lazily on the first translate and are shared process-wide per (model, device, quant),
so "nllb" and a "chain" step using nllb hold one copy. Each bubble goes through
textproc.prepare -> batched beam search -> post-translation check (repetition loops, leftover
source script, missing glossary terms) -> retry with stricter decoding -> textproc.finish.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..models import GlossaryEntry
from ..registry import TRANSLATORS
from . import textproc as tp
from .base import TranslationItem, TranslationRequest

log = logging.getLogger(__name__)


@dataclass
class _Loaded:
    tokenizer: Any
    model: Any
    lock: threading.Lock


_MODELS: dict[tuple[str, str, bool], _Loaded] = {}
_MODELS_LOCK = threading.Lock()


def load_model(repo: str, device: str = "cpu", quantize: bool = False) -> _Loaded:
    """Load (or reuse) a tokenizer+model; tries the local HF cache before touching the network."""
    key = (repo, device, quantize)
    with _MODELS_LOCK:
        if key in _MODELS:
            return _MODELS[key]
        # transformers would otherwise phone the hub on every load to offer a safetensors PR
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        def _from(local: bool) -> tuple[Any, Any]:
            tok = AutoTokenizer.from_pretrained(repo, local_files_only=local)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(repo, local_files_only=local)
            return tok, mdl

        try:
            tok, mdl = _from(True)
        except OSError:
            log.info("downloading %s from the Hugging Face hub", repo)
            tok, mdl = _from(False)
        mdl = mdl.to(device).eval()
        if quantize and device == "cpu":
            mdl = torch.ao.quantization.quantize_dynamic(mdl, {torch.nn.Linear}, dtype=torch.qint8)
        mdl.generation_config.max_length = None  # we always pass max_new_tokens
        _MODELS[key] = _Loaded(tok, mdl, threading.Lock())
        return _MODELS[key]


def _batches(order: list[int], size: int) -> Iterable[list[int]]:
    for i in range(0, len(order), size):
        yield order[i : i + size]


class Seq2SeqTranslator:
    """Japanese bubbles -> target language with a multilingual seq2seq model."""

    vision = False

    def __init__(
        self,
        repo: str,
        family: str,
        device: str = "cpu",
        num_beams: int = 4,
        batch_size: int = 8,
        threads: int | None = None,
        max_ratio: float = 2.5,
        max_tokens: int = 200,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.0,
        retries: int = 1,
        interjections: bool = True,
        glossary: bool = True,
        quantize: bool = False,
        casual: bool = True,
        pre_dict: dict[str, str] | None = None,
        post_dict: dict[str, str] | None = None,
    ) -> None:
        if family not in ("nllb", "m2m100"):
            raise ValueError(f"unknown model family '{family}'")
        self.repo, self.family, self.device = repo, family, device
        self.num_beams, self.batch_size = max(1, int(num_beams)), max(1, int(batch_size))
        self.threads = threads
        self.max_ratio, self.max_tokens = float(max_ratio), int(max_tokens)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.repetition_penalty = float(repetition_penalty)
        self.retries = int(retries)
        self.interjections, self.use_glossary, self.quantize = interjections, glossary, quantize
        self.casual = casual
        self.pre_dict, self.post_dict = pre_dict or {}, post_dict or {}

    def code(self, lang: str) -> str:
        return tp.nllb_code(lang) if self.family == "nllb" else tp.m2m_code(lang)

    def _loaded(self) -> _Loaded:
        if self.threads:
            import torch

            torch.set_num_threads(int(self.threads))
        return load_model(self.repo, self.device, self.quantize)

    def generate(self, texts: list[str], source: str, target: str, **overrides: Any) -> list[str]:
        """Raw model output for already-prepared texts (batched by length to limit padding)."""
        if not texts:
            return []
        import torch

        m = self._loaded()
        tok = m.tokenizer
        params = {
            "num_beams": self.num_beams,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "repetition_penalty": self.repetition_penalty,
            **overrides,
        }
        params["early_stopping"] = params["num_beams"] > 1
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [""] * len(texts)
        with m.lock, torch.inference_mode():
            tok.src_lang = self.code(source)
            tgt = self.code(target)
            bos = tok.convert_tokens_to_ids(tgt) if self.family == "nllb" else tok.get_lang_id(tgt)
            for idx in _batches(order, self.batch_size):
                enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to(self.device)
                n_in = int(enc["attention_mask"].sum(1).max())
                gen = m.model.generate(
                    **enc,
                    forced_bos_token_id=bos,
                    max_new_tokens=min(self.max_tokens, int(n_in * self.max_ratio) + 8),
                    **params,
                )
                for i, s in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                    out[i] = s.strip()
        return out

    def _check(self, prep: tp.Prepared, out: str, target: str) -> bool:
        if tp.is_bad(prep.text, out, target):
            return False
        fixed = tp.restore_terms(out, prep.terms)
        return all(tp.has_term(fixed, t) for t in prep.terms)

    def translate_texts(
        self,
        texts: list[str],
        source: str,
        target: str,
        glossary: Iterable[GlossaryEntry] = (),
    ) -> list[str]:
        if source == "auto":
            source = tp.detect_lang(texts)
        if tp.iso(source) == tp.iso(target):
            return list(texts)
        gl = list(glossary) if self.use_glossary else []
        preps = [
            tp.prepare(tp.apply_dict(t, self.pre_dict), source, target, gl, self.interjections)
            for t in texts
        ]
        # text that is only names/latin after preparation ("Sakamoto-kun？") needs no model
        todo = [i for i, p in enumerate(preps) if p.text and tp.has_source_script(p.text, source)]
        raw = {i: p.text for i, p in enumerate(preps) if p.text and i not in todo}
        raw.update(zip(todo, self.generate([preps[i].text for i in todo], source, target)))

        bad = [i for i in todo if not self._check(preps[i], raw[i], target)]
        strict = {
            "num_beams": max(self.num_beams, 5),
            "no_repeat_ngram_size": 2,
            "repetition_penalty": max(self.repetition_penalty, 1.3),
        }
        for attempt in range(self.retries):
            if not bad:
                break
            log.info("retranslating %d/%d bubbles (attempt %d)", len(bad), len(todo), attempt + 1)
            again = self.generate([preps[i].text for i in bad], source, target, **strict)
            for i, s in zip(bad, again):
                if self._check(preps[i], s, target) or tp.is_bad(preps[i].text, raw[i], target):
                    raw[i] = s
            bad = [i for i in bad if not self._check(preps[i], raw[i], target)]
        for i in bad:
            log.warning("translation still suspect after retries: %r -> %r", texts[i], raw[i])
            raw[i] = tp.collapse_repetition(raw[i])

        return [
            tp.apply_dict(tp.finish(p, raw.get(i, ""), target, self.casual), self.post_dict)
            for i, p in enumerate(preps)
        ]

    def translate(self, req: TranslationRequest) -> list[TranslationItem]:
        targets = self.translate_texts(
            req.sources, req.source_lang, req.target_lang, req.context.glossary
        )
        return [TranslationItem(source=s, target=t) for s, t in zip(req.sources, targets)]


_KNOWN = {
    "nllb": ("facebook/nllb-200-distilled-600M", "nllb",
             "NLLB-200 600M distilled (offline, CPU, ~2.5GB RAM): fluent ja->id; glossary + manga pre/post."),
    "nllb_big": ("facebook/nllb-200-distilled-1.3B", "nllb",
                 "NLLB-200 1.3B distilled (offline, downloads ~5GB on first use, slower)."),
    "m2m100": ("facebook/m2m100_418M", "m2m100",
               "M2M100 418M (offline, fastest): ja->id on par with nllb-600M on page01; ja->en pivot."),
    "m2m100_big": ("facebook/m2m100_1.2B", "m2m100",
                   "M2M100 1.2B (offline, downloads ~5GB on first use)."),
}


def _factory(repo: str, family: str):
    def build(settings, model: str | None = None, device: str | None = None, **opts: Any):
        """`model` may point at a fine-tuned checkpoint (HF id or local dir) of the same family."""
        return Seq2SeqTranslator(model or repo, family, device=device or settings.device, **opts)

    return build


for _name, (_repo, _family, _desc) in _KNOWN.items():
    TRANSLATORS.register(
        _name, description=_desc, requires=("torch", "transformers", "sentencepiece")
    )(_factory(_repo, _family))
