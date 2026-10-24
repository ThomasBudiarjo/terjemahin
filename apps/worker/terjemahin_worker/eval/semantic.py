"""Meaning-level translation adequacy: multilingual sentence embeddings on onnxruntime.

chrF against one colloquial reference cannot tell a free but correct paraphrase ("Eh, itu
saja?" for "Eh, cuma itu?", chrF 22) from a wrong meaning ("Hmm, ya, baik-baik saja" for "dia
baik, kan?", chrF 19). Cosine similarity of paraphrase-multilingual-MiniLM-L12-v2 embeddings
does: correct paraphrases on page01 score >= 0.89, the wrong meanings 0.59-0.75. The int8 ONNX
export (~120 MB) is fetched from the Hugging Face hub on first use.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)

REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_FILE = "onnx/model_quint8_avx2.onnx"


class Embedder:
    def __init__(self, repo: str = REPO, model_file: str = MODEL_FILE, offline: bool = False) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        def get(f: str) -> str:
            return hf_hub_download(repo, f, local_files_only=offline)

        self.tok = Tokenizer.from_file(get("tokenizer.json"))
        self.tok.enable_padding()
        self.tok.enable_truncation(128)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        self.sess = ort.InferenceSession(get(model_file), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalised mean-pooled embeddings, one row per text."""
        if not texts:
            return np.zeros((0, 384), np.float32)
        enc = self.tok.encode_batch([t or " " for t in texts])
        ids = np.array([e.ids for e in enc], np.int64)
        am = np.array([e.attention_mask for e in enc], np.int64)
        feed = {"input_ids": ids, "attention_mask": am}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        h = self.sess.run(None, feed)[0]
        v = (h * am[..., None]).sum(1) / np.maximum(am.sum(1, keepdims=True), 1)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def similarity(self, hyps: list[str], refs: list[str]) -> np.ndarray:
        """[len(hyps), len(refs)] cosine similarities; an empty hypothesis scores 0."""
        s = self.embed(hyps) @ self.embed(refs).T
        s[[not h.strip() for h in hyps]] = 0.0
        return s


@lru_cache(maxsize=1)
def get_embedder() -> Embedder | None:
    """Shared instance, or None (with a warning) when the model cannot be fetched or loaded."""
    try:
        return Embedder(offline=os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0"))
    except Exception as e:  # noqa: BLE001 - offline / no cache: the metric is reported as n/a
        log.warning("semantic adequacy unavailable (%s: %s)", type(e).__name__, e)
        return None
