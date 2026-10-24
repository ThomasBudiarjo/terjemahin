"""manga-ocr (kha-white/manga-ocr-base) on onnxruntime, no torch.

The koharu export is one graph `(image, token_ids) -> logits`, so every decoding step would
re-run the ViT encoder. At load time the graph is re-cut (see `_split_graph`): the encoder,
plus the cross-attention K/V projections of the decoder, run once per crop; each step only
runs the token-dependent part of the 2-layer BERT decoder, and the LM head only for the newest
position. That is ~8x faster than the single graph per page. If the cut fails the full graph
is used.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from ..models import TextRegion
from ..registry import OCRS
from ..weights import ensure
from .base import OcrResult
from .common import estimate_colors, normalize_ja, text_pieces

log = logging.getLogger(__name__)

SIZE = 224
START, EOS = 2, 3  # [CLS], [SEP] in the char-level BERT vocab
SPECIAL = {0, 1, 2, 3, 4}  # [PAD] [UNK] [CLS] [SEP] [MASK]


def preprocess(crops_bgr: list[np.ndarray]) -> np.ndarray:
    """Exactly manga-ocr's input: grey -> RGB, PIL bilinear resize to 224, (x/255 - .5)/.5."""
    from PIL import Image

    out = np.empty((len(crops_bgr), 3, SIZE, SIZE), np.float32)
    for i, c in enumerate(crops_bgr):
        rgb = Image.fromarray(np.ascontiguousarray(c[..., ::-1]))
        g = np.asarray(rgb.convert("L").resize((SIZE, SIZE), Image.BILINEAR), np.float32)
        out[i] = g / 127.5 - 1.0
    return out


def load_vocab(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").split("\n")[:-1]


def decode_tokens(ids: list[int], vocab: list[str]) -> str:
    return normalize_ja("".join(vocab[i].removeprefix("##") for i in ids if i not in SPECIAL))


def banned_ngram_tokens(seq: list[int], n: int) -> set[int]:
    """Tokens that would repeat an n-gram already in `seq` (HF no_repeat_ngram_size)."""
    if n <= 0 or len(seq) < n:
        return set()
    prefix = tuple(seq[len(seq) - n + 1 :])
    return {seq[i + n - 1] for i in range(len(seq) - n + 1) if tuple(seq[i : i + n - 1]) == prefix}


class MangaOcrBase:
    """Crop -> batch -> decode -> join, shared by the ONNX and transformers backends.

    A region is normally one crop of the whole padded block; extremely long blocks are read
    in pieces (see common.text_pieces) whose texts are concatenated.
    """

    vocab: list[str]

    def __init__(
        self,
        max_len: int = 128,
        batch_size: int = 8,
        no_repeat_ngram: int = 3,
        pad: float = 0.12,
        split_aspect: float = 10.0,
        chunk_aspect: float = 4.0,
    ) -> None:
        from ..opts import as_float, as_int

        self.max_len = as_int(max_len, "ocr max_len", 1, 512)
        self.batch_size = as_int(batch_size, "ocr batch_size", 1, 1024)
        self.no_repeat_ngram = as_int(no_repeat_ngram, "ocr no_repeat_ngram", 0)
        self.pad = as_float(pad, "ocr pad", 0.0, 2.0)
        self.split_aspect = as_float(split_aspect, "ocr split_aspect", 1.0)
        self.chunk_aspect = as_float(chunk_aspect, "ocr chunk_aspect", 1.0)

    def generate(self, pixels: np.ndarray) -> list[tuple[list[int], float]]:
        """(token ids without the start token, confidence in 0..1) per image."""
        raise NotImplementedError

    def recognize(self, image_bgr: np.ndarray, regions: list[TextRegion]) -> list[OcrResult]:
        pieces = [
            [
                p
                for p in text_pieces(image_bgr, r, self.pad, self.split_aspect, self.chunk_aspect)
                if p.size
            ]
            for r in regions
        ]
        flat = [p for ps in pieces for p in ps]
        decoded: list[tuple[str, float]] = []
        for i in range(0, len(flat), self.batch_size):
            for ids, conf in self.generate(preprocess(flat[i : i + self.batch_size])):
                decoded.append((decode_tokens(ids, self.vocab), conf))
        results, k = [], 0
        for ps in pieces:
            parts, k = decoded[k : k + len(ps)], k + len(ps)
            if not parts:
                results.append(OcrResult(text="", confidence=0.0))
                continue
            weights = [max(1, len(t)) for t, _ in parts]
            conf = sum(c * w for (_, c), w in zip(parts, weights)) / sum(weights)
            fg, bg = estimate_colors(np.concatenate([p.reshape(-1, 1, 3) for p in ps]))
            text = "".join(t for t, _ in parts)
            results.append(OcrResult(text=text, confidence=conf, fg_color=fg, bg_color=bg))
        return results


_SESSIONS: OrderedDict[tuple[str, int, bool], tuple[Any, Any, list[str], Any]] = OrderedDict()
_SESSIONS_LOCK = threading.Lock()
MAX_SESSIONS = 2  # each set is ~1 GB; options such as batch_size do not need their own


def _sessions(model: str, threads: int, split: bool) -> tuple[Any, Any, list[str], Any]:
    """(encoder, decoder, cached names, full graph) for a model file, shared by every instance
    with the same threads/split: a new instance per option set (batch_size=2, 3, ...) used to
    load another ~1 GB of sessions each time, growing a long-lived server without bound."""
    import onnxruntime as ort

    key = (str(model), int(threads), bool(split))
    with _SESSIONS_LOCK:
        if key in _SESSIONS:
            _SESSIONS.move_to_end(key)
            return _SESSIONS[key]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        cpu = ["CPUExecutionProvider"]
        encoder = decoder = full = None
        cached: list[str] = []
        if split:
            try:
                enc, dec, cached = _split_graph(str(model))
                encoder = ort.InferenceSession(enc, so, providers=cpu)
                decoder = ort.InferenceSession(dec, so, providers=cpu)
            except Exception:
                log.exception("could not split manga-ocr graph; decoding with the full graph")
                encoder = decoder = None
        if decoder is None:
            full = ort.InferenceSession(str(model), so, providers=cpu)
        _SESSIONS[key] = (encoder, decoder, cached, full)
        while len(_SESSIONS) > MAX_SESSIONS:
            _SESSIONS.popitem(last=False)
        return _SESSIONS[key]


class MangaOcrOnnx(MangaOcrBase):
    def __init__(
        self, model: str | Path, vocab: str | Path, threads: int = 0, split: bool = True, **opts
    ) -> None:
        from ..opts import as_bool, as_int

        super().__init__(**opts)
        threads = as_int(threads, "ocr threads", 0, 1024)
        split = as_bool(split, "ocr split")
        self.vocab = load_vocab(vocab)
        self.encoder, self.decoder, self.cached, self.full = _sessions(str(model), threads, split)

    def generate(self, pixels: np.ndarray) -> list[tuple[list[int], float]]:
        """Batched greedy decoding with HF's no-repeat-ngram rule; finished rows leave the batch."""
        B = len(pixels)
        cached = self.encoder.run(self.cached, {"image": pixels}) if self.decoder else []
        seqs = [[START] for _ in range(B)]
        logp = [0.0] * B
        active = list(range(B))
        for _ in range(self.max_len):
            tokens = np.array([seqs[i] for i in active], np.int64)
            if self.decoder is not None:
                feed = {n: c[active] for n, c in zip(self.cached, cached)}
                feed["token_ids"] = tokens
                logits = self.decoder.run(["logits"], feed)[0][:, -1]
            else:
                feed = {"image": pixels[active], "token_ids": tokens}
                logits = self.full.run(["logits"], feed)[0][:, -1]
            still = []
            for row, i in enumerate(active):
                lp = _log_softmax(logits[row])
                banned = banned_ngram_tokens(seqs[i], self.no_repeat_ngram)
                if banned:
                    lp[list(banned)] = -np.inf
                    lp = _log_softmax(lp)
                tok = int(lp.argmax())
                seqs[i].append(tok)
                logp[i] += float(lp[tok])
                if tok != EOS:
                    still.append(i)
            active = still
            if not active:
                break
        return [(s[1:], math.exp(lp / max(1, len(s) - 1))) for s, lp in zip(seqs, logp)]


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def _split_graph(path: str) -> tuple[bytes, bytes, list[str]]:
    """(encoder, decoder, tensors the encoder hands to the decoder) cut from the export.

    The decoder's token-independent work (key/value projections of the encoder states in each
    cross-attention) moves into the encoder graph; the encoder states themselves are still
    passed because the decoder reads their shape.
    """
    import onnx
    from onnx import TensorProto, helper, utils

    m = onnx.load(path)
    # besides the logits, the only rank-3 output is the encoder's last hidden state
    hidden = next(
        o.name
        for o in m.graph.output
        if o.name != "logits" and len(o.type.tensor_type.shape.dim) == 3
    )
    _last_position_head(m)
    full_dec = utils.Extractor(m).extract_model(["token_ids", hidden], ["logits"])
    cached = [hidden, *_token_independent(full_dec.graph, hidden)]
    known = {v.name for v in [*m.graph.value_info, *m.graph.output, *m.graph.input]}
    m.graph.value_info.extend(
        helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in cached if n not in known
    )
    ex = utils.Extractor(m)
    enc = ex.extract_model(["image"], cached).SerializeToString()
    dec = ex.extract_model(["token_ids", *cached], ["logits"]).SerializeToString()
    return enc, dec, cached


def _token_independent(graph, hidden: str) -> list[str]:
    """Heavy tensors computed from the encoder states alone that token-dependent nodes read."""
    tok, hid, heavy = {"token_ids"}, {hidden}, set()
    for n in graph.node:  # topologically sorted
        ins = set(n.input)
        if ins & tok:
            tok.update(n.output)
        elif ins & hid:
            hid.update(n.output)
            if n.op_type in ("MatMul", "Gemm") or ins & heavy:
                heavy.update(n.output)
    out: list[str] = []
    for n in graph.node:
        if set(n.input) & tok:
            out += [i for i in n.input if i in heavy and i not in out]
    return out


def _last_position_head(m) -> None:
    """Feed only the newest position into the LM head (dense -> LN -> 6144-way projection)."""
    from onnx import helper, numpy_helper

    head = next(
        (n for n in m.graph.node if n.name.endswith("/cls/predictions/transform/dense/MatMul")),
        None,
    )
    if head is None:
        return
    src, sliced = head.input[0], head.input[0] + "/last"
    consts = {
        "starts": [-1],
        "ends": [np.iinfo(np.int64).max],
        "axes": [1],
    }
    names = []
    for k, v in consts.items():
        names.append(f"{sliced}/{k}")
        m.graph.initializer.append(numpy_helper.from_array(np.array(v, np.int64), names[-1]))
    idx = list(m.graph.node).index(head)
    m.graph.node.insert(idx, helper.make_node("Slice", [src, *names], [sliced], name=sliced))
    head.input[0] = sliced


@OCRS.register(
    "manga_ocr",
    description="manga-ocr ViT+BERT on onnxruntime (split encoder/decoder, batched greedy). "
    "Japanese, whole-bubble crops.",
    requires=("onnxruntime",),  # `onnx` too for the fast split graph; without it: full graph
    weights=("manga_ocr_onnx", "manga_ocr_vocab"),
)
def _build(settings, **opts):
    """`weights=` loads a fine-tuned export (training/export_ocr_onnx.py), whose vocab sits
    next to it as <name>.vocab.txt unless `vocab=` says otherwise."""
    opts.setdefault("threads", int(os.environ.get("OCR_THREADS", "0")))
    weights, vocab = opts.pop("weights", None), opts.pop("vocab", None)
    if weights and not Path(weights).is_file():
        raise FileNotFoundError(f"ocr.weights: no such ONNX model: {weights}")
    if weights and not vocab:
        vocab = Path(weights).with_suffix(".vocab.txt")
    return MangaOcrOnnx(
        weights or ensure("manga_ocr_onnx", settings),
        vocab or ensure("manga_ocr_vocab", settings),
        **opts,
    )
