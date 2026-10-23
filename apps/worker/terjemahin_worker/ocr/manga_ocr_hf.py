"""manga-ocr through transformers (torch): the reference implementation of the ONNX method.

Same crops, preprocessing, decoding rule and post-processing as `manga_ocr`, so the two agree
token for token. Useful on GPU, for beam search (`beams=4` is what the manga_ocr package does)
and as a check on the ONNX port.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..registry import OCRS
from .manga_ocr_onnx import EOS, START, MangaOcrBase, load_vocab

REPO = "kha-white/manga-ocr-base"


class MangaOcrHF(MangaOcrBase):
    def __init__(self, repo: str = REPO, device: str = "cpu", beams: int = 1, **opts) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import VisionEncoderDecoderModel

        super().__init__(**opts)
        self.torch = torch
        self.model = VisionEncoderDecoderModel.from_pretrained(repo).to(device).eval()
        local = Path(repo) / "vocab.txt"  # a training/train_ocr.py checkpoint dir
        self.vocab = load_vocab(local if local.exists() else hf_hub_download(repo, "vocab.txt"))
        self.device = device
        self.beams = beams
        self.gen_kwargs = {
            "max_length": self.max_len + 1,
            "num_beams": beams,
            "no_repeat_ngram_size": self.no_repeat_ngram,
            "do_sample": False,
            "decoder_start_token_id": START,
            "eos_token_id": EOS,
            "pad_token_id": 0,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if beams > 1:  # the manga_ocr package's settings (from the model config)
            self.gen_kwargs.update(early_stopping=True, length_penalty=2.0)

    def generate(self, pixels: np.ndarray) -> list[tuple[list[int], float]]:
        torch = self.torch
        with torch.inference_mode():
            out = self.model.generate(torch.from_numpy(pixels).to(self.device), **self.gen_kwargs)
            beam_idx = out.beam_indices if self.beams > 1 else None
            scores = self.model.compute_transition_scores(
                out.sequences, out.scores, beam_idx, normalize_logits=self.beams == 1
            )
        gen = out.sequences[:, 1:][:, : scores.shape[1]].cpu()
        is_eos = (gen == EOS).int()
        keep = (torch.cumsum(is_eos, 1) - is_eos) == 0  # up to and including the first EOS
        n = keep.sum(1).clamp(min=1)
        confs = (torch.where(keep, scores.cpu(), 0.0).sum(1) / n).exp().tolist()
        return [(row[k].tolist(), c) for row, k, c in zip(gen, keep, confs)]


@OCRS.register(
    "manga_ocr_hf",
    description="manga-ocr via transformers/torch (kha-white/manga-ocr-base); supports beam search.",
    requires=("torch", "transformers"),
)
def _build(settings, **opts):
    opts.setdefault("device", settings.device)
    return MangaOcrHF(**opts)
