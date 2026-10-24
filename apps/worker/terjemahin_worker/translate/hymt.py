"""Offline translation with Tencent HY-MT1.5 (a 1.8B causal LM trained only for translation).

It reuses the seq2seq translator's manga pipeline (textproc.prepare -> generate -> checks ->
retry -> textproc.finish), so glossary terms, romanised names, honorifics, interjections, the
とか hedge and stutters behave exactly as for nllb/m2m100. What differs is generation:

- the model's own prompt templates (README of tencent/HY-MT1.5-1.8B); the first pass uses the
  "contextual translation" template with the preceding bubbles of the page, so subjects and
  pronouns follow the conversation instead of being guessed bubble by bubble;
- context sometimes makes the model echo a name from it in Japanese ("坂本さん itu baik"); the
  shared post-check flags leftover source script and the retry runs WITHOUT context;
- greedy decoding, batched with left padding.

Measured (4 CPU cores, bfloat16): ~3.6 GB on disk, ~4 GB RSS, 12 s for page01's 13 bubbles.
Strict harness: page01 9/13 adequate (chrF 37.4), held-out lines 23/36 (chrF 42.9), vs m2m100
10/13 (32.4) and 23/36 (40.0): more fluent, not yet more faithful. Without context: 10/13, 21/36.
Licence: Tencent HY Community License (not valid in the EU, UK and South Korea).
"""

from __future__ import annotations

import logging
from typing import Any

from ..opts import as_int
from ..registry import TRANSLATORS
from . import textproc as tp
from .hf_seq2seq import Seq2SeqTranslator, _batches, _num_threads
from .local_llm import load_causal

log = logging.getLogger(__name__)

DEFAULT_MODEL = "tencent/HY-MT1.5-1.8B"

LANG_NAMES = {
    "id": "Indonesian", "en": "English", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "zh-hant": "Traditional Chinese", "ms": "Malay", "th": "Thai", "vi": "Vietnamese",
    "tl": "Filipino", "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
    "it": "Italian", "ru": "Russian", "ar": "Arabic", "tr": "Turkish", "hi": "Hindi",
    "pl": "Polish", "nl": "Dutch", "uk": "Ukrainian", "cs": "Czech", "he": "Hebrew",
}
_ZH_NAMES = {"zh": "中文", "zh-hant": "繁体中文"}


def build_prompt(text: str, target: str, context: list[str]) -> str:
    """HY-MT's documented templates: Chinese wording when Chinese is involved, else English;
    the contextual one when there are preceding lines."""
    lang = tp.iso(target)
    if context:
        name = _ZH_NAMES.get(lang) or LANG_NAMES.get(lang, lang)
        return ("\n".join(context) + f"\n参考上面的信息，把下面的文本翻译成{name}，"
                "注意不需要翻译上文，也不要额外解释：\n" + text)
    if lang in _ZH_NAMES:
        return f"将以下文本翻译为{_ZH_NAMES[lang]}，注意只需要输出翻译后的结果，不要额外解释：\n{text}"
    return (f"Translate the following segment into {LANG_NAMES.get(lang, lang)}, "
            f"without additional explanation.\n{text}")


def clean_output(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'「『“":
        s = s[1:-1].strip()
    return s.split("\n")[0].strip() if "\n" in s else s


class HyMtTranslator(Seq2SeqTranslator):
    """Japanese bubbles -> target with HY-MT1.5, the preceding bubbles as context."""

    FAMILIES = ("hymt",)

    def __init__(
        self,
        repo: str = DEFAULT_MODEL,
        device: str = "cpu",
        context: int = 2,
        dtype: str = "bfloat16",
        batch_size: int = 4,
        **opts: Any,
    ) -> None:
        opts.setdefault("num_beams", 1)
        opts.setdefault("no_repeat_ngram_size", 0)
        opts.setdefault("repetition_penalty", 1.05)
        super().__init__(repo, "hymt", device=device, batch_size=batch_size, **opts)
        self.context = as_int(context, "translator context", 0, 16)
        self.dtype = dtype

    def code(self, lang: str) -> str:
        return tp.iso(lang)

    def _loaded(self):  # type: ignore[override]
        return load_causal(self.repo, self.device, self.dtype)

    def _generate(self, texts: list[str], source: str, target: str, **overrides: Any) -> list[str]:
        """`texts` arrive in page order, so each one's context is the lines before it. A retry
        (any decoding override) runs without context, which is what fixes echoed source names."""
        if not texts:
            return []
        import torch

        m = self._loaded()
        tok = m.tokenizer
        n_ctx = 0 if overrides else self.context
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": build_prompt(t, target, texts[max(0, i - n_ctx):i])}],
                add_generation_prompt=True, tokenize=False,
            )
            for i, t in enumerate(texts)
        ]
        penalty = max(self.repetition_penalty, float(overrides.get("repetition_penalty", 0)))
        order = sorted(range(len(texts)), key=lambda i: len(prompts[i]))
        out = [""] * len(texts)
        n_threads = self._auto() if self.threads is None else None
        side = tok.padding_side
        with m.lock, torch.inference_mode(), _num_threads(n_threads):
            tok.padding_side = "left"
            try:
                for idx in _batches(order, self.batch_size):
                    enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                              add_special_tokens=False).to(self.device)
                    n_src = max(len(tok(texts[i], add_special_tokens=False)["input_ids"]) for i in idx)
                    gen = m.model.generate(
                        **enc,
                        max_new_tokens=min(self.max_tokens, int(n_src * self.max_ratio) + 16),
                        do_sample=False, top_p=None, top_k=None, temperature=None,
                        repetition_penalty=penalty,
                        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                    )
                    new = gen[:, enc["input_ids"].shape[1]:]
                    for i, s in zip(idx, tok.batch_decode(new, skip_special_tokens=True)):
                        out[i] = clean_output(s)
            finally:
                tok.padding_side = side
        return out


@TRANSLATORS.register(
    "hymt",
    description=("Tencent HY-MT1.5 1.8B (offline, ~3.6GB, ~20s/page CPU): dedicated translation "
                 "LLM, previous bubbles as context; best offline chrF, adequacy on par with "
                 "m2m100 (page01 9/13, held-out 23/36). Tencent HY licence (not EU/UK/KR)."),
    requires=("torch", "transformers"),
)
def _build(settings, model: str | None = None, device: str | None = None, **opts: Any):
    return HyMtTranslator(model or DEFAULT_MODEL, device=device or settings.device, **opts)
