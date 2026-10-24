"""Offline LLM translation with a local transformers chat model (default Qwen2.5-3B-Instruct).

The whole page goes to the model in ONE generation, so it sees every bubble of the conversation
at once (who answers whom, what "それだけ" refers to). Everything protocol-related is shared with
the API translators through llm_common.LLMTranslator: system prompt, glossary and previous-page
context, tolerant parsing, the repair request for missing / still-Japanese items and the
fallback translator. What is local-specific lives here:

- the answer format defaults to `<|n|> translation` lines instead of JSON items (half the
  output tokens; on CPU the page time is dominated by decoding);
- a short few-shot exchange (own example lines, not from any eval page) pins the format and
  the casual register for small models;
- names are made unambiguous before the model sees them: glossary terms and common kanji
  surnames are substituted in romaji (textproc), so a 3B model cannot leave 坂本 in kanji;
- greedy decoding, with a token budget derived from the source length so a looping model stops
  early; a repeated identical prompt (parse retry) samples instead of reproducing the failure;
- per-item fallback to a seq2seq translator (nllb / m2m100) for items that stay missing, empty,
  in Japanese, looping or absurdly long.

Memory / speed on 4 CPU cores (AMX bf16): Qwen2.5-3B in bfloat16 ~6.5 GB RSS; see `methods`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from ..models import PageContext
from ..registry import TRANSLATORS
from . import prompt as P
from . import textproc as tp
from .base import TranslationItem, TranslationRequest
from .llm_common import ChatRequest, LLMTranslator, as_bool, as_int, split_opts

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


@dataclass
class _Loaded:
    tokenizer: Any
    model: Any
    lock: threading.Lock


_MODELS: dict[tuple[str, str, str], _Loaded] = {}
_MODELS_LOCK = threading.Lock()


def _dtype(name: str) -> Any:
    import torch

    name = (name or "bfloat16").strip().lower()
    table = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp32": torch.float32,
             "float32": torch.float32, "fp16": torch.float16, "float16": torch.float16}
    if name == "auto":
        return "auto"
    if name not in table:
        raise ValueError(f"dtype must be bfloat16|float32|float16|auto, got {name!r}")
    return table[name]


def load_causal(repo: str, device: str = "cpu", dtype: str = "bfloat16") -> _Loaded:
    """Load (or reuse) a chat model; the local HF cache is tried before the network."""
    key = (repo, device, str(dtype))
    with _MODELS_LOCK:
        if key in _MODELS:
            return _MODELS[key]
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        def _from(local: bool) -> tuple[Any, Any]:
            tok = AutoTokenizer.from_pretrained(repo, local_files_only=local)
            mdl = AutoModelForCausalLM.from_pretrained(
                repo, local_files_only=local, dtype=_dtype(dtype), low_cpu_mem_usage=True
            )
            return tok, mdl

        t = time.perf_counter()
        try:
            tok, mdl = _from(True)
        except OSError:
            log.info("downloading %s from the Hugging Face hub", repo)
            tok, mdl = _from(False)
        mdl = mdl.to(device).eval()
        mdl.generation_config.max_length = None  # we always pass max_new_tokens
        log.info("local_llm: loaded %s (%s) in %.1fs", repo, dtype, time.perf_counter() - t)
        _MODELS[key] = _Loaded(tok, mdl, threading.Lock())
        return _MODELS[key]


# --- few-shot example (own lines, deliberately unrelated to the eval pages) -------------------

_SHOT_JA = ["おい、ケンジ！遅刻するぞー！", "わ、わかってるって…", "また夜更かししたんでしょ？",
            "べ、別にいいだろ！", "ふーん。佐藤先生に言っちゃおうかなー"]
_SHOTS = {
    "id": ["Hei, Kenji! Nanti telat lho!", "A-aku tahu kok...", "Kamu begadang lagi, kan?",
           "B-bukan urusanmu!", "Hmm. Kuaduin ke Sato-sensei, ah."],
    "en": ["Hey, Kenji! You're gonna be late!", "I-I know...", "You stayed up late again, didn't you?",
           "S-so what!", "Hmm. Maybe I'll tell Sato-sensei."],
}
_NUMBERED_LINE = re.compile(r"^<\|\d+\|>\s?(?:\([^)]*\)\s)?(.*)$", re.MULTILINE)


def few_shot(target: str, source: str, output: str) -> list[dict[str, str]]:
    """One user/assistant exchange in the exact request/answer format, or [] if none fits."""
    tgt = tp.iso(target)
    if tgt not in _SHOTS or tp.iso(source) != "ja":
        return []
    lines = [P.Line(i + 1, t) for i, t in enumerate(_SHOT_JA)]
    user = P.text_task(lines, target, source, "lines" if output == "lines" else "json")
    if output == "lines":
        answer = "\n".join(f"<|{i + 1}|> {t}" for i, t in enumerate(_SHOTS[tgt]))
    else:
        import json

        answer = json.dumps({"items": [{"index": i + 1, "source": s, "target": t, "speaker": None}
                                       for i, (s, t) in enumerate(zip(_SHOT_JA, _SHOTS[tgt]))]},
                            ensure_ascii=False)
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def token_budget(user_text: str, output: str, cap: int) -> int:
    """Generous output budget from the numbered source lines: a looping model stops early."""
    lines = _NUMBERED_LINE.findall(user_text)
    if not lines:
        return cap
    # "<|n|> " vs a JSON object that also echoes the source (its characters counted again)
    per_item = 10 if output == "lines" else 40
    chars = sum(len(s) for s in lines) * (3 if output == "lines" else 5)
    return max(32, min(cap, 48 + per_item * len(lines) + chars))


class LocalLLMBackend:
    """llm_common Backend running a transformers causal LM in-process (text only)."""

    vision = False

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "cpu",
        dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
        threads: int | None = None,
        output: str = "lines",
        shots: bool = True,
        repetition_penalty: float = 1.0,
        source_lang: str = "ja",
        target_lang: str = "id",
    ) -> None:
        self.name = "local_llm"
        self.repo, self.device, self.dtype = model, device, dtype
        self.max_new_tokens = int(max_new_tokens)
        self.threads = threads
        self.output = output
        self.shots = shots
        self.repetition_penalty = float(repetition_penalty)
        self.langs = (source_lang, target_lang)  # set per request by the translator
        self._last_prompt = ""
        self.stats: list[dict[str, float]] = []  # per generation: prompt/new tokens, seconds

    def loaded(self) -> _Loaded:
        if self.threads:
            import torch

            torch.set_num_threads(int(self.threads))
        return load_causal(self.repo, self.device, self.dtype)

    def messages(self, req: ChatRequest) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": req.system}]
        if self.shots:
            msgs += few_shot(self.langs[1], self.langs[0], self.output)
        msgs.append({"role": "user", "content": req.text()})
        return msgs

    def complete(self, req: ChatRequest) -> str:
        import torch

        m = self.loaded()
        tok = m.tokenizer
        msgs = self.messages(req)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        digest = hashlib.sha1(text.encode()).hexdigest()
        resample = digest == self._last_prompt  # same prompt again: greedy would repeat itself
        self._last_prompt = digest
        enc = tok(text, return_tensors="pt").to(self.device)
        budget = token_budget(req.text(), self.output, min(self.max_new_tokens, req.max_tokens))
        params: dict[str, Any] = {"max_new_tokens": budget, "repetition_penalty": self.repetition_penalty}
        if resample:
            params.update(do_sample=True, temperature=0.7, top_p=0.9)
        else:
            params.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        t = time.perf_counter()
        with m.lock, torch.inference_mode():
            out = m.model.generate(**enc, pad_token_id=tok.pad_token_id or tok.eos_token_id, **params)
        n_in = enc["input_ids"].shape[1]
        new = out[0, n_in:]
        dt = time.perf_counter() - t
        self.stats.append({"prompt_tokens": n_in, "new_tokens": int(new.shape[0]), "seconds": dt})
        log.info("local_llm: %d prompt + %d new tokens in %.1fs (%.1f tok/s)%s",
                 n_in, new.shape[0], dt, new.shape[0] / max(dt, 1e-6), " [sampled]" if resample else "")
        return tok.decode(new, skip_special_tokens=True)


class LocalLLMTranslator:
    """LLMTranslator over LocalLLMBackend, with name pre-processing and a per-item safety net.

    `pivot="en"`: the LLM translates the page into English (what small chat models write best)
    and a seq2seq model (`pivot_translator`, NLLB by default: its en->id is far better than its
    ja->id) takes it to the target. Interjections, stutters and glossary are then resolved once
    against the final language by textproc.prepare/finish, as in the chain translator.
    """

    vision = False

    def __init__(
        self, settings: Any, llm: LLMTranslator, backend: LocalLLMBackend, fallback: Any,
        names: bool = True, pivot: str | None = None, pivot_translator: str = "nllb",
        prepare: str = "hybrid", casual: bool = True,
    ) -> None:
        self.settings, self.llm, self.backend, self.fallback = settings, llm, backend, fallback
        self.names, self.pivot, self.pivot_name = names, pivot, pivot_translator
        self.prepare_pivot, self.casual = prepare, casual
        self._pivot_tr: Any = None

    def prepare(self, text: str, source: str, context: PageContext) -> str:
        """Glossary terms and common surnames in romaji, honorifics attached ('Sakamoto-kun')."""
        if not text.strip() or tp.iso(source) != "ja":
            return text
        out, _ = tp.apply_glossary(text, context.glossary)
        if self.names:
            out, _ = tp.romanize_surnames(out)
            out = tp.map_honorifics(out)
        return out

    def _llm_page(self, req: TranslationRequest, sources: list[str], source: str, target: str) -> list[str]:
        """One LLM generation for the page (plus repair), then the per-item safety net."""
        self.backend.langs = (source, target)
        sub = TranslationRequest(
            page_bgr=req.page_bgr, regions=req.regions, sources=sources, context=req.context,
            source_lang=source, target_lang=target,
        )
        targets = [it.target.strip() for it in self.llm.translate(sub)]
        bad = [
            i for i, (p, t) in enumerate(zip(sources, targets))
            if p.strip() and (not t or tp.is_bad(p, t, target))
        ]
        if bad and self.fallback is not None:
            log.warning("local_llm: %d item(s) unusable (%s); using the fallback translator",
                        len(bad), [i + 1 for i in bad])
            fb_req = TranslationRequest(
                page_bgr=req.page_bgr, regions=[req.regions[i] for i in bad],
                sources=[sources[i] for i in bad], context=req.context,
                source_lang=source, target_lang=target,
            )
            for i, it in zip(bad, self.fallback(fb_req)):
                if it.target.strip():
                    targets[i] = it.target.strip()
        return targets

    def translate(self, req: TranslationRequest) -> list[TranslationItem]:
        source = tp.detect_lang(req.sources) if req.source_lang == "auto" else req.source_lang
        if not self.pivot or tp.iso(self.pivot) == tp.iso(req.target_lang):
            prepared = [self.prepare(s, source, req.context) for s in req.sources]
            targets = self._llm_page(req, prepared, source, req.target_lang)
            return [TranslationItem(source=s, target=t) for s, t in zip(req.sources, targets)]

        pivot = tp.iso(self.pivot)
        preps = [tp.prepare(s, source, req.target_lang, req.context.glossary, names=self.names)
                 for s in req.sources]
        if self.prepare_pivot == "strip":
            # the LLM only sees the core text; interjections come from textproc's table
            sources = [p.text for p in preps]
        else:
            # "hybrid": the LLM sees the whole line, interjections included (わあ、いいね is
            # "wow, nice", not "yeah"), but they are still rendered from textproc's table (the
            # scanlation convention: "Eh", "Fufu"), so the English ones are trimmed off below
            sources = [self.prepare(tp.strip_stutter(tp.normalize_ja(s)) if tp.iso(source) == "ja"
                                    else s, source, req.context) if p.text else ""
                       for s, p in zip(req.sources, preps)]
        mid = self._llm_page(req, sources, source, pivot)
        log.debug("local_llm: %s pivot %s", pivot, mid)
        names = [pivot_names(t) for t in mid]  # romanised by the LLM ("Hey, Mei, ...")
        if self.prepare_pivot != "strip" and pivot == "en":
            mid = [trim_interjections(t, bool(p.lead), bool(p.trail)) for t, p in zip(mid, preps)]
        if self._pivot_tr is None:
            from ..registry import TRANSLATORS

            self._pivot_tr = TRANSLATORS.build(self.pivot_name, self.settings)
        todo = [i for i, t in enumerate(mid) if t]
        out = [""] * len(mid)
        done = self._pivot_tr.translate_texts([mid[i] for i in todo], pivot, req.target_lang,
                                              req.context.glossary)
        for i, t in zip(todo, done):
            out[i] = t
        # the seq2seq hop sometimes copies English it does not know ("Yummy"): such lines go to
        # the direct fallback, which keeps them only if it also answers differently (names)
        copied = [i for i in todo if out[i] and _letters(out[i]) == _letters(mid[i])
                  and tp.has_source_script(req.sources[i], source)]
        if copied and self.fallback is not None:
            fb_req = TranslationRequest(
                page_bgr=req.page_bgr, regions=[req.regions[i] for i in copied],
                sources=[req.sources[i] for i in copied], context=req.context,
                source_lang=source, target_lang=req.target_lang,
            )
            for i, it in zip(copied, self.fallback(fb_req)):
                if it.target.strip() and _letters(it.target) != _letters(out[i]):
                    log.info("local_llm: pivot hop copied %r; using direct %r", out[i], it.target)
                    out[i], preps[i] = it.target, tp.Prepared(text="", end=preps[i].end)
        out = [tp.finish(p, t, req.target_lang, self.casual) for p, t in zip(preps, out)]
        out = [recase(t, ns) for t, ns in zip(out, names)]
        return [TranslationItem(source=s, target=t) for s, t in zip(req.sources, out)]


_EN_INTERJ = (r"(?:hey|hei|eh|huh|wow|whoa|woah|oh|ah+|um+|uh+|er+m?|hm+|yay|yeah|yes|yep|no|nope|"
              r"well|so|look|geez|jeez|damn|ugh|hehe+|haha+|ahaha+|ehehe+|fufu+|hoho+|heh|right|"
              r"okay|ok|alright|sorry|thanks|wait|what|why|i see|uh-huh|mm+|ooh|aah)")
# only interjections set off by punctuation count ("Hey, Mei" but not "what else")
_LEAD_INTERJ = re.compile(rf"^(?:{_EN_INTERJ}(?:[\s,]+{_EN_INTERJ})*[,.!?…~-]+\s*)+", re.IGNORECASE)
_TRAIL_INTERJ = re.compile(rf"(?:[,.!?…~-]+\s*{_EN_INTERJ})+([.!?…]*)$", re.IGNORECASE)


def trim_interjections(text: str, lead: bool, trail: bool) -> str:
    """Drop the English interjections the LLM wrote for the ones textproc re-attaches."""
    out = text
    if lead:
        out = _LEAD_INTERJ.sub("", out)
    if trail:
        out = _TRAIL_INTERJ.sub(r"\1", out)
    out = out.strip(" ,")
    return out if re.search(r"[^\W\d_]", out) else text


_MID_CAP = re.compile(r"(?<=[^\W\d_][,;]\s)([A-Z][a-z]+(?:-[a-z]+)?)\b|(?<=[a-z]\s)([A-Z][a-z]+(?:-[a-z]+)?)\b")
_NOT_NAMES = frozenset(["I", "I'm", "I'll", "I've", "I'd", "OK", "Okay", "Mom", "Dad", "Mr", "Mrs", "Ms", "Miss", "Sir", "God"])


def pivot_names(text: str) -> list[str]:
    """Capitalised words inside an English sentence ('Hey, Mei, what...'): names to protect."""
    out = []
    for m in _MID_CAP.finditer(text):
        w = m.group(1) or m.group(2)
        if w not in _NOT_NAMES and w not in out:
            out.append(w)
    return out


def recase(text: str, names: list[str]) -> str:
    """finish() lower-cases the first word after a re-attached interjection; names keep case."""
    for n in names:
        text = re.sub(rf"(?<![^\W\d_]){re.escape(n.lower())}(?![^\W\d_])", n, text)
    return text


def _letters(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalpha())


_LOCAL_OPTS = {
    "model": str, "dtype": str, "max_new_tokens": as_int, "threads": lambda v: as_int(v) if v else None,
    "shots": as_bool, "repetition_penalty": float, "device": str,
}


@TRANSLATORS.register(
    "local_llm",
    description="Local chat LLM via transformers (offline; default Qwen/Qwen2.5-3B-Instruct bf16, "
    "~6.5GB RAM, ~1 min/page on 4 CPU cores): whole page in one generation, glossary + previous "
    "pages, per-item fallback to nllb/m2m100. Opts: model, dtype, max_new_tokens, threads.",
    requires=("torch", "transformers"),
    offline=True,
)
def _build(settings, **opts: Any) -> LocalLLMTranslator:
    opts = dict(opts)
    local = {k: conv(opts.pop(k)) for k, conv in _LOCAL_OPTS.items() if k in opts}
    fallback_name = opts.pop("fallback", "m2m100")
    names = as_bool(opts.pop("names", True))
    pivot = opts.pop("pivot", None)
    pivot = None if pivot is None or str(pivot).lower() in ("", "none", "null") else str(pivot)
    pivot_translator = str(opts.pop("pivot_translator", "nllb"))
    prepare = str(opts.pop("prepare", "hybrid")).strip().lower()
    if prepare not in ("hybrid", "strip"):
        raise ValueError(f"local_llm: prepare must be hybrid|strip, got {prepare!r}")
    core, _http, rest = split_opts(opts)
    if rest:
        log.warning("local_llm: ignoring unknown options %s", sorted(rest))
    core.setdefault("output", "lines")
    core.setdefault("mode", "text")
    core.setdefault("temperature", None)
    if core["mode"] != "text":
        raise ValueError("local_llm is text-only (mode=text)")
    backend = LocalLLMBackend(
        model=local.get("model") or DEFAULT_MODEL,
        device=local.get("device") or settings.device,
        dtype=local.get("dtype", "bfloat16"),
        max_new_tokens=local.get("max_new_tokens", 1024),
        threads=local.get("threads"),
        output=core["output"],
        shots=local.get("shots", True),
        repetition_penalty=local.get("repetition_penalty", 1.0),
    )
    core.setdefault("max_tokens", backend.max_new_tokens)

    fb = None
    if fallback_name and str(fallback_name).lower() not in ("none", "null", ""):
        holder: list[Any] = []

        def fb(sub: TranslationRequest) -> list[TranslationItem]:
            if not holder:
                holder.append(TRANSLATORS.build(str(fallback_name), settings))
            return holder[0].translate(sub)

    llm = LLMTranslator(backend, fallback=fb, **core)
    return LocalLLMTranslator(settings, llm, backend, fb, names, pivot, pivot_translator, prepare)
