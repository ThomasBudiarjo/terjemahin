"""Pivot translation: pipe bubbles through several translators / languages.

ja->id has little direct parallel data, while ja->en and en->id are among the best-covered
pairs, so e.g. m2m100 (ja->en) then nllb (en->id) can beat a single direct hop.
"""

from __future__ import annotations

import json
from typing import Any

from ..registry import TRANSLATORS
from . import textproc as tp
from .base import TranslationItem, TranslationRequest


def parse_steps(steps: Any) -> list[dict[str, Any]]:
    """Accepts [{"translator": "m2m100", "to": "en", "opts": {...}}, ...], its JSON string,
    or the shorthand "m2m100:en,nllb:id" (the last step's language defaults to the target)."""
    if isinstance(steps, str):
        s = steps.strip()
        if s.startswith("["):
            steps = json.loads(s)
        else:
            steps = [
                dict(zip(("translator", "to"), part.split(":", 1)))
                for part in s.split(",") if part.strip()
            ]
    out: list[dict[str, Any]] = []
    for step in steps or []:
        step = dict(step)
        name = step.pop("translator", None) or step.pop("name", None)
        if not name:
            raise ValueError(f"chain step without 'translator': {step}")
        if name == "chain":
            raise ValueError("a chain cannot contain itself")
        to = step.pop("to", None)
        opts = {**step.pop("opts", {}), **step}
        out.append({"translator": str(name).strip(), "to": to and str(to).strip(), "opts": opts})
    if not out:
        raise ValueError("chain needs at least one step")
    return out


class ChainTranslator:
    """Interjections, stutters and glossary are resolved once against the *final* language
    (textproc.prepare); only the remaining core text travels through the pivot hops, so
    "え？" stays "Eh?" instead of becoming "Huh?" -> "Apa itu?"."""

    def __init__(self, settings, steps: Any, prepare: bool = True, casual: bool = True) -> None:
        self.steps = parse_steps(steps)
        self.prepare, self.casual = prepare, casual
        self.translators = [
            TRANSLATORS.build(s["translator"], settings, **s["opts"]) for s in self.steps
        ]
        self.vision = bool(getattr(self.translators[0], "vision", False))

    def _hops(self, source: str, target: str) -> list[tuple[str, str]]:
        hops, src = [], source
        for i, s in enumerate(self.steps):
            dst = s["to"] or (target if i == len(self.steps) - 1 else None)
            if dst is None:
                raise ValueError(f"chain step {i} ({s['translator']}) needs 'to'")
            hops.append((src, dst))
            src = dst
        if src != target:
            raise ValueError(f"chain ends in '{src}' but the target language is '{target}'")
        return hops

    def translate(self, req: TranslationRequest) -> list[TranslationItem]:
        source = tp.detect_lang(req.sources) if req.source_lang == "auto" else req.source_lang
        preps = None
        if self.prepare and not self.vision:
            preps = [
                tp.prepare(s, source, req.target_lang, req.context.glossary) for s in req.sources
            ]
        items: list[TranslationItem] = []
        texts = [p.text for p in preps] if preps else list(req.sources)
        for tr, (src, dst) in zip(self.translators, self._hops(source, req.target_lang)):
            step_req = TranslationRequest(
                page_bgr=req.page_bgr, regions=req.regions, sources=texts,
                context=req.context, source_lang=src, target_lang=dst,
            )
            step_items = tr.translate(step_req)
            if not items:  # the first hop owns the source text (vision translators read it)
                items = [
                    TranslationItem(source=it.source or s, target=it.target, speaker=it.speaker)
                    for it, s in zip(step_items, req.sources)
                ]
            texts = [it.target for it in step_items]
        if preps:
            texts = [tp.finish(p, t, req.target_lang, self.casual) for p, t in zip(preps, texts)]
        return [it.model_copy(update={"target": t}) for it, t in zip(items, texts)]

    def translate_texts(self, texts: list[str], source: str, target: str, glossary=()) -> list[str]:
        """Text-only entry point (same as the seq2seq translators) for eval and the API."""
        import numpy as np

        from ..models import PageContext

        req = TranslationRequest(
            page_bgr=np.zeros((1, 1, 3), np.uint8), regions=[], sources=list(texts),
            context=PageContext(glossary=list(glossary)), source_lang=source, target_lang=target,
        )
        return [it.target for it in self.translate(req)]


@TRANSLATORS.register(
    "chain",
    description="Pivot through several translators, e.g. steps='m2m100:en,nllb:id'.",
)
def _build(settings, steps: Any = "m2m100:en,nllb:id", prepare: bool = True, casual: bool = True, **_):
    return ChainTranslator(settings, steps, prepare, casual)
