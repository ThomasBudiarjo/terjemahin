from __future__ import annotations

from ..registry import TRANSLATORS
from .base import TranslationItem, TranslationRequest


class OriginalTranslator:
    """Copies the source text through (useful to test OCR + typesetting), or blanks it."""

    vision = False

    def __init__(self, blank: bool = False) -> None:
        self.blank = blank

    def translate(self, req: TranslationRequest) -> list[TranslationItem]:
        return [TranslationItem(source=s, target="" if self.blank else s) for s in req.sources]


@TRANSLATORS.register("original", description="Keep the OCR text (debug OCR/typesetting).")
def _original(settings, **_):
    return OriginalTranslator()


@TRANSLATORS.register("none", description="Erase text only: inpaint without typesetting.")
def _none(settings, **_):
    return OriginalTranslator(blank=True)
