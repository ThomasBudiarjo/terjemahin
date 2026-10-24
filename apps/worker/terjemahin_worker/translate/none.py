from __future__ import annotations

from ..opts import warn_unknown
from ..registry import TRANSLATORS
from .base import TranslationItem, TranslationRequest


class OriginalTranslator:
    """Copies the source text through (useful to test OCR + typesetting), or blanks it."""

    vision = False

    def __init__(self, blank: bool = False) -> None:
        self.blank = blank
        self.erase_only = blank  # empty targets are the point: the pipeline still erases

    def translate(self, req: TranslationRequest) -> list[TranslationItem]:
        return [TranslationItem(source=s, target="" if self.blank else s) for s in req.sources]


@TRANSLATORS.register("original", description="Keep the OCR text (debug OCR/typesetting).")
def _original(settings, **extra):
    warn_unknown("translator 'original'", extra)
    return OriginalTranslator()


@TRANSLATORS.register("none", description="Erase text only: inpaint without typesetting.")
def _none(settings, **extra):
    warn_unknown("translator 'none'", extra)
    return OriginalTranslator(blank=True)
