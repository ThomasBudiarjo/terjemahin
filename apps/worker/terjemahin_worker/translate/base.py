from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from ..models import PageContext, TextRegion


@dataclass
class TranslationRequest:
    page_bgr: np.ndarray
    regions: list[TextRegion]
    sources: list[str]  # OCR output, one per region ("" when OCR was skipped)
    context: PageContext = field(default_factory=PageContext)
    source_lang: str = "ja"
    target_lang: str = "id"


class TranslationItem(BaseModel):
    source: str = ""  # vision translators fill this with what they read
    target: str = ""
    speaker: str | None = None


class Translator(Protocol):
    vision: bool
    """True if the translator reads text from the image itself (works with ocr='none')."""

    def translate(self, req: TranslationRequest) -> list[TranslationItem]:
        """Exactly one item per region, in the same order."""
        ...
