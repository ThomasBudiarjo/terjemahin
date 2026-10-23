from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel

from ..registry import TRANSLATORS
from .base import TranslationItem, TranslationRequest
from .prompt import SYSTEM_PROMPT, build_user_prompt

PAGE_MAX_SIDE = 1536
CROP_PAD = 8


class _Item(BaseModel):
    index: int
    source: str
    target: str
    speaker: str | None = None


class _Response(BaseModel):
    items: list[_Item]


def _jpeg(image_bgr: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("failed to encode image")
    return buf.tobytes()


def _downscale(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return image_bgr
    return cv2.resize(image_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


class GeminiTranslator:
    """Vision LLM: reads each crop and translates it in one call, with page + series context."""

    vision = True

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        if not api_key:
            raise ValueError("GEMINI_API_KEY is empty")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def translate(self, req: TranslationRequest) -> list[TranslationItem]:
        from google.genai import types

        page_bgr, regions, context = req.page_bgr, req.regions, req.context
        if not regions:
            return []

        H, W = page_bgr.shape[:2]
        contents: list[types.Part | str] = [
            build_user_prompt(context, len(regions)),
            types.Part.from_bytes(data=_jpeg(_downscale(page_bgr, PAGE_MAX_SIDE)), mime_type="image/jpeg"),
        ]
        for r in regions:
            b = r.bbox.pad(CROP_PAD).clamp(W, H)
            crop = page_bgr[b.y1 : b.y2, b.x1 : b.x2]
            contents.append(f"Crop balon #{r.id}:")
            contents.append(types.Part.from_bytes(data=_jpeg(crop, 92), mime_type="image/jpeg"))

        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_Response,
                temperature=0.4,
            ),
        )
        parsed = _Response.model_validate_json(resp.text or "{}")
        by_index = {it.index: it for it in parsed.items}

        items: list[TranslationItem] = []
        for i, r in enumerate(regions):
            it = by_index.get(r.id)
            items.append(
                TranslationItem(
                    source=(it.source if it else req.sources[i] if i < len(req.sources) else "").strip(),
                    target=(it.target if it else "").strip(),
                    speaker=it.speaker if it else None,
                )
            )
        return items


@TRANSLATORS.register(
    "gemini",
    description="Gemini vision: reads + translates every crop with page and series context.",
    requires=("google.genai",),
    needs_key="gemini_api_key",
    offline=False,
)
def _build(settings, model: str | None = None, **_):
    return GeminiTranslator(settings.gemini_api_key, model or settings.gemini_model)
