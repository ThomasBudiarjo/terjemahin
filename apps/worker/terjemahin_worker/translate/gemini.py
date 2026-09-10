from __future__ import annotations

import cv2
import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel

from ..models import Bubble, PageContext, TextRegion
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
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is empty")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def translate(
        self,
        page_bgr: np.ndarray,
        regions: list[TextRegion],
        context: PageContext,
    ) -> list[Bubble]:
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

        bubbles: list[Bubble] = []
        for r in regions:
            it = by_index.get(r.id)
            bubbles.append(
                Bubble(
                    region=r,
                    source_text=(it.source if it else "").strip(),
                    target_text=(it.target if it else "").strip(),
                    speaker=it.speaker if it else None,
                )
            )
        return bubbles
