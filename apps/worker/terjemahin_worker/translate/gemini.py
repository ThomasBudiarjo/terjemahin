"""Gemini through the REST generateContent endpoint, plain httpx (no google-genai SDK).

Uses the shared LLM core: vision mode (page + crops, Gemini reads the text) when OCR is off, text
mode with the page image attached otherwise. JSON is enforced with responseSchema; safety filters
are relaxed because manga routinely contains violence and insults that are fine to translate.
"""

from __future__ import annotations

import logging
from typing import Any

from ..registry import TRANSLATORS
from .llm_common import ChatRequest, Http, Image, LLMError, make_translator, split_opts

log = logging.getLogger(__name__)

API_URL = "https://generativelanguage.googleapis.com/v1beta"
_SAFETY = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


def gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """JSON Schema -> Gemini's OpenAPI subset (upper-case types, nullable, no additionalProperties)."""
    out: dict[str, Any] = {}
    t = schema.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        out["type"] = non_null[0].upper() if non_null else "STRING"
        if "null" in t:
            out["nullable"] = True
    elif t:
        out["type"] = t.upper()
    if "properties" in schema:
        out["properties"] = {k: gemini_schema(v) for k, v in schema["properties"].items()}
    if "items" in schema:
        out["items"] = gemini_schema(schema["items"])
    if "required" in schema:
        out["required"] = list(schema["required"])
    return out


class GeminiBackend:
    def __init__(self, http: Http, api_key: str, model: str, *, base_url: str = API_URL,
                 structured: bool = True) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is empty")
        self.http = http
        self.url = f"{base_url.rstrip('/')}/models/{model}:generateContent"
        self.model = model
        self.name = f"gemini:{model}"
        self.vision = True
        self.headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        self.structured = structured
        self.usage: dict[str, int] = {}

    def body(self, req: ChatRequest) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        for p in req.parts:
            if isinstance(p, Image):
                parts.append({"inlineData": {"mimeType": "image/jpeg", "data": p.b64}})
            else:
                parts.append({"text": p})
        gen: dict[str, Any] = {"maxOutputTokens": req.max_tokens, "responseMimeType": "application/json"}
        if req.temperature is not None:
            gen["temperature"] = req.temperature
        if self.structured:
            gen["responseSchema"] = gemini_schema(req.schema)
        return {
            "systemInstruction": {"parts": [{"text": req.system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": gen,
            "safetySettings": _SAFETY,
        }

    def complete(self, req: ChatRequest) -> str:
        try:
            data = self.http.post(self.url, self.headers, self.body(req))
        except LLMError as e:
            if e.status == 400 and self.structured and "schema" in str(e).lower():
                log.info("%s: responseSchema rejected; using prompt-only JSON", self.name)
                self.structured = False
                data = self.http.post(self.url, self.headers, self.body(req))
            else:
                raise
        block = (data.get("promptFeedback") or {}).get("blockReason")
        if block:
            raise LLMError(f"{self.name}: prompt blocked ({block})")
        for k, v in (data.get("usageMetadata") or {}).items():
            if isinstance(v, int):
                self.usage[k] = self.usage.get(k, 0) + v
        cands = data.get("candidates") or []
        if not cands:
            raise LLMError(f"{self.name}: no candidates: {str(data)[:300]}", retryable=True)
        cand = cands[0]
        text = "".join(
            p.get("text", "") for p in (cand.get("content") or {}).get("parts") or [] if not p.get("thought")
        )
        reason = cand.get("finishReason")
        if not text and reason not in (None, "STOP", "MAX_TOKENS"):
            raise LLMError(f"{self.name}: no answer (finishReason={reason})")
        if reason == "MAX_TOKENS":
            log.warning("%s: answer truncated at maxOutputTokens", self.name)
        return text


@TRANSLATORS.register(
    "gemini",
    description="Gemini (REST): reads + translates every crop with page and series context when "
    "OCR is off, or translates OCR lines with the page image attached.",
    needs_key="gemini_api_key",
    offline=False,
)
def _build(settings, **opts):
    core, http_opts, rest = split_opts(dict(opts))
    core.setdefault("temperature", None)  # Gemini 3 is tuned for its default temperature
    fallback = rest.pop("fallback", None)
    backend = GeminiBackend(
        Http(**http_opts),
        rest.pop("api_key", None) or settings.gemini_api_key,
        rest.pop("model", None) or settings.gemini_model,
        base_url=rest.pop("base_url", None) or API_URL,
        structured=bool(rest.pop("structured", True)),
    )
    if rest:
        log.warning("gemini: ignoring unknown options %s", sorted(rest))
    return make_translator(backend, settings, core, fallback)
