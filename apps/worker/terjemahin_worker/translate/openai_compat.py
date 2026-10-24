"""OpenAI Chat Completions protocol: OpenAI, OpenRouter, Groq, DeepSeek, Ollama, LM Studio, vLLM.

Plain httpx, no SDK. `provider=` presets fill in the base URL of common hosts; `base_url=` always
wins. Structured output degrades gracefully: json_schema -> json_object -> prompt-only, dropped
automatically when an endpoint answers 400 to it. Images are only sent by default to hosts whose
default models see (OpenAI, OpenRouter); a 400 about image input switches to text-only requests.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..registry import TRANSLATORS
from .llm_common import (
    ChatRequest,
    Http,
    Image,
    ImagesRejected,
    LLMError,
    as_bool,
    images_rejected,
    make_translator,
    split_opts,
)

log = logging.getLogger(__name__)

PRESETS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "together": "https://api.together.xyz/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "vllm": "http://localhost:8000/v1",
}
_OPTIONAL = ("response_format", "temperature", "max_tokens", "max_completion_tokens", "top_p")
_VISION_HOSTS = ("api.openai.com", "openrouter.ai")
LOCAL_TIMEOUT = 900.0  # a CPU-only 3B-7B model needs minutes for one manga page


class OpenAICompatBackend:
    def __init__(
        self,
        http: Http,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        name: str = "openai",
        vision: bool = True,
        json_mode: str = "auto",
        headers: dict[str, str] | None = None,
        image_detail: str | None = None,
    ) -> None:
        if json_mode not in ("auto", "schema", "object", "none"):
            raise ValueError("json_mode must be auto|schema|object|none")
        self.http = http
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.name = f"{name}:{model}"
        self.vision = vision
        if json_mode == "auto":
            json_mode = "schema" if "api.openai.com" in base_url or "openrouter.ai" in base_url else "object"
        self.json_mode = json_mode
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.image_detail = image_detail
        self.usage: dict[str, int] = {}

    def _content(self, req: ChatRequest) -> str | list[dict[str, Any]]:
        if not req.has_images:  # plain string: some servers (DeepSeek, older vLLM) reject arrays
            return req.text()
        out: list[dict[str, Any]] = []
        for p in req.parts:
            if isinstance(p, Image):
                url: dict[str, Any] = {"url": f"data:image/jpeg;base64,{p.b64}"}
                if self.image_detail:
                    url["detail"] = self.image_detail
                out.append({"type": "image_url", "image_url": url})
            else:
                out.append({"type": "text", "text": p})
        return out

    def body(self, req: ChatRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": self._content(req)},
            ],
            "max_tokens": req.max_tokens,
        }
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if self.json_mode == "schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "manga_translation", "strict": True, "schema": req.schema},
            }
        elif self.json_mode == "object":
            body["response_format"] = {"type": "json_object"}
        return body

    def complete(self, req: ChatRequest) -> str:
        try:
            data = self.http.post_adaptive(self.url, self.headers, self.body(req), _OPTIONAL)
        except LLMError as e:
            if req.has_images and images_rejected(e):
                raise ImagesRejected(str(e), e.status) from e
            raise
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"{self.name}: response has no choices: {str(data)[:300]}", retryable=True)
        choice = choices[0]
        msg = choice.get("message") or {}
        if msg.get("refusal"):
            raise LLMError(f"{self.name} refused: {msg['refusal']}")
        if choice.get("finish_reason") == "length":
            log.warning("%s: answer truncated at max_tokens", self.name)
        for k, v in (data.get("usage") or {}).items():
            if isinstance(v, int):
                self.usage[k] = self.usage.get(k, 0) + v
        content = msg.get("content")
        if isinstance(content, list):  # some proxies return content parts
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        return content or ""


def _is_local(url: str) -> bool:
    return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"))


def _build(
    settings: Any,
    opts: dict[str, Any],
    *,
    name: str,
    base_url: str,
    model: str,
    api_key: str,
    vision: bool | None,
    json_mode: str = "auto",
    http_defaults: dict[str, Any] | None = None,
) -> Any:
    core, http_opts, rest = split_opts(dict(opts))
    if http_defaults is None and _is_local(rest.get("base_url") or base_url):
        http_defaults = {"timeout": LOCAL_TIMEOUT, "timeout_retries": 0}
    http_opts = {**(http_defaults or {}), **http_opts}
    preset = rest.pop("provider", None)
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"unknown provider '{preset}', choose from: {', '.join(PRESETS)}")
        base_url = PRESETS[preset]
    base_url = rest.pop("base_url", None) or base_url
    if not str(base_url).lower().startswith(("http://", "https://")):
        raise ValueError(f"translator base_url must start with http:// or https://, not {base_url!r}")
    model = rest.pop("model", None) or model
    api_key = rest.pop("api_key", None) or api_key
    vision = rest.pop("vision", vision)
    vision = any(h in base_url for h in _VISION_HOSTS) if vision is None else as_bool(vision)
    fallback = rest.pop("fallback", None)
    backend = OpenAICompatBackend(
        Http(**http_opts), base_url, model, api_key,
        name=name, vision=vision, json_mode=rest.pop("json_mode", json_mode),
        headers=rest.pop("headers", None), image_detail=rest.pop("image_detail", None),
    )
    if rest:
        log.warning("%s: ignoring unknown options %s", name, sorted(rest))
    return make_translator(backend, settings, core, fallback)


@TRANSLATORS.register(
    "openai",
    description="Any OpenAI-compatible chat API (OpenAI, OpenRouter, Groq, LM Studio, vLLM...): "
    "numbered-line JSON protocol, vision with page + crops when OCR is off.",
    needs_key="openai_api_key",
    offline=False,
)
def _openai(settings, **opts):
    return _build(
        settings, opts, name="openai", base_url=settings.openai_base_url,
        model=settings.openai_model, api_key=settings.openai_api_key, vision=None,
    )


@TRANSLATORS.register(
    "deepseek",
    description="DeepSeek chat API (text only, JSON mode).",
    needs_key="deepseek_api_key",
    offline=False,
)
def _deepseek(settings, **opts):
    return _build(
        settings, opts, name="deepseek",
        base_url=getattr(settings, "deepseek_base_url", "") or PRESETS["deepseek"],
        model=settings.deepseek_model, api_key=settings.deepseek_api_key, vision=False,
        json_mode="object",
    )


@TRANSLATORS.register(
    "ollama",
    description="Local LLM through Ollama's OpenAI-compatible endpoint (no key; vision=true for "
    "VLMs such as qwen2.5vl / gemma3; 900s timeout for CPU inference).",
    offline=True,
)
def _ollama(settings, **opts):
    return _build(
        settings, opts, name="ollama",
        base_url=getattr(settings, "ollama_base_url", "") or os.environ.get("OLLAMA_BASE_URL")
        or PRESETS["ollama"],
        model=getattr(settings, "ollama_model", "") or os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b",
        api_key="", vision=False, json_mode="object",
        http_defaults={"timeout": float(os.environ.get("OLLAMA_TIMEOUT") or LOCAL_TIMEOUT),
                       "timeout_retries": 0},
    )
