"""Claude through the Messages API (POST /v1/messages), plain httpx.

API version 2026-10-21: sampling parameters (temperature/top_p) are no longer accepted, so none
are sent; depth is steered with `output_config.effort` instead. The JSON answer is enforced with
structured outputs (`output_config.format`), and refusals on models that support it are rescued
by the server-side fallback (`fallbacks: "default"`). Both are dropped automatically if an
endpoint (proxy, older gateway) rejects them.
"""

from __future__ import annotations

import logging
from typing import Any

from ..registry import TRANSLATORS
from .llm_common import ChatRequest, Http, Image, LLMError, as_bool, make_translator, split_opts

log = logging.getLogger(__name__)

API_URL = "https://api.anthropic.com/v1"
API_VERSION = "2026-10-21"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")


class AnthropicBackend:
    def __init__(
        self,
        http: Http,
        api_key: str,
        model: str,
        *,
        base_url: str = API_URL,
        version: str = API_VERSION,
        structured: bool = True,
        effort: str | None = None,
        fallbacks: bool | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is empty")
        self.http = http
        self.url = base_url.rstrip("/") + "/messages"
        self.model = model
        self.name = f"anthropic:{model}"
        self.vision = True
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": version,
            "content-type": "application/json",
        }
        self.structured = structured
        self.effort = effort
        if fallbacks is None:
            fallbacks = model in _FALLBACK_MODELS and "api.anthropic.com" in base_url
        self.fallbacks = fallbacks
        self.usage: dict[str, int] = {}

    def body(self, req: ChatRequest) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for p in req.parts:
            if isinstance(p, Image):
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": p.b64},
                })
            else:
                content.append({"type": "text", "text": p})
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": req.max_tokens,
            "system": req.system,
            "messages": [{"role": "user", "content": content}],
        }
        output_config: dict[str, Any] = {}
        if self.structured:
            output_config["format"] = {"type": "json_schema", "schema": req.schema}
        if self.effort:
            output_config["effort"] = self.effort
        if output_config:
            body["output_config"] = output_config
        if self.fallbacks:
            body["fallbacks"] = "default"
        return body

    def _headers(self) -> dict[str, str]:
        if self.fallbacks:
            return {**self.headers, "anthropic-beta": FALLBACK_BETA}
        return self.headers

    def complete(self, req: ChatRequest) -> str:
        for _ in range(3):
            try:
                data = self.http.post(self.url, self._headers(), self.body(req))
                break
            except LLMError as e:
                msg = str(e).lower()
                if e.status != 400:
                    raise
                if self.fallbacks and "fallback" in msg:
                    log.info("%s: server-side fallback rejected; disabling it", self.name)
                    self.fallbacks = False
                elif self.structured and ("output_config" in msg or "format" in msg or "schema" in msg):
                    log.info("%s: structured output rejected; using prompt-only JSON", self.name)
                    self.structured = False
                else:
                    raise
        else:
            raise LLMError(f"{self.name}: request rejected")
        for k, v in (data.get("usage") or {}).items():
            if isinstance(v, int):
                self.usage[k] = self.usage.get(k, 0) + v
        stop = data.get("stop_reason")
        if stop == "refusal":
            details = data.get("stop_details") or {}
            raise LLMError(f"{self.name} refused ({details.get('category')}): {details.get('explanation')}")
        if stop == "max_tokens":
            log.warning("%s: answer truncated at max_tokens", self.name)
        return "".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text")


@TRANSLATORS.register(
    "anthropic",
    description="Claude (Messages API): text mode with page image for context, or full vision "
    "(reads the crops) when OCR is off; structured JSON output.",
    needs_key="anthropic_api_key",
    offline=False,
)
def _build(settings, **opts):
    core, http_opts, rest = split_opts(dict(opts))
    core.setdefault("temperature", None)  # not accepted by API version 2026-10-21
    fallback = rest.pop("fallback", None)
    fb_models = rest.pop("fallbacks", None)
    backend = AnthropicBackend(
        Http(**http_opts),
        rest.pop("api_key", None) or settings.anthropic_api_key,
        rest.pop("model", None) or settings.anthropic_model,
        base_url=rest.pop("base_url", None) or getattr(settings, "anthropic_base_url", "") or API_URL,
        version=rest.pop("version", API_VERSION),
        structured=as_bool(rest.pop("structured", True)),
        effort=rest.pop("effort", None),
        fallbacks=None if fb_models is None else as_bool(fb_models),
    )
    if rest:
        log.warning("anthropic: ignoring unknown options %s", sorted(rest))
    return make_translator(backend, settings, core, fallback)
