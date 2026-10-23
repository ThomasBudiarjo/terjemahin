"""LLM translators against httpx.MockTransport fakes of each provider's real wire format."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import numpy as np
import pytest

from terjemahin_worker.config import Settings
from terjemahin_worker.models import BBox, GlossaryEntry, PageContext, TextRegion
from terjemahin_worker.translate import TRANSLATORS, TranslationItem, TranslationRequest
from terjemahin_worker.translate.llm_common import LLMError, parse_items
from terjemahin_worker.translate.prompt import system_prompt

SOURCES = ["ねえねえ、めいは坂本のことどう思う？", "坂本くん？", "えー、それだけー？"]
TARGETS = ["Hei, hei, Mei, menurutmu Sakamoto gimana?", "Sakamoto-kun?", "Eh, cuma itu?"]


def _page() -> np.ndarray:
    img = np.full((400, 300, 3), 255, np.uint8)
    img[40:120, 200:260] = 30
    return img


def _regions(n: int = 3) -> list[TextRegion]:
    return [TextRegion(id=i, bbox=BBox(x1=200 - 60 * i, y1=40, x2=260 - 60 * i, y2=120)) for i in range(n)]


def _req(sources: list[str] | None = None, target: str = "id", **ctx: Any) -> TranslationRequest:
    sources = SOURCES if sources is None else sources
    return TranslationRequest(
        page_bgr=_page(), regions=_regions(len(sources)), sources=sources,
        context=PageContext(**ctx), target_lang=target,
    )


def _items_json(targets: list[str], start: int = 1, sources: list[str] | None = None) -> str:
    sources = sources or [""] * len(targets)
    return json.dumps({"items": [
        {"index": start + i, "source": s, "target": t, "speaker": None}
        for i, (s, t) in enumerate(zip(sources, targets))
    ]}, ensure_ascii=False)


def _openai_response(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-4o-mini",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
    }


class Fake:
    """Records requests; answers with a queue of (status, json, headers) or a callable."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        ans = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if callable(ans):
            ans = ans(request)
        status, body, headers = ans if isinstance(ans, tuple) else (200, ans, {})
        return httpx.Response(status, json=body, headers=headers)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


def _settings(**kw: Any) -> Settings:
    base = {"openai_api_key": "sk-test", "anthropic_api_key": "ak-test", "gemini_api_key": "gk-test",
                "deepseek_api_key": "dk-test"}
    base.update(kw)
    return Settings(**base)


def _build(name: str, fake: Fake, **opts: Any):
    sleeps: list[float] = []
    tr = TRANSLATORS.build(name, _settings(), transport=fake.transport(), sleep=sleeps.append, **opts)
    tr._sleeps = sleeps
    return tr


# --- providers -------------------------------------------------------------------------------

def test_openai_text_mode_request_and_parse():
    fake = Fake(_openai_response(_items_json(TARGETS)))
    tr = _build("openai", fake, page_image=False)
    items = tr.translate(_req(glossary=[GlossaryEntry(source="坂本", target="Sakamoto")]))

    assert [i.target for i in items] == TARGETS
    assert [i.source for i in items] == SOURCES  # text mode keeps OCR text
    req = fake.requests[0]
    assert str(req.url) == "https://api.openai.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = fake.bodies[0]
    assert body["model"] == "gpt-4o-mini"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    system, user = body["messages"]
    assert system["role"] == "system" and "JSON" in system["content"]
    assert isinstance(user["content"], str)  # no images -> plain string content
    assert "<|1|> ねえねえ、めいは坂本のことどう思う？" in user["content"]
    assert "<|3|> えー、それだけー？" in user["content"]
    assert "坂本 -> Sakamoto" in user["content"]
    assert "TEPAT 3 item" in user["content"]


def test_openai_vision_mode_payload_contains_page_and_crops():
    answer = _items_json(TARGETS, sources=SOURCES)
    fake = Fake(_openai_response(answer))
    tr = _build("openai", fake)
    assert tr.vision
    items = tr.translate(_req(["", "", ""]))  # OCR skipped -> auto picks vision

    assert [i.source for i in items] == SOURCES  # model read the crops
    assert [i.target for i in items] == TARGETS
    content = fake.bodies[0]["messages"][1]["content"]
    images = [c for c in content if c["type"] == "image_url"]
    texts = [c["text"] for c in content if c["type"] == "text"]
    assert len(images) == 1 + 3  # page + one crop per region
    for im in images:
        url = im["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(url.split(",", 1)[1])[:2] == b"\xff\xd8"  # real JPEG
    assert any("Crop balon #1:" == t for t in texts) and any("Crop balon #3:" == t for t in texts)
    assert "crop balon 1..3" in texts[0]


def test_openai_text_mode_with_page_image_sends_positions():
    fake = Fake(_openai_response(_items_json(TARGETS)))
    tr = _build("openai", fake)  # vision model -> page image attached for context
    tr.translate(_req())
    content = fake.bodies[0]["messages"][1]["content"]
    assert [c["type"] for c in content] == ["text", "image_url"]
    assert "<|1|> (77%,20%)" in content[0]["text"]


def test_deepseek_and_ollama_formats():
    fake = Fake(_openai_response("```json\n" + _items_json(TARGETS) + "\n```"))
    tr = _build("deepseek", fake)
    assert not tr.vision
    assert [i.target for i in tr.translate(_req())] == TARGETS
    assert str(fake.requests[0].url) == "https://api.deepseek.com/v1/chat/completions"
    assert fake.bodies[0]["response_format"] == {"type": "json_object"}

    fake = Fake(_openai_response(_items_json(TARGETS)))
    info = TRANSLATORS.get("ollama")
    assert info.needs_key is None and info.offline
    tr = TRANSLATORS.build("ollama", Settings(), transport=fake.transport(), model="qwen2.5:3b")
    assert [i.target for i in tr.translate(_req())] == TARGETS
    req = fake.requests[0]
    assert str(req.url) == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in req.headers
    assert fake.bodies[0]["model"] == "qwen2.5:3b"


def test_openai_provider_preset_and_base_url():
    fake = Fake(_openai_response(_items_json(TARGETS)))
    _build("openai", fake, provider="openrouter", model="google/gemma-3-27b-it").translate(_req())
    assert str(fake.requests[0].url) == "https://openrouter.ai/api/v1/chat/completions"
    fake = Fake(_openai_response(_items_json(TARGETS)))
    _build("openai", fake, base_url="http://127.0.0.1:1234/v1/").translate(_req())
    assert str(fake.requests[0].url) == "http://127.0.0.1:1234/v1/chat/completions"


def _anthropic_response(text: str, stop: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
        "content": [{"type": "thinking", "thinking": "", "signature": "x"}, {"type": "text", "text": text}],
        "stop_reason": stop, "stop_details": None,
        "usage": {"input_tokens": 900, "output_tokens": 80},
    }


def test_anthropic_vision_request_and_response():
    fake = Fake(_anthropic_response(_items_json(TARGETS, sources=SOURCES)))
    tr = _build("anthropic", fake)
    items = tr.translate(_req(["", "", ""]))
    assert [i.target for i in items] == TARGETS and [i.source for i in items] == SOURCES

    req = fake.requests[0]
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "ak-test"
    assert req.headers["anthropic-version"] == "2026-10-21"
    assert req.headers["anthropic-beta"] == "server-side-fallback-2026-07-01"
    body = fake.bodies[0]
    assert body["model"] == "claude-sonnet-5-5"
    assert "temperature" not in body and "top_p" not in body
    assert body["fallbacks"] == "default"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert isinstance(body["system"], str) and "JSON" in body["system"]
    blocks = body["messages"][0]["content"]
    imgs = [b for b in blocks if b["type"] == "image"]
    assert len(imgs) == 4
    assert imgs[0]["source"]["type"] == "base64" and imgs[0]["source"]["media_type"] == "image/jpeg"


def test_anthropic_refusal_and_rejected_fields():
    fake = Fake(_anthropic_response("", stop="refusal"))
    with pytest.raises(LLMError, match="refused"):
        _build("anthropic", fake, parse_retries=0).translate(_req())

    # A proxy that does not know the fallback beta or structured outputs: both dropped once.
    def proxy(request: httpx.Request):
        body = json.loads(request.content)
        if "fallbacks" in body:
            return 400, {"type": "error", "error": {"type": "invalid_request_error",
                                                    "message": "fallbacks: Extra inputs are not permitted"}}, {}
        if "output_config" in body:
            return 400, {"type": "error", "error": {"type": "invalid_request_error",
                                                    "message": "output_config: unknown field"}}, {}
        return 200, _anthropic_response(_items_json(TARGETS)), {}

    fake = Fake(proxy)
    items = _build("anthropic", fake).translate(_req())
    assert [i.target for i in items] == TARGETS
    assert len(fake.requests) == 3


def _gemini_response(text: str) -> dict[str, Any]:
    return {
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "thinking...", "thought": True}, {"text": text}]},
            "finishReason": "STOP", "index": 0,
        }],
        "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 90, "totalTokenCount": 1090},
        "modelVersion": "gemini-3.8-flash",
    }


def test_gemini_rest_request_and_response():
    fake = Fake(_gemini_response(_items_json(TARGETS, sources=SOURCES)))
    tr = _build("gemini", fake)
    items = tr.translate(_req(["", "", ""]))
    assert [i.target for i in items] == TARGETS and [i.source for i in items] == SOURCES

    req = fake.requests[0]
    assert str(req.url).endswith("/v1beta/models/gemini-3.8-flash:generateContent")
    assert req.headers["x-goog-api-key"] == "gk-test"
    body = fake.bodies[0]
    assert "JSON" in body["systemInstruction"]["parts"][0]["text"]
    parts = body["contents"][0]["parts"]
    assert sum("inlineData" in p for p in parts) == 4
    assert parts[1]["inlineData"]["mimeType"] == "image/jpeg"
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    schema = gen["responseSchema"]["properties"]["items"]["items"]
    assert schema["properties"]["speaker"] == {"type": "STRING", "nullable": True}
    assert "additionalProperties" not in json.dumps(gen["responseSchema"])
    assert {s["threshold"] for s in body["safetySettings"]} == {"BLOCK_NONE"}


def test_gemini_blocked_prompt_raises():
    fake = Fake({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}})
    with pytest.raises(LLMError, match="blocked"):
        _build("gemini", fake).translate(_req())


# --- robustness ------------------------------------------------------------------------------

@pytest.mark.parametrize("answer", [
    ('Here you go:\n```json\n{"items": [{"index": 1, "target": "A"}, {"index": 2, "target": "B"},'
    ' {"index": 3, "target": "C"},]}\n```\nHope it helps!'),
    ('<think>The user wants JSON.</think>{"translations": [{"id": 1, "translation": "A"},'
    ' {"id": 2, "translation": "B"}, {"id": 3, "translation": "C"}]}'),
    '[{"index": 0, "target": "A"}, {"index": 1, "target": "B"}, {"index": 2, "target": "C"}]',
    '{"1": "A", "2": "B", "3": "C"}',
    '["A", "B", "C"]',
    "<|1|> A\n<|2|> B\n<|3|> C",
    "1. A\n2) B\n3: C",
    '{"items": [{"index": 1, "target": "A"}, {"index": 2, "target": "B"}, {"index": 3, "target": "C"',
])
def test_parse_tolerates_malformed_answers(answer: str):
    got = parse_items(answer, [1, 2, 3])
    if answer.endswith('"C"'):  # truncated: the complete objects are salvaged, 3 gets repaired
        assert {k: v.target for k, v in got.items()} == {1: "A", 2: "B"}
        return
    assert {k: v.target for k, v in got.items()} == {1: "A", 2: "B", 3: "C"}


def test_salvages_items_from_bad_brackets():
    # Real Qwen2.5-1.5B output on page01: the array is closed with '}}' instead of ']}'.
    answer = ('{"items": [{"index": 1, "source": "坂本くん？", "target": "Sakamoto-kun?", "speaker": null}, '
              '{"index": 2, "source": "え？", "target": "Eh?", "speaker": null}}')
    got = parse_items(answer, [1, 2])
    assert {k: v.target for k, v in got.items()} == {1: "Sakamoto-kun?", 2: "Eh?"}
    assert got[1].source == "坂本くん？"


def test_numbered_lines_fallback_multiline_and_unexpected_index():
    got = parse_items("<|1|> Halo,\napa kabar?\n<|2|> Baik.\n<|9|> ngawur", [1, 2])
    assert {k: v.target for k, v in got.items()} == {1: "Halo, apa kabar?", 2: "Baik."}


def test_count_mismatch_is_repaired_with_explicit_indices():
    first = _openai_response(_items_json(TARGETS[:1]))  # model skipped 2 and 3
    second = _openai_response(_items_json(TARGETS[1:], start=2))
    fake = Fake(first, second)
    items = _build("openai", fake, page_image=False).translate(_req())
    assert [i.target for i in items] == TARGETS
    assert len(fake.requests) == 2
    repair = fake.bodies[1]["messages"][1]["content"]
    assert "jawaban sebelumnya tidak lengkap" in repair and "2, 3" in repair


def test_vision_repair_sends_only_missing_crops():
    first = _openai_response(_items_json(TARGETS[:2], sources=SOURCES[:2]))
    second = _openai_response(json.dumps({"items": [{"index": 3, "source": SOURCES[2], "target": TARGETS[2]}]}))
    fake = Fake(first, second)
    items = _build("openai", fake).translate(_req(["", "", ""]))
    assert [i.target for i in items] == TARGETS
    content = fake.bodies[1]["messages"][1]["content"]
    assert sum(c["type"] == "image_url" for c in content) == 2  # page + crop #3
    assert [c["text"] for c in content if c["type"] == "text"][1] == "Crop balon #3:"


def test_echoed_source_counts_as_missing():
    first = _openai_response(_items_json([TARGETS[0], "坂本くん？", TARGETS[2]]))
    second = _openai_response(_items_json([TARGETS[1]], start=2))
    fake = Fake(first, second)
    items = _build("openai", fake, page_image=False).translate(_req())
    assert [i.target for i in items] == TARGETS


def test_retries_429_and_5xx_with_backoff():
    ok = _openai_response(_items_json(TARGETS))
    fake = Fake(
        (429, {"error": {"message": "Rate limit reached", "type": "requests"}}, {"retry-after": "2"}),
        (503, {"error": {"message": "overloaded"}}, {}),
        ok,
    )
    tr = _build("openai", fake, page_image=False)
    assert [i.target for i in tr.translate(_req())] == TARGETS
    assert len(fake.requests) == 3
    assert tr._sleeps[0] == 2.0  # honours Retry-After
    assert 0 < tr._sleeps[1] < 10


def test_timeout_then_success():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=_anthropic_response(_items_json(TARGETS)))

    tr = TRANSLATORS.build("anthropic", _settings(), transport=httpx.MockTransport(handler), sleep=lambda s: None)
    assert [i.target for i in tr.translate(_req())] == TARGETS
    assert calls["n"] == 2


def test_auth_error_is_not_retried():
    fake = Fake((401, {"error": {"message": "Incorrect API key provided"}}, {}))
    with pytest.raises(LLMError, match="401"):
        _build("openai", fake).translate(_req())
    assert len(fake.requests) == 1


def test_rejected_response_format_is_dropped():
    def server(request: httpx.Request):
        body = json.loads(request.content)
        if "response_format" in body:
            return 400, {"error": {"message": "'response_format' of type 'json_schema' is not supported"}}, {}
        return 200, _openai_response("<|1|> A\n<|2|> B\n<|3|> C"), {}

    fake = Fake(server)
    items = _build("openai", fake, page_image=False).translate(_req())
    assert [i.target for i in items] == ["A", "B", "C"]
    assert len(fake.requests) == 2


def test_never_fewer_items_and_fallback_translator():
    fake = Fake(_openai_response("Sorry, I can't help with that."))
    items = _build("openai", fake, page_image=False).translate(_req())
    assert len(items) == 3 and all(i.target == "" for i in items)
    assert [i.source for i in items] == SOURCES

    fake = Fake(_openai_response(_items_json([TARGETS[0]])))
    tr = _build("openai", fake, page_image=False, fallback="original")
    items = tr.translate(_req())
    assert len(items) == 3
    assert items[0].target == TARGETS[0]
    assert [i.target for i in items[1:]] == SOURCES[1:]  # 'original' fallback copies the source


def test_empty_sources_skipped_in_text_mode():
    fake = Fake(_openai_response(_items_json([TARGETS[0], TARGETS[2]])))
    items = _build("deepseek", fake).translate(_req([SOURCES[0], "", SOURCES[2]]))
    assert [i.target for i in items] == [TARGETS[0], "", TARGETS[2]]
    user = fake.bodies[0]["messages"][1]["content"]
    assert "<|2|> えー" in user and "<|3|>" not in user  # renumbered densely


def test_chunking_by_max_items():
    def server(request: httpx.Request):
        user = json.loads(request.content)["messages"][1]["content"]
        n = user.count("<|")
        return 200, _openai_response(_items_json([f"T{i}" for i in range(1, n + 1)])), {}

    fake = Fake(server)
    srcs = [f"テキスト{i}" for i in range(5)]
    items = _build("openai", fake, page_image=False, max_items=2).translate(_req(srcs))
    assert len(fake.requests) == 3
    assert [i.target for i in items] == ["T1", "T2", "T1", "T2", "T1"]


def test_mode_option_and_vision_capability():
    with pytest.raises(ValueError, match="vision"):
        TRANSLATORS.build("deepseek", _settings(), mode="vision")
    fake = Fake(_openai_response(_items_json(TARGETS, sources=SOURCES)))
    tr = _build("openai", fake, mode="vision")
    tr.translate(_req())  # sources given but vision forced: crops are sent
    assert sum(c["type"] == "image_url" for c in fake.bodies[0]["messages"][1]["content"]) == 4
    assert tr.resolve_mode(_req()) == "vision"
    assert _build("openai", Fake({})).resolve_mode(_req()) == "text"


def test_target_language_prompt():
    assert "penerjemah manga profesional" in system_prompt("text", "ja", "id")
    en = system_prompt("text", "ja", "en")
    assert "into English" in en and "penerjemah" not in en
    fr = system_prompt("vision", "ja", "fr")
    assert "into French" in fr
    fake = Fake(_openai_response(_items_json(["Hey, hey", "Sakamoto-kun?", "Huh, that's all?"])))
    _build("openai", fake, page_image=False).translate(_req(target="en"))
    assert "Translate the following 3 lines from Japanese into English" in fake.bodies[0]["messages"][1]["content"]


def test_previous_pages_context_and_memory():
    fake = Fake(_openai_response(_items_json(TARGETS)))
    tr = _build("openai", fake, page_image=False, remember_pages=2)
    tr.translate(_req(previous_pages=[["Kemarin hujan."]]))
    assert "Kemarin hujan." in fake.bodies[0]["messages"][1]["content"]
    tr.translate(_req())  # no explicit context: the translator's own memory of the last page
    assert TARGETS[0] in fake.bodies[1]["messages"][1]["content"]


def test_registered_methods():
    for name in ("openai", "deepseek", "anthropic", "gemini"):
        info = TRANSLATORS.get(name)
        assert info.needs_key and not info.offline and info.requires == ()
        ok, reason = info.status(Settings(**{info.needs_key: ""}))
        assert not ok and "not set" in reason
    assert TRANSLATORS.get("ollama").status(Settings())[0]


def test_no_regions_makes_no_request():
    fake = Fake({})
    assert _build("openai", fake).translate(_req([])) == []
    assert fake.requests == []
    assert isinstance(TranslationItem(), TranslationItem)


def test_wrong_language_answer_is_retried_with_language_note():
    # Real Qwen2.5-1.5B behaviour on page01: fluent English when Indonesian was asked.
    english = ["Oh, what do you think about Sakamoto?", "Sakamoto?", "Oh, just that? I don't know"]
    fake = Fake(_openai_response(_items_json(english)), _openai_response(_items_json(TARGETS)))
    items = _build("openai", fake, page_image=False).translate(_req())
    assert [i.target for i in items] == TARGETS
    assert len(fake.requests) == 2
    assert "bukan bahasa Inggris" in fake.bodies[1]["messages"][1]["content"]


def test_wrong_language_check():
    from terjemahin_worker.translate.llm_common import wrong_language

    assert wrong_language(["What do you think about him?", "I'm sorry, I don't know"], "id")
    assert not wrong_language(TARGETS, "id")
    assert not wrong_language(["Hey, hey, what do you think about Sakamoto?"], "en")
    assert not wrong_language(["Sakamoto-kun?"], "id")  # too short to judge
