"""local_llm: the in-process chat-model translator, tested with fake models (no weights loaded).

The real model run (Qwen2.5-3B, ~1 min/page on CPU) is opt-in: TERJEMAHIN_SLOW_LLM=1.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any

import numpy as np
import pytest

from terjemahin_worker.config import Settings
from terjemahin_worker.models import BBox, GlossaryEntry, PageContext, TextRegion
from terjemahin_worker.registry import TRANSLATORS, load_all
from terjemahin_worker.translate import local_llm as LL
from terjemahin_worker.translate import textproc as tp
from terjemahin_worker.translate.base import TranslationItem, TranslationRequest
from terjemahin_worker.translate.llm_common import ChatRequest, LLMTranslator

load_all()

SOURCES = ["ねえねえ、めいは坂本のことどう思う？", "坂本くん？", "え？", "わ、私は…別にそんな…"]


def _req(sources: list[str], target: str = "id", **ctx: Any) -> TranslationRequest:
    regions = [TextRegion(id=i, bbox=BBox(x1=10, y1=10 + 50 * i, x2=90, y2=50 + 50 * i))
               for i in range(len(sources))]
    return TranslationRequest(page_bgr=np.zeros((400, 300, 3), np.uint8), regions=regions,
                              sources=sources, context=PageContext(**ctx), target_lang=target)


class FakeBackend(LL.LocalLLMBackend):
    """LocalLLMBackend with `complete` answered from a queue (strings or callables)."""

    def __init__(self, answers: list[Any], **kw: Any) -> None:
        super().__init__(**kw)
        self.answers = list(answers)
        self.requests: list[ChatRequest] = []

    def complete(self, req: ChatRequest) -> str:
        self.requests.append(req)
        a = self.answers.pop(0) if self.answers else ""
        return a(req) if callable(a) else a


def _translator(answers: list[Any], pivot: str | None = None, fallback: Any = None,
                pivot_tr: Any = None) -> tuple[LL.LocalLLMTranslator, FakeBackend]:
    backend = FakeBackend(answers)
    llm = LLMTranslator(backend, mode="text", output="lines", temperature=None, fallback=fallback)
    tr = LL.LocalLLMTranslator(Settings(), llm, backend, fallback, pivot=pivot)
    tr._pivot_tr = pivot_tr
    return tr, backend


def _numbered(req: ChatRequest) -> list[str]:
    return re.findall(r"^<\|(\d+)\|> (.*)$", req.text(), re.MULTILINE)


# --- registry / options ------------------------------------------------------------------------

def test_registered_offline_and_lazy():
    info = TRANSLATORS.get("local_llm")
    assert info.offline and "transformers" in info.requires
    tr = TRANSLATORS.build("local_llm", Settings(), max_new_tokens="300", threads="2", dtype="float32",
                           shots="false", fallback="nllb")
    assert isinstance(tr, LL.LocalLLMTranslator) and not tr.vision
    b = tr.backend
    assert (b.max_new_tokens, b.threads, b.dtype, b.shots, b.repo) == (300, 2, "float32", False, LL.DEFAULT_MODEL)
    assert tr.pivot == "en" and tr.llm.output == "lines"  # defaults
    assert LL._MODELS == {} or all(k[0] != "never-loaded" for k in LL._MODELS)
    with pytest.raises(ValueError):
        TRANSLATORS.build("local_llm", Settings(), mode="vision")
    with pytest.raises(ValueError):
        TRANSLATORS.build("local_llm", Settings(), prepare="bogus")
    assert TRANSLATORS.build("local_llm", Settings(), pivot="none").pivot is None


def test_dtype_names():
    import torch

    assert LL._dtype("bf16") is torch.bfloat16 and LL._dtype("float32") is torch.float32
    with pytest.raises(ValueError):
        LL._dtype("int3")


# --- prompt pieces -----------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["id", "en"])
def test_few_shot_is_one_exchange_in_the_answer_format(target):
    msgs = LL.few_shot(target, "ja", "lines")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    asked = re.findall(r"^<\|(\d+)\|>", msgs[0]["content"], re.MULTILINE)
    answered = re.findall(r"^<\|(\d+)\|> (.+)$", msgs[1]["content"], re.MULTILINE)
    assert asked == [n for n, _ in answered] and len(asked) == 5
    assert not any(tp.has_source_script(t, "ja") for _, t in answered)
    # no first names a small model could copy into an unrelated page
    assert "Kenji" not in msgs[1]["content"]
    assert LL.few_shot("ko", "ja", "lines") == [] and LL.few_shot("id", "zh", "lines") == []
    assert '"items"' in LL.few_shot(target, "ja", "json")[1]["content"]


def test_token_budget_scales_with_the_page_and_is_capped():
    one = LL.token_budget("<|1|> えー、それだけ？", "lines", 1024)
    many = LL.token_budget("\n".join(f"<|{i}|> 反応がいちいち面白いなとか" for i in range(1, 14)), "lines", 1024)
    assert 32 <= one < many <= 1024
    assert LL.token_budget("\n".join(f"<|{i}|> " + "長い" * 50 for i in range(1, 40)), "lines", 512) == 512
    assert LL.token_budget("no numbered lines", "lines", 700) == 700


def test_messages_contain_system_shot_and_task():
    b = LL.LocalLLMBackend(shots=True)
    b.langs = ("ja", "en")
    msgs = b.messages(ChatRequest("SYS", ["<|1|> おい"], None, 100, None))
    assert msgs[0] == {"role": "system", "content": "SYS"} and len(msgs) == 4
    assert msgs[-1]["content"] == "<|1|> おい"


# --- generation (fake tokenizer/model) ---------------------------------------------------------

class _Tok:
    pad_token_id = None
    eos_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "\n".join(m["content"] for m in msgs)

    def __call__(self, text, return_tensors="pt"):
        import torch

        class Enc(dict):
            def to(self, device):
                return self

        return Enc(input_ids=torch.ones((1, 7), dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return "<|1|> Hey!"


class _Model:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, input_ids, **kw):
        import torch

        self.calls.append(kw)
        return torch.cat([input_ids, torch.full((1, 3), 5, dtype=torch.long)], 1)


def test_backend_greedy_budget_and_resample_on_identical_retry(monkeypatch):
    model = _Model()
    monkeypatch.setattr(LL, "load_causal", lambda *a, **k: LL._Loaded(_Tok(), model, threading.Lock()))
    import torch

    before = torch.get_num_threads()
    b = LL.LocalLLMBackend(max_new_tokens=900, shots=False, lookup=8, threads=1)
    req = ChatRequest("sys", ["<|1|> おい！"], None, 4096, None)
    assert b.complete(req) == "<|1|> Hey!"
    first = model.calls[-1]
    assert first["do_sample"] is False and first["prompt_lookup_num_tokens"] == 8
    assert first["max_new_tokens"] == LL.token_budget(req.text(), "lines", 900)
    import torch

    assert torch.get_num_threads() == before  # restored after generation
    b.complete(req)  # the same prompt again (a parse retry): sample instead of repeating itself
    assert model.calls[-1]["do_sample"] is True and "prompt_lookup_num_tokens" not in model.calls[-1]
    assert b.stats[-1]["new_tokens"] == 3 and b.stats[-1]["prompt_tokens"] == 7


# --- page translation --------------------------------------------------------------------------

def test_direct_page_one_generation_with_names_romanised_for_the_model():
    answer = ("<|1|> Hei, hei, Mei, menurutmu Sakamoto gimana?\n<|2|> Sakamoto-kun?\n"
              "<|3|> Eh?\n<|4|> A-aku sih... nggak gimana-gimana...")
    tr, backend = _translator([answer])
    out = tr.translate(_req(SOURCES, glossary=[GlossaryEntry(source="めい", target="Mei")]))
    assert [o.target for o in out] == ["Hei, hei, Mei, menurutmu Sakamoto gimana?", "Sakamoto-kun?",
                                       "Eh?", "A-aku sih... nggak gimana-gimana..."]
    assert len(backend.requests) == 1  # the whole page in one generation
    sent = dict(_numbered(backend.requests[0]))
    assert "Mei" in sent["1"] and "Sakamoto" in sent["1"] and "坂本" not in sent["1"]
    assert sent["2"] == "Sakamoto-kun？"


def test_unusable_items_go_to_the_fallback_per_item():
    calls: list[list[str]] = []

    def fallback(sub: TranslationRequest) -> list[TranslationItem]:
        calls.append(list(sub.sources))
        return [TranslationItem(source=s, target=f"FB:{len(s)}") for s in sub.sources]

    looping = "ya " * 30
    answers = [f"<|1|> Hei, Mei!\n<|2|> {looping}\n<|3|> まだ日本語", "<|3|> まだ日本語"]
    tr, _ = _translator(answers, fallback=fallback)
    out = tr.translate(_req(["ねえ、めい！", "坂本くん？", "別に"]))
    assert out[0].target == "Hei, Mei!"
    assert out[1].target.startswith("FB:") and out[2].target.startswith("FB:")
    assert sorted(len(c) for c in calls) == [1, 1]  # still Japanese after repair, then looping


def test_garbage_answer_falls_back_for_the_whole_page():
    fb = lambda sub: [TranslationItem(source=s, target="fallback") for s in sub.sources]
    tr, backend = _translator(["I cannot help with that.", "???", "", ""], fallback=fb)
    out = tr.translate(_req(["坂本くん？", "別にそんな"]))
    assert [o.target for o in out] == ["fallback", "fallback"]
    assert len(backend.requests) >= 2  # parse retry before giving up


class _Pivot:
    """Stands in for nllb en->id: records the English it gets."""

    def __init__(self) -> None:
        self.got: list[str] = []

    def translate_texts(self, texts, source, target, glossary=()):
        self.got += texts
        table = {"what do you think about Sakamoto?": "bagaimana pendapatmu tentang Sakamoto?",
                 "Mei, what do you think about Sakamoto?": "Mei, bagaimana pendapatmu tentang Sakamoto?",
                 "I don't really...": "Aku tidak benar-benar...",
                 "Then what about Mesia?": "Lalu bagaimana dengan Mesia?"}
        return [table.get(t, t) for t in texts]


def test_pivot_through_english_reattaches_interjections_stutters_and_names():
    answer = ("<|1|> Hey, hey, Mei, what do you think about Sakamoto?\n<|2|> Sakamoto-kun?\n"
              "<|3|> I-I don't really...\n<|4|> Then what about Mesia?")
    piv = _Pivot()
    tr, backend = _translator([answer], pivot="en", pivot_tr=piv)
    out = [o.target for o in tr.translate(_req(SOURCES[:2] + ["わ、私は…別にそんな…", "じゃあメシアはどうなの？"]))]
    # the LLM was asked for English, with the bare interjection え？ left out of the prompt
    assert "English" in backend.requests[0].system or "English" in backend.requests[0].text()
    assert out[0] == "Hei, hei, Mei, bagaimana pendapatmu tentang Sakamoto?"
    assert out[1] == "Sakamoto-kun?"
    assert out[2].startswith("A-aku")
    assert out[3].startswith("Kalau gitu,") and "Mesia" in out[3]
    assert "Hey, hey, Mei, what do you think about Sakamoto?" not in piv.got  # interjection trimmed


def test_pure_interjection_page_needs_no_model():
    tr, backend = _translator([], pivot="en", pivot_tr=_Pivot())
    out = tr.translate(_req(["え？", "うんうん"]))
    assert [o.target for o in out] == ["Eh?", "Iya, iya"] and backend.requests == []


def test_trim_interjections_and_names_helpers():
    assert LL.trim_interjections("Hey, hey, Mei, what's up?", True, False) == "Mei, what's up?"
    assert LL.trim_interjections("Fine, I guess. Yeah.", False, True) == "Fine, I guess."
    assert LL.trim_interjections("Wow!", True, False) == "Wow!"  # nothing left: keep it
    assert LL.pivot_names("Hey, Mei, what about Sakamoto-kun?") == ["Mei", "Sakamoto-kun"]
    assert LL.recase("hei, mei, bagaimana?", ["Mei"]) == "hei, Mei, bagaimana?"
    assert LL.destutter("I-I don't... W-well, s-so what?") == "I don't... well, so what?"
    assert LL.destutter("A well-known, run-down X-ray") == "A well-known, run-down X-ray"
    assert LL.destutter("I… I don't really...", True) == "I don't really..."
    assert LL.destutter("I… I don't really...") == "I… I don't really..."
    assert LL.trim_interjections("Eh, really? That's it?", True, False) == "Really? That's it?"


# --- real model (opt-in) ------------------------------------------------------------------------

def _qwen_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(LL.DEFAULT_MODEL, "config.json"), str)


@pytest.mark.skipif(not os.environ.get("TERJEMAHIN_SLOW_LLM") or not _qwen_cached(),
                    reason="set TERJEMAHIN_SLOW_LLM=1 (needs Qwen2.5-3B-Instruct cached; ~2 min)")
def test_real_model_page():
    tr = TRANSLATORS.build("local_llm", Settings())
    out = [o.target for o in tr.translate(_req(["坂本くん？", "えー、それだけー？", "わ、私は…別にそんな…"]))]
    assert out[0] == "Sakamoto-kun?" and out[1].startswith("Eh,")
    assert all(o and not tp.has_source_script(o, "ja") for o in out)


def test_auto_threads_leave_busy_cores_alone(monkeypatch):
    from terjemahin_worker.translate import hf_seq2seq as S

    monkeypatch.setattr(S.os, "cpu_count", lambda: 4)
    for busy, want in ((2.1, 2), (9.0, 1), (0.0, 4)):
        monkeypatch.setattr(S, "busy_cores_elsewhere", lambda window=0.25, b=busy: b)
        assert LL.auto_threads() == want
    # no /proc/stat: the load average decides
    monkeypatch.setattr(S, "busy_cores_elsewhere", lambda window=0.25: None)
    monkeypatch.setattr(S.os, "getloadavg", lambda: (3.0, 0.0, 0.0))
    assert LL.auto_threads() == 1
    assert LL.LocalLLMBackend(threads=3).num_threads() == 3
    assert TRANSLATORS.build("local_llm", Settings(), threads="auto").backend.threads is None


def test_busy_cores_elsewhere_ignores_this_process():
    from terjemahin_worker.translate import hf_seq2seq as S

    busy = S.busy_cores_elsewhere(0.05)
    assert busy is None or busy >= 0.0


def test_thread_governor_resizes_the_pool_between_steps(monkeypatch):
    import torch

    seen = iter([1, 1, 3])
    monkeypatch.setattr(LL, "auto_threads", lambda window=0.25: next(seen))
    before = torch.get_num_threads()
    try:
        gov = LL._ThreadGovernor(every=2)
        ids = torch.ones((1, 4), dtype=torch.long)
        stops = [bool(gov(ids, None)[0]) for _ in range(6)]
        assert stops == [False] * 6 and torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(before)
    # auto threads: the governor rides along; explicit threads: no governor
    model = _Model()
    monkeypatch.setattr(LL, "load_causal", lambda *a, **k: LL._Loaded(_Tok(), model, threading.Lock()))
    monkeypatch.setattr(LL, "auto_threads", lambda window=0.25: 1)
    LL.LocalLLMBackend(shots=False).complete(ChatRequest("s", ["<|1|> a"], None, 100, None))
    assert isinstance(model.calls[-1]["stopping_criteria"][0], LL._ThreadGovernor)
    LL.LocalLLMBackend(shots=False, threads=1).complete(ChatRequest("s", ["<|1|> b"], None, 100, None))
    assert "stopping_criteria" not in model.calls[-1]
