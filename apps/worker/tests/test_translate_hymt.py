import os

import pytest

from terjemahin_worker.config import Settings
from terjemahin_worker.registry import TRANSLATORS, load_all
from terjemahin_worker.translate import hymt

load_all()


def test_registered_and_prompts_follow_model_card():
    assert "hymt" in TRANSLATORS.names()
    p = hymt.build_prompt("坂本くん？", "id", [])
    assert p == "Translate the following segment into Indonesian, without additional explanation.\n坂本くん？"
    assert hymt.build_prompt("え？", "zh", []).startswith("将以下文本翻译为中文")
    ctx = hymt.build_prompt("え？", "id", ["ねえねえ", "坂本くん？"])
    assert ctx.startswith("ねえねえ\n坂本くん？\n参考上面的信息") and ctx.endswith("Indonesian，注意不需要翻译上文，也不要额外解释：\nえ？")


def test_clean_output_strips_quotes_and_extra_lines():
    assert hymt.clean_output(' "Eh?" ') == "Eh?"
    assert hymt.clean_output("Wah, bagus!\n(Note: casual)") == "Wah, bagus!"


def test_bad_options_fail_at_build():
    with pytest.raises(ValueError):
        TRANSLATORS.build("hymt", Settings(), context=-1)


def _cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(hymt.DEFAULT_MODEL, "config.json"), str)
    except (ImportError, OSError, ValueError):
        return False


@pytest.mark.skipif(not _cached() or not os.environ.get("TERJEMAHIN_SLOW_LLM"),
                    reason="needs tencent/HY-MT1.5-1.8B in the HF cache and TERJEMAHIN_SLOW_LLM=1")
def test_real_model_page_lines():
    tr = TRANSLATORS.build("hymt", Settings())
    out = tr.translate_texts(["坂本くん？", "え？", "わあ、いいね！"], "ja", "id")
    assert out[0] == "Sakamoto-kun?" and out[1] == "Eh?"
    assert out[2].startswith("Wah")
