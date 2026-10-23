import json
import re
from pathlib import Path

import pytest

from terjemahin_worker.config import Settings
from terjemahin_worker.models import GlossaryEntry, PageContext
from terjemahin_worker.registry import TRANSLATORS, load_all
from terjemahin_worker.translate import chain as chain_mod
from terjemahin_worker.translate import textproc as tp
from terjemahin_worker.translate.base import TranslationItem, TranslationRequest
from terjemahin_worker.translate.chain import ChainTranslator, parse_steps
from terjemahin_worker.translate.hf_seq2seq import Seq2SeqTranslator

GT = Path(__file__).resolve().parent.parent / "eval" / "fixtures" / "page01.gt.json"
GLOSSARY = [
    GlossaryEntry(source="坂本", target="Sakamoto"),
    GlossaryEntry(source="めい", target="Mei"),
    GlossaryEntry(source="メシア", target="Mesia"),
]


def _cached(repo: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return any(
        isinstance(try_to_load_from_cache(repo, f), str)
        for f in ("model.safetensors", "pytorch_model.bin")
    )


NLLB, M2M = "facebook/nllb-200-distilled-600M", "facebook/m2m100_418M"
needs_nllb = pytest.mark.skipif(not _cached(NLLB), reason="NLLB not in the HF cache")
needs_both = pytest.mark.skipif(not (_cached(NLLB) and _cached(M2M)), reason="models not cached")


# --- textproc -------------------------------------------------------------------------------

def test_language_codes():
    assert tp.nllb_code("ja") == "jpn_Jpan" and tp.nllb_code("ID") == "ind_Latn"
    assert tp.m2m_code("zh-TW") == "zh" and tp.nllb_code("zh-tw") == "zho_Hant"
    assert tp.iso("jpn") == "ja"
    with pytest.raises(ValueError):
        tp.iso("xx")
    assert tp.detect_lang(["坂本くん"]) == "ja" and tp.detect_lang(["안녕"]) == "ko"
    assert tp.detect_lang(["你好"]) == "zh" and tp.detect_lang(["hello"]) == "en"


def test_normalize_ja():
    assert tp.normalize_ja("それだけー？") == "それだけ？"
    assert tp.normalize_ja("わ、私はー・・・別に．．．") == "わ、私は…別に…"
    assert tp.normalize_ja("うーんそうだなぁ") == "ふむそうだなあ"
    assert tp.normalize_ja("くれてっ・・") == "くれて…"
    assert tp.normalize_ja("何！？！") == "何？！"
    assert tp.normalize_ja("メシアー") == "メシアー"  # katakana long vowel is part of the word


def test_kana_to_romaji():
    assert tp.kana_to_romaji("メシア") == "meshia"
    assert tp.kana_to_romaji("きょう") == "kyou"
    assert tp.kana_to_romaji("がっこう") == "gakkou"
    assert tp.kana_to_romaji("シャーロット") == "shaarotto"


def test_prepare_interjections_and_passthrough():
    p = tp.prepare("え？", "ja", "id")
    assert p.text == "" and tp.finish(p, "", "id") == "Eh?"
    p = tp.prepare("うんうん", "ja", "id")
    assert tp.finish(p, "", "id") == "Iya, iya"
    p = tp.prepare("えー、それだけー？", "ja", "id")
    assert p.lead == "Eh," and p.text == "それだけ？" and p.end == "?"
    assert tp.finish(p, "Itu saja.", "id") == "Eh, itu saja?"
    p = tp.prepare("あっあとね", "ja", "id")
    assert p.lead == "Ah," and p.text == "あとね"
    p = tp.prepare("うーんそうだなぁ・・優しいよね", "ja", "en")
    assert p.lead == "Hmm," and p.text.startswith("そうだな")
    assert tp.prepare("え？", "ja", "id", interjections=False).text == "え？"


def test_prepare_glossary_honorifics_names():
    p = tp.prepare("ねえねえ、めいは坂本のことどう思う？", "ja", "id", GLOSSARY)
    assert "Sakamoto" in p.text and "Mei" in p.text and "坂本" not in p.text
    assert p.lead == "Hei, hei," and set(p.terms) == {"Sakamoto", "Mei"}
    p = tp.prepare("坂本くん？", "ja", "id", GLOSSARY)
    assert p.text == "Sakamoto-kun？" and not tp.has_source_script(p.text, "ja")
    assert tp.finish(p, p.text, "id") == "Sakamoto-kun?"
    # no glossary: common surnames are romanised, katakana names get honorifics
    assert tp.prepare("坂本くん？", "ja", "id").text == "Sakamoto-kun？"
    assert tp.prepare("メシアちゃん、来て", "ja", "id").text.startswith("Meshia-chan")
    # common nouns keep their さん (店員さん = the clerk), katakana ones too (メイドさん = the maid)
    assert "店員さん" in tp.prepare("店員さんに聞いた", "ja", "id").text
    assert tp.prepare("メイドさん、こっち！", "ja", "id").text == "メイドさん、こっち！"
    assert tp.prepare("シスターちゃん", "ja", "id").text == "シスターちゃん"


@pytest.mark.parametrize("text", [
    "途中田んぼがある", "日本間で寝よう", "大西洋を渡る船だ", "小川で遊んだ", "大石が落ちてきた！",
    "登山中に雨が降った", "望月の夜だね",
])
def test_surnames_not_matched_inside_words(text):
    p = tp.prepare(text, "ja", "id")
    assert not p.terms and not re.search("[A-Za-z]", p.text), p.text


def test_surnames_in_context():
    assert tp.prepare("小川さんと遊んだ", "ja", "id").text.startswith("Ogawa-san")
    assert tp.prepare("中田は来た", "ja", "id").terms == ["Nakata"]
    assert tp.prepare("中田は来た", "ja", "id", names=False).text == "中田は来た"
    assert FakeSeq2Seq(names=False).names is False


def test_glossary_edge_cases():
    p = tp.prepare("メシアだ", "ja", "id", [GlossaryEntry(source="メシア", target="")])
    assert p.text == "メシアだ" and not p.terms  # an empty target must not delete the word
    # a pivot hop gets the term already substituted; it must still be required in the output
    p = tp.prepare("Mei は Sakamoto のこと", "ja", "en", GLOSSARY)
    assert "Mei" in p.terms and p.text.startswith("Mei は")
    assert tp.has_term("Meiは", "Mei") and not tp.has_term("Meiko", "Mei")


def test_prepare_stutter():
    p = tp.prepare("わ、私はー・・・別にそんな・・・", "ja", "id")
    assert p.stutter and p.text == "私は…別にそんな…" and p.end == "..."
    assert tp.finish(p, "Aku tidak seperti itu.", "id") == "A-aku tidak seperti itu..."
    assert not tp.prepare("あっ、あとね", "ja", "id").stutter
    # a stutter on an interjection stays on it instead of eating it
    assert tp.finish(tp.prepare("あ、あの…", "ja", "id"), "", "id") == "A-anu..."
    # a pronoun stutter lands on the pronoun even when MT opens with something else
    p = tp.prepare("わ、私は", "ja", "id")
    assert tp.finish(p, "Maksudku, aku tidak", "id") == "Maksudku, a-aku tidak"
    assert tp.finish(tp.prepare("わ、私は", "ja", "en"), "I don't", "en") == "I-I don't"


def test_interjection_context():
    assert tp.prepare("もう？", "ja", "id").text == "もう？"  # "already?", not "geez"
    assert tp.finish(tp.prepare("もう！", "ja", "id"), "", "id") == "Ih!"


def test_finish_punctuation_casual_terms():
    p = tp.prepare("反応が面白いなとか", "ja", "id")
    assert tp.finish(p, "Saya pikir reaksi Anda lucu.", "id") == "Aku pikir reaksi kamu lucu"
    assert tp.finish(p, "Saya pikir.", "id", casual=False) == "Saya pikir"
    p = tp.prepare("じゃあメシアはどうなの？", "ja", "id", GLOSSARY)
    assert tp.finish(p, "Bagaimana dengan Messias?", "id") == "Bagaimana dengan Mesia?"
    assert tp.clean_output("Tunggu . . . apa ?!") == "Tunggu... apa?!"


def test_repetition_guard():
    looping = "Hei, Hei, Hei, Hei, Hei, Hei, Hei, Hei"
    assert tp.is_repetitive(looping)
    assert tp.is_repetitive("Ya ya ya ya ya ya ya")
    assert tp.is_repetitive("aaaaaaaaaaa")
    assert not tp.is_repetitive("Hei, hei, apa pendapat Mei tentang Sakamoto?")
    assert not tp.is_repetitive("Iya, iya")
    assert tp.collapse_repetition(looping) == "Hei, Hei"
    assert tp.is_bad("坂本", "", "id") and tp.is_bad("坂本", "坂本さん", "id")
    assert not tp.is_bad("それだけ？", "Itu saja?", "id")


def test_chrf():
    assert tp.chrf(["Iya, iya"], ["Iya, iya"]) == pytest.approx(100.0)
    assert tp.chrf(["xyz"], ["abc"]) == 0.0
    good = tp.chrf(["Eh, cuma itu?"], ["Eh, cuma itu ya?"])
    bad = tp.chrf(["Apa yang kau lakukan?"], ["Eh, cuma itu ya?"])
    assert 50 < good < 100 and bad < 30


def test_apply_dict():
    assert tp.apply_dict("Halo Kak", {r"\bKak\b": "Kakak"}) == "Halo Kakak"
    assert tp.apply_dict("abc", None) == "abc"


# --- Seq2SeqTranslator with a fake model ------------------------------------------------------

class FakeSeq2Seq(Seq2SeqTranslator):
    """generate() scripted: first pass loops for one input, the strict retry fixes it."""

    def __init__(self, **kw):
        super().__init__("fake/repo", "nllb", **kw)
        self.calls: list[tuple[list[str], dict]] = []

    def generate(self, texts, source, target, **overrides):
        self.calls.append((list(texts), overrides))
        out = []
        for t in texts:
            if "優しい" in t:
                out.append("Baik." if overrides else "ya ya ya ya ya ya ya ya")
            else:
                out.append(f"terjemahan {len(t)}.")
        return out


def test_translator_retry_passthrough_and_order():
    tr = FakeSeq2Seq()
    srcs = ["え？", "坂本くん？", "うーんそうだなぁ・・優しいよね", "店の店員さんとか", ""]
    req = TranslationRequest(page_bgr=None, regions=[], sources=srcs,
                             context=PageContext(glossary=GLOSSARY))
    items = tr.translate(req)
    assert [i.source for i in items] == srcs
    targets = [i.target for i in items]
    assert targets[0] == "Eh?" and targets[1] == "Sakamoto-kun?"
    assert targets[2] == "Hmm, baik" and targets[4] == ""
    assert targets[3].startswith("Terjemahan")
    first, retry = tr.calls
    assert len(first[0]) == 2 and first[1] == {}
    assert retry[0] == ["そうだなあ…優しいよね"] and retry[1]["no_repeat_ngram_size"] == 2


def test_translator_same_language_is_identity():
    assert FakeSeq2Seq().translate_texts(["halo"], "id", "id") == ["halo"]


def test_registered_methods():
    load_all()
    for name in ("nllb", "nllb_big", "m2m100", "m2m100_big", "chain"):
        assert name in TRANSLATORS.names()
    tr = TRANSLATORS.build("m2m100_big", Settings(), num_beams=2)
    assert tr.repo == "facebook/m2m100_1.2B" and tr.num_beams == 2  # lazy: nothing downloaded
    assert TRANSLATORS.build("nllb", Settings(), model="/ckpt/mine").repo == "/ckpt/mine"


# --- chain ------------------------------------------------------------------------------------

def test_parse_steps():
    assert parse_steps("m2m100:en,nllb:id") == [
        {"translator": "m2m100", "to": "en", "opts": {}},
        {"translator": "nllb", "to": "id", "opts": {}},
    ]
    steps = parse_steps('[{"translator": "nllb", "to": "en", "num_beams": 2}, {"translator": "nllb"}]')
    assert steps[0]["opts"] == {"num_beams": 2} and steps[1]["to"] is None
    with pytest.raises(ValueError):
        parse_steps([])
    with pytest.raises(ValueError):
        parse_steps([{"translator": "chain"}])


class Tagger:
    vision = False

    def __init__(self, tag):
        self.tag, self.seen = tag, []

    def translate(self, req):
        """Echoes its input as `source`, like Seq2SeqTranslator does."""
        self.seen.append((req.source_lang, req.target_lang, list(req.sources)))
        return [
            TranslationItem(source=s, target=f"[{self.tag}] {s}" if s else "")
            for s in req.sources
        ]


def test_chain_hops(monkeypatch):
    built = {}

    def fake_build(name, settings, **opts):
        built[name] = Tagger(name)
        return built[name]

    monkeypatch.setattr(chain_mod.TRANSLATORS, "build", fake_build)
    ch = ChainTranslator(Settings(), "a:en,b:id")
    req = TranslationRequest(page_bgr=None, regions=[], sources=["え？", "えー、それだけー？"])
    items = ch.translate(req)
    assert built["a"].seen == [("ja", "en", ["", "それだけ？"])]
    assert built["b"].seen[0][:2] == ("en", "id")
    # sources stay the OCR text, not the prepared text the first hop saw
    assert [i.source for i in items] == ["え？", "えー、それだけー？"]
    assert ChainTranslator(Settings(), "a:eng,b:ind")._hops("jpn", "id") == [
        ("ja", "en"), ("en", "id"),
    ]
    assert items[0].target == "Eh?"  # resolved once for the final language, not via English
    assert items[1].target == "Eh, [b] [a] それだけ?"
    with pytest.raises(ValueError):
        ChainTranslator(Settings(), "a:en").translate(req)  # ends in en, target is id


# --- real models on the page01 ground truth ---------------------------------------------------

def _gt():
    bubbles = json.loads(GT.read_text())["bubbles"]
    return [b["ja"] for b in bubbles], [b["id_ref"] for b in bubbles]


def _assert_indonesian(outs):
    assert all(o.strip() for o in outs)
    assert not any(tp.is_repetitive(o) for o in outs), outs
    assert all(tp.foreign_ratio(o, "id") == 0 for o in outs), outs
    words = " ".join(outs).lower().split()
    assert sum(w.strip(",.?!") in {"aku", "itu", "apa", "saja", "dengan", "kamu", "iya"}
               for w in words) >= 5


@needs_nllb
def test_nllb_page01_quality():
    src, ref = _gt()
    tr = TRANSLATORS.build("nllb", Settings(), threads=2)
    outs = tr.translate_texts(src, "ja", "id", GLOSSARY)
    for s, o in zip(src, outs):
        print(f"{s} -> {o}")
    _assert_indonesian(outs)
    assert outs[1] == "Sakamoto-kun?" and "Sakamoto" in outs[0] and "Mesia" in outs[4]
    assert tp.chrf(outs, ref) > 25  # raw NLLB (no pre/post processing) scores 17.6


@needs_both
def test_m2m100_and_chain_page01():
    src, ref = _gt()
    m2m = TRANSLATORS.build("m2m100", Settings(), threads=2)
    outs = m2m.translate_texts(src, "ja", "id", GLOSSARY)
    _assert_indonesian(outs)
    assert tp.chrf(outs, ref) > 25  # raw M2M100 scores 19.0 with 3/13 looping outputs

    ch = TRANSLATORS.build("chain", Settings(), steps=[
        {"translator": "m2m100", "to": "en", "threads": 2},
        {"translator": "nllb", "to": "id", "threads": 2},
    ])
    outs = ch.translate_texts(src, "ja", "id", GLOSSARY)
    for s, o in zip(src, outs):
        print(f"{s} -> {o}")
    _assert_indonesian(outs)
    assert tp.chrf(outs, ref) > 24
