"""Hiragana first names are romanised before MT where the text marks them as names."""

from terjemahin_worker.translate import textproc as tp


def test_hiragana_name_before_honorific():
    assert tp.romanize_kana_names("ゆいちゃん、おはよう！") == ("Yui-chan、おはよう！", ["Yui"])
    assert tp.romanize_kana_names("けんたくんはどこ？")[1] == ["Kenta"]
    assert tp.romanize_kana_names("あいつはゆいちゃんが好き")[1] == ["Yui"]  # after a particle


def test_hiragana_name_after_namae_wa():
    text, found = tp.romanize_kana_names("私の名前はさくらです")
    assert found == ["Sakura"] and "Sakura" in text and "さくら" not in text


def test_hiragana_words_with_honorifics_are_not_names():
    for t in ("おかあさんが来た", "みなさん、こんにちは", "お姉ちゃん！", "あかちゃんみたい",
              "店員さんとかにも", "かみさまお願い", "ねえさんどこ"):
        assert tp.romanize_kana_names(t) == (t, []), t


def test_prepare_keeps_the_name_as_a_term():
    p = tp.prepare("私の名前はさくらです", "ja", "id")
    assert "Sakura" in p.text and "Sakura" in p.terms
    p = tp.prepare("ゆいちゃん、おはよう！", "ja", "id")
    assert "Yui-chan" in p.text


def test_casual_namaku():
    assert tp.apply_dict("Nama saya Sakura", tp.CASUAL["id"]) == "Namaku Sakura"


def test_vocative_senpai_is_kept_as_a_form_of_address():
    assert tp.romanize_vocative("先輩、好きです！") == ("Senpai、好きです！", ["Senpai"])
    assert tp.romanize_vocative("先生に怒られた") == ("先生に怒られた", [])  # a subject: translated
    assert "Senpai" in tp.prepare("先輩、好きです！", "ja", "id").terms


def test_leading_name_keeps_its_comma():
    p = tp.prepare("先輩、好きです！", "ja", "id")
    assert tp.finish(p, "Senpai aku suka!", "id") == "Senpai, aku suka!"
    p = tp.prepare("ゆいちゃん、おはよう！", "ja", "id")
    assert tp.finish(p, "Yui-chan", "id") == "Yui-chan, Selamat pagi!"
