"""Text processing shared by the offline (machine translation) translators.

Seq2seq MT models are trained on clean prose, not manga: elongation marks (それだけー), stutters
(わ、私), dot runs (・・・), honorifics and bare interjections (え？, うんうん) are where they
hallucinate. `prepare` turns a bubble into clean source text plus deterministic pieces that are
re-attached by `finish` after translation; it also enforces the glossary. `chrf` scores output.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..models import GlossaryEntry

# ISO 639-1 -> (NLLB / FLORES-200 code, M2M100 code, English name)
LANGS: dict[str, tuple[str, str, str]] = {
    "ja": ("jpn_Jpan", "ja", "Japanese"),
    "id": ("ind_Latn", "id", "Indonesian"),
    "en": ("eng_Latn", "en", "English"),
    "zh": ("zho_Hans", "zh", "Chinese (Simplified)"),
    "zh-hant": ("zho_Hant", "zh", "Chinese (Traditional)"),
    "ko": ("kor_Hang", "ko", "Korean"),
    "ms": ("zsm_Latn", "ms", "Malay"),
    "jv": ("jav_Latn", "jv", "Javanese"),
    "su": ("sun_Latn", "su", "Sundanese"),
    "tl": ("tgl_Latn", "tl", "Tagalog"),
    "vi": ("vie_Latn", "vi", "Vietnamese"),
    "th": ("tha_Thai", "th", "Thai"),
    "fr": ("fra_Latn", "fr", "French"),
    "de": ("deu_Latn", "de", "German"),
    "es": ("spa_Latn", "es", "Spanish"),
    "pt": ("por_Latn", "pt", "Portuguese"),
    "it": ("ita_Latn", "it", "Italian"),
    "nl": ("nld_Latn", "nl", "Dutch"),
    "pl": ("pol_Latn", "pl", "Polish"),
    "cs": ("ces_Latn", "cs", "Czech"),
    "hu": ("hun_Latn", "hu", "Hungarian"),
    "ro": ("ron_Latn", "ro", "Romanian"),
    "ru": ("rus_Cyrl", "ru", "Russian"),
    "uk": ("ukr_Cyrl", "uk", "Ukrainian"),
    "sr": ("srp_Cyrl", "sr", "Serbian"),
    "hr": ("hrv_Latn", "hr", "Croatian"),
    "tr": ("tur_Latn", "tr", "Turkish"),
    "ar": ("arb_Arab", "ar", "Arabic"),
    "hi": ("hin_Deva", "hi", "Hindi"),
}
_ALIASES = {"zh-cn": "zh", "zh-hans": "zh", "zh-tw": "zh-hant", "jp": "ja", "kr": "ko", "in": "id",
            "jpn": "ja", "ind": "id", "eng": "en", "chs": "zh", "cht": "zh-hant", "kor": "ko"}


def iso(lang: str) -> str:
    code = lang.strip().lower().replace("_", "-")
    code = _ALIASES.get(code, code)
    if code not in LANGS:
        raise ValueError(f"unsupported language '{lang}', choose from: {', '.join(sorted(LANGS))}")
    return code


def nllb_code(lang: str) -> str:
    return LANGS[iso(lang)][0]


def m2m_code(lang: str) -> str:
    return LANGS[iso(lang)][1]


_KANA = re.compile(r"[぀-ヿㇰ-ㇿｦ-ﾟ]")
_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_HANGUL = re.compile(r"[가-힯ᄀ-ᇿ]")
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯ｦ-ﾟ]")
_LETTER = re.compile(r"[^\W\d_]")


def detect_lang(texts: Iterable[str]) -> str:
    """Script-based guess, enough to resolve source_lang='auto' for manga (ja / zh / ko / en)."""
    s = "".join(texts)
    if _KANA.search(s):
        return "ja"
    if _HANGUL.search(s):
        return "ko"
    if _HAN.search(s):
        return "zh"
    return "en"


def detect_langs(texts: list[str]) -> list[str]:
    """source_lang='auto' per bubble: one stray kana in an OCR misread no longer makes a whole
    English page 'ja' (every English bubble then passed through untranslated). A bubble without
    letters ('...', '!?') takes the page's most common language."""
    own = [detect_lang([t]) if _LETTER.search(t) else None for t in texts]
    counts = Counter(x for x in own if x)
    page = counts.most_common(1)[0][0] if counts else "en"
    return [x or page for x in own]


CJK_LANGS = ("ja", "zh", "zh-hant", "ko")
HAN_TARGETS = ("zh", "zh-hant")  # write Japanese names in kanji themselves


def has_source_script(text: str, lang: str) -> bool:
    """True if `text` still contains something to translate from a CJK source."""
    if iso(lang) in ("ja", "zh", "zh-hant", "ko"):
        return bool(_CJK.search(text))
    return bool(_LETTER.search(text))


def foreign_ratio(text: str, target: str) -> float:
    """Share of letters in a script the target language does not use (CJK in Latin output)."""
    letters = _LETTER.findall(text)
    if not letters or iso(target) in ("ja", "zh", "zh-hant", "ko"):
        return 0.0
    return sum(1 for c in letters if _CJK.match(c)) / len(letters)


# --- kana -> romaji (Hepburn), for katakana names that have no glossary entry -------------

def _words(s: str) -> list[str]:
    return s.split()


_ROMAJI = dict(
    zip(
        "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽゔ",
        _words(
            "a i u e o ka ki ku ke ko sa shi su se so ta chi tsu te to na ni nu ne no ha hi fu he ho "
            "ma mi mu me mo ya yu yo ra ri ru re ro wa o n ga gi gu ge go za ji zu ze zo da ji zu "
            "de do ba bi bu be bo pa pi pu pe po vu"
        ),
    )
)
_SMALL = {"ゃ": "ya", "ゅ": "yu", "ょ": "yo", "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o"}


def _hira(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def kana_to_romaji(text: str) -> str:
    hira = _hira(text)
    out: list[str] = []
    i = 0
    while i < len(hira):
        c = hira[i]
        nxt = hira[i + 1] if i + 1 < len(hira) else ""
        if c == "っ":
            following = _ROMAJI.get(nxt, "")
            out.append(following[:1] if following and following[0] not in "aiueon" else "")
        elif c == "ー":
            if out and out[-1]:
                out.append(out[-1][-1])
        elif c in _SMALL:
            vowel = _SMALL[c]
            if out and out[-1] and out[-1][-1] in "aiueo":
                prev = out.pop()
                stem = prev[:-1]
                if c in "ゃゅょ":
                    stem = stem if stem.endswith(("sh", "ch", "j")) else stem + "y"
                    vowel = vowel[1]
                out.append(stem + vowel)
            else:
                out.append(vowel)
        else:
            out.append(_ROMAJI.get(c, c))
        i += 1
    return "".join(out)


# --- Japanese source normalisation -----------------------------------------------------------

HONORIFICS = {
    "くん": "kun", "君": "kun", "さん": "san", "ちゃん": "chan", "さま": "sama", "様": "sama",
    "せんぱい": "senpai", "先輩": "senpai", "せんせい": "sensei", "先生": "sensei", "殿": "dono",
    "たん": "tan", "氏": "shi",
}
_HONORIFIC_ALT = "|".join(sorted(map(re.escape, HONORIFICS), key=len, reverse=True))
# Honorifics are only mapped after a romanised name (glossary term) or a katakana name, never
# after common nouns (店員さん is "the clerk", not "Ten'in-san").
_LATIN_HON = re.compile(rf"([A-Za-z][A-Za-z'.\-]*)\s*({_HONORIFIC_ALT})")
_KATA_HON = re.compile(rf"([ァ-ヺ][ァ-ヺー]*)({_HONORIFIC_ALT})")
# Katakana common nouns that take さん/ちゃん (メイドさん = "the maid"): never romanised as names.
KATAKANA_NOUNS = frozenset(_words(
    "メイド シスター ナース コック パパ ママ バイト ウェイトレス ウエイトレス ウェイター ドクター "
    "マスター オーナー サンタ ガイド スタッフ プロ カメラマン モデル アイドル シェフ パティシエ "
    "ママン ベビー オバ オジ ボス リーダー キャプテン コーチ マネージャー ドライバー ポリス "
    "パイロット ファン ゲスト レディ ヒーロー プリンセス"
))

_DOTS = re.compile(r"(?:[・･.．。‥…]\s*){2,}|[…‥]")
_ELONG_AFTER = re.compile(r"(?<=[぀-ゟ一-鿿])[ーｰ〜～~]+")
_SMALL_VOWEL = str.maketrans("ぁぃぅぇぉ", "あいうえお")  # ねぇ/なぁ are elongations: ねえ/なあ
_TRAIL_TSU = re.compile(r"(?<=[ぁ-ゖ一-鿿])っ+(?=[、。！？!?….\s]|$)")
_STUTTER = re.compile(r"^([ぁ-ゖァ-ヺ])[、,，]\s*(?=(.))")
# Readings of kanji that commonly follow a stutter (わ、私 / ぼ、僕); kana compare directly.
_KANJI_ONSET = {"私": "わ", "僕": "ぼ", "俺": "お", "何": "な", "別": "べ", "違": "ち", "大": "だ",
                "好": "す", "本": "ほ", "知": "し", "君": "き", "待": "ま", "自": "じ", "全": "ぜ"}


def _stutter_on(text: str) -> str:
    """'わ、私は' -> '私' / 'ち、ちがう' -> 'ち': the char a lone repeated kana stutters on."""
    m = _STUTTER.match(text)
    if not m:
        return ""
    head, nxt = m.group(1), m.group(2)
    return nxt if _hira(head) == _hira(_KANJI_ONSET.get(nxt, nxt)) else ""


_PRONOUN_KANJI = frozenset("私僕俺")
PRONOUNS = {"id": ("aku", "saya", "gue", "gw"), "en": ("i", "me")}

_PUNCT_JA = str.maketrans({"!": "！", "?": "？", ",": "、", "，": "、", "｡": "。", "､": "、"})


def normalize_ja(text: str) -> str:
    """Strip manga-only noise MT models choke on; keep meaning-bearing punctuation."""
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT_JA)
    # OCR spaces between CJK chars are noise; spaces next to Latin words (a pivot hop's "Mei は")
    # keep the words apart
    text = re.sub(r"(?<=[^0-9A-Za-z\s])\s+(?=[^0-9A-Za-z\s])", "", text.strip())
    text = _DOTS.sub("…", text)
    text = re.sub(r"ー{2,}", "ー", text).translate(_SMALL_VOWEL)
    text = re.sub(r"う[ー〜～]+ん", "ふむ", text)  # "hmm", before ー stripping turns it into "yes"
    text = re.sub(r"ふ[ー〜～う]+ん", "ふうん", text)  # neutral "I see", not ふん "hmph"
    text = _ELONG_AFTER.sub("", text)
    text = _TRAIL_TSU.sub("", text)
    text = re.sub(r"([！？])[！？]+", lambda m: "？！" if "？" in m.group() and "！" in m.group() else m.group(1), text)
    text = re.sub(r"([、。])[、。]+", r"\1", text)
    return text


# Spoken contractions -> the standard forms MT models were trained on (信じらんない is OOV for
# NLLB, 信じられない is not). Applied to NFKC text before elongation marks are stripped, so the
# patterns allow a trailing ー/え (うるせー). Only unambiguous forms; order matters.
_ELONG = "[ーえぇ〜～]*+"  # possessive: "わかんねーよ" must not match without its ー
_ELONG1 = "[ーえぇ〜～]++"  # at least one: うめー is "tasty", うめ is a plum
_EOW = "(?=[よなねぞぜわさっ]|[^ぁ-ゖ]|$)"  # end of word: a sentence-final particle may follow
COLLOQUIAL_JA: list[tuple[str, str]] = [
    (r"こないだ", "この間"),
    (r"らんない", "られない"),
    (r"(?<=[かまさ知分])んない", "らない"),
    (rf"(?<=[知分わか])んね{_ELONG}{_EOW}", "らない"),
    # 知らねー -> 知らない; spelled out, since "だからね" must stay
    (rf"(知|わか|分か|い|たま|つま)らね{_ELONG}(?=[よなぞぜ]|[^ぁ-ゖ]|$)", r"\1らない"),
    (rf"うるせ{_ELONG}", "うるさい"), (rf"すげ{_ELONG}{_EOW}", "すごい"),
    (rf"やべ{_ELONG}{_EOW}", "やばい"), (rf"でけ{_ELONG1}{_EOW}", "でかい"),
    (rf"うめ{_ELONG1}{_EOW}", "うまい"),
    (r"マジで", "本当に"), (r"マジ(?=[？！。、…]|$)", "本当"),
    (r"じゃん(?=[？！。、…]|$)", "じゃないか"),
    (r"なきゃ", "なければ"), (r"なくちゃ", "なくては"),
    (r"ちゃおう", "てしまおう"), (r"じゃおう", "でしまおう"),
    (r"ちゃった", "てしまった"), (r"じゃった", "でしまった"),
    (r"ちゃう(?![ぁ-ゖ])", "てしまう"),
    # progressive てる/てた/てて after a te-form (使ってる, 聞いてた, してて, 食べてる); never
    # after the stems of verbs that end in て themselves (捨てる, 育てる, 慌てる)
    # (not 〜でたまらない "unbearably": 嫌いでたまらない is a na-adjective, not 嫌いでいた)
    (r"(?<=[っいしみきれけせべめげ])て(?=る|た(?!まら)|て|ない|ます)", "てい"),
    (r"(?<=[んい])(?<!なん)(?<!せい)で(?=る|た(?!まら)|て)", "でい"),  # 読んでる, 泳いでた (not なんで)
    # とか as a hedge ("...or something"): dropped mid-sentence (店員さんとかにも) and at the end
    # (not ことかも / 何とか / なんとか / ちょっとかも, where the と belongs to another word)
    (r"(?<![こ何ひそっ])(?<!なん)とか(?=[にもはがをでへ])", ""),
]
_TOKA_END = re.compile(
    rf"(?:(?<=[ぁ-ゖ一-鿿ー])(?:な|ね|よ)|(?<=[ぁ-ゖ一-鿿ー])(?<![こ何ひそっ])(?<!なん)){_ELONG}とか{_ELONG}(?=[。！？….]*$)"
)


# a dropped final とか ("reaksinya lucu, gitu") is rendered as the target's hedge word
HEDGES = {"id": "gitu", "en": "or something"}


def standardize_ja(text: str) -> tuple[str, bool]:
    """Colloquial Japanese -> standard forms; True if a final hedge とか was dropped."""
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT_JA)
    for pat, repl in COLLOQUIAL_JA:
        text = re.sub(pat, repl, text)
    hedge = _TOKA_END.search(text) is not None
    return _TOKA_END.sub("", text), hedge


def strip_stutter(text: str) -> str:
    """'わ、私は' -> '私は' when the head is a real stutter (see `prepare`), else unchanged."""
    return _STUTTER.sub("", text) if _stutter_on("".join(text.split())) else text


def map_honorifics(text: str) -> str:
    text = _LATIN_HON.sub(lambda m: f"{m.group(1)}-{HONORIFICS[m.group(2)]}", text)

    def kata(m: re.Match[str]) -> str:
        if m.group(1) in KATAKANA_NOUNS:
            return m.group()
        return f"{kana_to_romaji(m.group(1)).capitalize()}-{HONORIFICS[m.group(2)]}"

    return _KATA_HON.sub(kata, text)


# Common bare interjections. Leading/trailing ones are cut off and rendered from this table,
# because MT models turn a lone "え？" into "Apa itu?" and "うんうん" into "ya ya ya ya".
INTERJECTIONS: dict[str, dict[str, str]] = {
    "id": {
        "ねえねえ": "Hei, hei", "ねえ": "Hei", "ね": "Ya", "なあ": "Hei", "おい": "Hei",
        "ううん": "Nggak", "うん": "Iya", "うんうん": "Iya, iya", "ええ": "Iya",
        "はい": "Ya", "いや": "Nggak", "いいえ": "Bukan", "え": "Eh", "えと": "Anu", "えっと": "Anu",
        "あの": "Anu", "あ": "Ah", "ああ": "Ah", "あれ": "Lho", "お": "Oh", "おお": "Oh",
        "わあ": "Wah", "うわ": "Uwah", "へえ": "Hee", "ほう": "Hoo", "ふむ": "Hmm", "ふうん": "Hmm",
        "ふふ": "Fufu", "ふふふ": "Fufufu", "はは": "Haha", "あはは": "Ahaha", "ははは": "Hahaha",
        "へへ": "Hehe", "えへへ": "Ehehe", "くそ": "Sial", "ちくしょう": "Sialan",
        "ごめん": "Maaf", "ごめんなさい": "Maaf", "すみません": "Permisi", "ありがとう": "Makasih",
        "やった": "Hore", "よし": "Oke", "ほら": "Nih", "さあ": "Ayo", "まあ": "Yah", "もう": "Ih",
        "ちょっと": "Tunggu", "きゃ": "Kyaa", "ひ": "Hii", "は": "Hah", "ん": "Hm", "なに": "Apa",
        "なんで": "Kenapa", "そう": "Begitu", "そっか": "Oh, begitu", "なるほど": "Begitu ya",
        "ふん": "Hmph", "ちぇ": "Cih", "じゃあ": "Kalau gitu", "さて": "Nah", "ただいま": "Aku pulang",
        "おかえり": "Selamat datang", "おかえりなさい": "Selamat datang", "おはよう": "Selamat pagi",
        "おやすみ": "Selamat tidur", "じゃあね": "Dah", "またね": "Sampai nanti",
        "いただきます": "Selamat makan", "しょうがない": "Mau gimana lagi",
        "しょうがないなあ": "Ya ampun", "しょうがないな": "Ya ampun",
    },
    "en": {
        "ねえねえ": "Hey, hey", "ねえ": "Hey", "ね": "Right", "なあ": "Hey", "おい": "Hey",
        "ううん": "Nope", "うん": "Yeah", "うんうん": "Uh-huh", "ええ": "Yes",
        "はい": "Yes", "いや": "No", "いいえ": "No", "え": "Huh", "えと": "Um", "えっと": "Um",
        "あの": "Um", "あ": "Ah", "ああ": "Ah", "あれ": "Huh", "お": "Oh", "おお": "Oh",
        "わあ": "Wow", "うわ": "Whoa", "へえ": "Huh", "ほう": "Oho", "ふむ": "Hmm", "ふうん": "Hmm",
        "ふふ": "Hehe", "ふふふ": "Hehehe", "はは": "Haha", "あはは": "Ahaha", "ははは": "Hahaha",
        "へへ": "Hehe", "えへへ": "Ehehe", "くそ": "Damn", "ちくしょう": "Damn it",
        "ごめん": "Sorry", "ごめんなさい": "I'm sorry", "すみません": "Excuse me",
        "ありがとう": "Thanks", "やった": "Yes", "よし": "Alright", "ほら": "Look", "さあ": "Come on",
        "まあ": "Well", "もう": "Geez", "ちょっと": "Hey", "きゃ": "Kyaa", "ひ": "Eek", "は": "Hah",
        "ん": "Hm", "なに": "What", "なんで": "Why", "そう": "I see", "そっか": "I see",
        "なるほど": "I see",
        "ふん": "Hmph", "ちぇ": "Tch", "じゃあ": "Then", "さて": "Now then", "ただいま": "I'm home",
        "おかえり": "Welcome back", "おかえりなさい": "Welcome back", "おはよう": "Good morning",
        "おやすみ": "Good night", "じゃあね": "See ya", "またね": "See you",
        "いただきます": "Let's eat", "しょうがない": "Can't be helped",
        "しょうがないなあ": "Oh well", "しょうがないな": "Oh well",
    },
}
# Unambiguous interjections that may also open a segment without a comma (うーんそうだな).
_PREFIX_OK = ("ねえねえ", "ふむ", "うんうん", "えっと", "あはは", "えへへ", "うわっ", "あっ", "えっ", "おっ",
              "じゃあ")
_SEG = re.compile(r"[^、。！？…]+[、。！？…]*")


# Words that are interjections only as exclamations: もう！ is "geez!", もう？ is "already?".
_NOT_AS_QUESTION = frozenset(("もう",))


def _interjection(seg: str, table: dict[str, str]) -> str | None:
    core = seg.rstrip("、。！？…")
    if "？" in seg[len(core):] and core in _NOT_AS_QUESTION:
        return None
    for key in (core, core.replace("ー", ""), re.sub(r"[ーっ]", "", core)):
        if key in table:
            return table[key]
    return None


_TO_LATIN_PUNCT = str.maketrans({"、": ",", "。": ".", "！": "!", "？": "?", "…": "..."})


def _latin_punct(seg_punct: str) -> str:
    return seg_punct.translate(_TO_LATIN_PUNCT)


@dataclass
class Prepared:
    """A bubble split into model input and deterministic parts re-attached by `finish`."""

    text: str  # what the model translates ("" = nothing left for the model)
    lead: str = ""  # already-target-language prefix (interjections)
    trail: str = ""  # already-target-language suffix
    stutter: bool = False  # "わ、私" -> "A-aku"
    stutter_on: str = ""  # the source char stuttered on ("私"), to find the matching word
    end: str = ""  # terminal punctuation of the source: "?", "!", "?!", "...", ""
    terms: list[str] = field(default_factory=list)  # glossary targets the output must contain
    verbatim: str = ""  # the whole result, no model needed (a punctuation-only bubble)
    kanji: dict[str, str] = field(default_factory=dict)  # romaji -> kanji, for Chinese output
    addressee: str = ""  # a leading name / form of address set off by a comma ("Senpai、")


def _end_punct(text: str) -> str:
    m = re.search(r"[！？!?…。.]+$", text)
    if not m:
        return ""
    p = m.group()
    if "？" in p or "?" in p:
        return "?!" if ("！" in p or "!" in p) else "?"
    if "！" in p or "!" in p:
        return "!"
    if "…" in p or ".." in p:
        return "..."
    return "."


# Common Japanese surnames (2+ kanji) with Hepburn readings. MT models cannot read kanji names
# (NLLB turns 坂本 into "Apa yang kau lakukan?"), so frequent ones are romanised before MT.
# Place names that are also surnames (千葉, 石川, 福島...) are left out; a glossary entry wins.
# Spelled the way scanlations and official releases do (Sato, Kudo: no long vowels).
SURNAMES: dict[str, str] = dict(
    pair.split(":")
    for pair in _words(
        "佐藤:Sato 鈴木:Suzuki 高橋:Takahashi 田中:Tanaka 伊藤:Ito 渡辺:Watanabe 山本:Yamamoto "
        "中村:Nakamura 小林:Kobayashi 加藤:Kato 吉田:Yoshida 山田:Yamada 佐々木:Sasaki "
        "山口:Yamaguchi 松本:Matsumoto 井上:Inoue 木村:Kimura 斎藤:Saito 斉藤:Saito 清水:Shimizu "
        "山崎:Yamazaki 池田:Ikeda 橋本:Hashimoto 阿部:Abe 山下:Yamashita 中島:Nakajima 石井:Ishii "
        "小川:Ogawa 前田:Maeda 岡田:Okada 長谷川:Hasegawa 藤田:Fujita 後藤:Goto 近藤:Kondo "
        "村上:Murakami 遠藤:Endo 青木:Aoki 坂本:Sakamoto 福田:Fukuda 太田:Ota 西村:Nishimura "
        "藤井:Fujii 金子:Kaneko 岡本:Okamoto 藤原:Fujiwara 中野:Nakano 三浦:Miura 原田:Harada "
        "中川:Nakagawa 松田:Matsuda 竹内:Takeuchi 小野:Ono 田村:Tamura 中山:Nakayama 和田:Wada "
        "石田:Ishida 森田:Morita 上田:Ueda 内田:Uchida 柴田:Shibata 酒井:Sakai 横山:Yokoyama "
        "高木:Takagi 安藤:Ando 宮本:Miyamoto 大野:Ono 小島:Kojima 谷口:Taniguchi 今井:Imai "
        "工藤:Kudo 高田:Takada 増田:Masuda 丸山:Maruyama 杉山:Sugiyama 村田:Murata 大塚:Otsuka "
        "新井:Arai 小山:Koyama 平野:Hirano 藤本:Fujimoto 河野:Kono 野口:Noguchi 武田:Takeda "
        "松井:Matsui 岩崎:Iwasaki 菅原:Sugawara 木下:Kinoshita 久保:Kubo 佐野:Sano 野村:Nomura "
        "松尾:Matsuo 市川:Ichikawa 菊地:Kikuchi 菊池:Kikuchi 杉本:Sugimoto 古川:Furukawa "
        "大西:Onishi 島田:Shimada 水野:Mizuno 桜井:Sakurai 高野:Takano 吉川:Yoshikawa "
        "山内:Yamauchi 西田:Nishida 飯田:Iida 西川:Nishikawa 小松:Komatsu 北村:Kitamura "
        "安田:Yasuda 五十嵐:Igarashi 川口:Kawaguchi 平田:Hirata 中田:Nakata 久保田:Kubota "
        "服部:Hattori 岩田:Iwata 土屋:Tsuchiya 本田:Honda 樋口:Higuchi 秋山:Akiyama 田口:Taguchi "
        "永井:Nagai 山中:Yamanaka 中西:Nakanishi 吉村:Yoshimura 川上:Kawakami 石原:Ishihara "
        "大橋:Ohashi 松岡:Matsuoka 馬場:Baba 浅野:Asano 荒木:Araki 大久保:Okubo 野田:Noda "
        "小沢:Ozawa 田辺:Tanabe 川村:Kawamura 星野:Hoshino 黒田:Kuroda 尾崎:Ozaki 望月:Mochizuki "
        "永田:Nagata 熊谷:Kumagai 内藤:Naito 松村:Matsumura 西山:Nishiyama 大谷:Otani "
        "平井:Hirai 岩本:Iwamoto 片山:Katayama 本間:Honma 早川:Hayakawa 横田:Yokota "
        "荒井:Arai 大石:Oishi 鎌田:Kamata 宮田:Miyata 小田:Oda 石橋:Ishibashi 篠原:Shinohara "
        "須藤:Sudo 河合:Kawai 宮下:Miyashita 小池:Koike 吉岡:Yoshioka 桑原:Kuwahara "
        "白石:Shiraishi 東山:Higashiyama 桐生:Kiryu 神崎:Kanzaki 如月:Kisaragi 天野:Amano"
)
)
# Surnames that are also everyday words (小川 stream, 大石 big rock, 望月 full moon, 山中 in the
# mountains, 如月 February...): romanised only when an honorific follows (小川さん).
AMBIGUOUS_SURNAMES = frozenset(_words(
    "小川 大石 望月 山中 小山 清水 平野 川上 川口 小島 小池 大橋 石橋 白石 馬場 古川 金子 大野 "
    "東山 西山 山下 木下 大谷 大塚 如月 中島 石原 山口 早川 土屋"
))
_HAN_CHAR = re.compile(r"[㐀-䶿一-鿿豈-﫿々]")
_SURNAME_RE = re.compile("|".join(sorted(map(re.escape, SURNAMES), key=len, reverse=True)))
_HONORIFIC_AHEAD = re.compile(rf"\s*(?:{_HONORIFIC_ALT})")


def romanize_surnames(text: str) -> tuple[str, list[str]]:
    """Japanese has no word boundaries, so a surname only counts when it is not glued into a
    kanji compound (途中田んぼ, 日本間, 大西洋), and ambiguous ones need a following honorific."""
    found: list[str] = []

    def sub(m: re.Match[str]) -> str:
        word, a, b = m.group(), m.start(), m.end()
        honorific = _HONORIFIC_AHEAD.match(text, b) is not None
        if a > 0 and _HAN_CHAR.match(text[a - 1]):
            return word
        if b < len(text) and _HAN_CHAR.match(text[b]) and not honorific:
            return word
        if word in AMBIGUOUS_SURNAMES and not honorific:
            return word
        name = SURNAMES[word]
        found.append(name)
        return f" {name} "

    text = _SURNAME_RE.sub(sub, text)
    text = re.sub(r"\s+([、。！？…])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip(), found


# Hiragana first names (さくら, ゆい, けんた): MT garbles them ('Nama aku Saura', 'Sasura') or
# translates them as words. They are romanised where the text itself marks them as names:
# before an honorific (ゆいちゃん), or after 名前は / 名は (私の名前はさくらです).
_KANA_HON = ("ちゃん", "くん", "さん", "さま", "せんぱい", "先輩", "君", "様")
_KANA_HON_RE = re.compile(
    rf"(?:(?<=[^ぁ-ゖ])|^|(?<=[はがをにともへの]))([ぁ-ゖ]{{2,4}})({'|'.join(_KANA_HON)})"
)
_KANA_NAME_IS = re.compile(r"(名前は|名は)([ぁ-ゖ]{2,5}?)(です|だよ|だ|って|でーす|と申し|。|！|!|、|$)")
# hiragana words that take an honorific and are no names (おかあさん, みなさん, かみさま...)
KANA_HONORIFIC_NOUNS = frozenset(_words(
    "かあ とう にい ねえ ばあ じい おば おじ みな みんな あか ぼっ ぼう おきゃく きゃく かみ "
    "ほとけ おまわり おいしゃ おひめ ひめ おうじ じょう おじょう せんせい おくさ おくさん だんな "
    "おやぶん ぼうや おぼっ おとう おかあ おにい おねえ おばあ おじい あね あに おっちゃ "
    "おばちゃ ばっちゃ じっちゃ かっちゃ ねぇ にぃ おにぃ おねぇ みなさ いもうと おとうと"
))


# 先輩 / 先生 said to someone ("先輩、好きです！") are forms of address, which scanlations keep:
# m2m100 made "Tuanku" (my lord) and nllb dropped it. As a subject or object (先生に怒られた)
# they stay words for the models to translate.
_VOCATIVE = re.compile(r"^(先輩|せんぱい|センパイ|先生|せんせい)(?=[、，,！!？?…]|$)")
_VOCATIVE_WORD = {"先輩": "Senpai", "せんぱい": "Senpai", "センパイ": "Senpai", "先生": "Sensei",
                  "せんせい": "Sensei"}


def romanize_vocative(text: str) -> tuple[str, list[str]]:
    m = _VOCATIVE.match(text)
    if not m:
        return text, []
    word = _VOCATIVE_WORD[m.group(1)]
    return word + text[m.end():], [word]


def romanize_kana_names(text: str) -> tuple[str, list[str]]:
    """'ゆいちゃん' -> 'Yui-chan', '私の名前はさくらです' -> '私の名前はSakuraです'."""
    found: list[str] = []

    def hon(m: re.Match[str]) -> str:
        word = m.group(1)
        if word in KANA_HONORIFIC_NOUNS or word.startswith("お") or word[-1] in "はがをにもへの":
            return m.group()
        name = kana_to_romaji(word).capitalize()
        found.append(name)
        return f" {name}-{HONORIFICS[m.group(2)]} "

    def named(m: re.Match[str]) -> str:
        name = kana_to_romaji(m.group(2)).capitalize()
        found.append(name)
        return f"{m.group(1)} {name} {m.group(3)}"

    text = _KANA_NAME_IS.sub(named, text)
    text = _KANA_HON_RE.sub(hon, text)
    text = re.sub(r"\s+([、。！？…])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip(), found


def surname_kanji(text: str, found: list[str]) -> dict[str, str]:
    """romaji -> the kanji it was romanised from in `text` (Ono may be 小野 or 大野)."""
    out: dict[str, str] = {}
    for m in _SURNAME_RE.finditer(text):
        r = SURNAMES[m.group()]
        if r in found:
            out.setdefault(r, m.group())
    return out


# --- Chinese output: Japanese surnames back in kanji ----------------------------------------
# Kanji surnames are romanised before MT (NLLB turns a kanji 坂本 into 山本 / 哈萨克本, 服部 into
# 服装部长), and the models then write the romaji either in Latin ('Sakamoto') or as a phonetic
# transliteration (萨卡莫托, 塔纳卡, 瓦塔纳贝, 苏苏基, 萨卡莫 - 昆). `restore_kanji` finds either
# form and writes the kanji of the source instead, so 坂本 reads 坂本 on a Chinese page.

# characters Chinese MT uses to transliterate each Japanese mora (Simplified; Traditional
# forms are added below). Common words (大, 小, 山, 本, 和...) are left out on purpose: a match
# only replaces text when a whole run of these characters spells the name's morae in order.
_TRANSLIT_SRC = (
    "a:阿亚安 i:伊依艾 u:乌宇 e:埃艾伊惠 o:奥欧 "
    "ka:卡加喀嘉凯 ki:基吉奇 ku:库久古克 ke:克凯科基 ko:科克寇古可 "
    "sa:萨沙佐莎 shi:西希什石志 su:苏斯素 se:塞赛瑟濑 so:索苏宗 "
    "ta:塔达太多 chi:奇基吉齐智 tsu:茨津楚祖苏次 te:特泰提帖 to:托多图特德陀 "
    "na:纳娜那奈 ni:尼妮仁 nu:努奴 ne:内涅妮尼 no:诺野努乃 "
    "ha:哈 hi:希喜西 fu:富福弗夫 he:赫海黑 ho:霍豪保波 "
    "ma:马玛麻真 mi:米美弥密 mu:穆姆牟武 me:梅美迈 mo:莫摩默茂 "
    "ya:亚雅娅 yu:尤优由于裕友 yo:约尤优洋 "
    "ra:拉腊良 ri:里利莉理 ru:鲁卢露留 re:雷勒蕾列 ro:罗洛若路 wa:瓦娃哇 wo:沃 "
    "ga:加嘉伽贺冈 gi:吉基义 gu:古格谷 ge:格盖吉 go:戈高哥郭果 "
    "za:扎萨札 ji:吉基治 zu:祖兹苏 ze:泽哲濑 zo:佐索藏 "
    "da:达塔 de:德代迪 do:多德都道 "
    "ba:巴芭波 bi:比碧毕 bu:布部武 be:贝别倍部 bo:博波坊 "
    "pa:帕巴 pi:皮比 pu:普布 pe:佩贝 po:波坡宝 "
    "kya:恰卡 kyu:丘久秋 kyo:乔京久恭 sha:沙夏莎 shu:修舒秀周 sho:肖秀修昭正 "
    "cha:查恰茶 chu:丘楚秋 cho:乔秋 ja:贾嘉加 ju:朱尤久树 jo:乔卓城 "
    "hyo:表兵 ryo:良辽廖柳龙 ryu:龙柳留 myo:妙 "
    # a mora + syllabic n, often one character (Kondo 孔多, Honda 洪达, Kanzaki 坎扎基)
    "an:安 kan:坎康甘菅 san:桑 tan:坦丹谭 nan:南楠 han:汉韩 man:曼满万 ran:兰 wan:万旺 "
    "gan:甘冈 zan:赞 dan:丹 ban:班坂 pan:潘 en:恩远 ken:肯健坚 sen:森仙 ten:滕腾 nen:宁 "
    "hen:亨 men:门 ren:伦连莲 gen:根元源 zen:曾善 den:登 ben:弁 in:因音印 kin:金钦近 "
    "shin:新辛申信 chin:钦金 jin:金仁神 min:明民 rin:林琳凛 nin:宁忍 hin:欣品 bin:宾 "
    "un:温云 kun:昆坤 sun:孙村 tsun:村 nun:农 hun:洪 fun:丰芬 mun:蒙 run:伦 gun:军贡 "
    "zun:尊 bun:文 on:翁恩安 kon:孔康空昆近 son:松宋孙 ton:东通顿 non:农 hon:洪宏 "
    "mon:蒙门 ron:隆 gon:贡冈 don:东敦 bon:邦 yon:永 "
    # a small っ, when written at all (Hattori 哈特里 / 哈托里)
    "Q:特克普斯茨 N:恩"
)
_TRAD = str.maketrans(
    "萨纳贝苏亚马玛罗鲁卢诺库乌达兰伦莲门东顿农贡军宾钦孙丰乔辽龙坚腾宁远冈韩汉满万赞谭贺"
    "盖泽濑迈弥娅优约里丽齐赛图内毕别宝贾凯释",
    "薩納貝蘇亞馬瑪羅魯盧諾庫烏達蘭倫蓮門東頓農貢軍賓欽孫豐喬遼龍堅騰寧遠岡韓漢滿萬讚譚賀"
    "蓋澤瀨邁彌婭優約裡麗齊賽圖內畢別寶賈凱釋",
)
_TRANSLIT: dict[str, frozenset[str]] = {
    k: frozenset(v + v.translate(_TRAD))
    for k, v in (pair.split(":") for pair in _words(_TRANSLIT_SRC))
}
_MORA = re.compile(r"(?:ky|gy|sh|ch|ts|ny|hy|my|ry|by|py|j|f|[kgsztdnhbpmyrwv])?[aiueo]")


def _morae(romaji: str) -> list[tuple[frozenset[str], bool]]:
    """'Hattori' -> slots [ha, (っ, optional), to, ri]: each a set of Han characters the mora
    may be written with, and whether it may be left out. A mora + syllabic n accepts either
    the fused character (Kon 孔) or the mora's own character, with an optional 恩 after it."""
    s, i, slots = romaji.lower(), 0, []
    while i < len(s):
        if s[i] == "n" and (i + 1 == len(s) or s[i + 1] not in "aiueoy"):
            if slots:
                prev = slots[-1][2]
                fused = _TRANSLIT.get(prev + "n", frozenset())
                slots[-1] = (slots[-1][0] | fused, False, prev + "n")
            slots.append((_TRANSLIT["N"], True, "N"))
            i += 1
            continue
        if i + 1 < len(s) and s[i] == s[i + 1] and s[i] not in "aiueon" or s.startswith("tch", i):
            slots.append((_TRANSLIT["Q"], True, "Q"))
            i += 1
            continue
        m = _MORA.match(s, i)
        if not m:
            i += 1  # a letter Hepburn does not use: skip it
            continue
        slots.append((_TRANSLIT.get(m.group(), frozenset()), False, m.group()))
        i = m.end()
    return [(chars, opt) for chars, opt, _ in slots]


def _match_at(run: str, start: int, slots: list[tuple[frozenset[str], bool]], misses: int,
              trunc: bool) -> int | None:
    """End of the longest match of `slots` in `run` from `start` (None: no match). `misses`
    required morae may be written with another character, and with `trunc` the last required
    mora may be missing (萨卡莫 for Sakamoto)."""
    best: int | None = None

    def go(pos: int, k: int, miss: int) -> None:
        nonlocal best
        if k == len(slots):
            best = pos if best is None else max(best, pos)
            return
        chars, optional = slots[k]
        if optional:
            go(pos, k + 1, miss)
        if pos < len(run):
            if run[pos] in chars:
                go(pos + 1, k + 1, miss)
            elif not optional and miss > 0 and k > 0:  # the first character must match
                go(pos + 1, k + 1, miss - 1)
        if trunc and not optional and miss == misses and k == len(slots) - 1 and k > 0:
            best = pos if best is None else max(best, pos)

    go(start, 0, misses)
    return best


def find_transliteration(text: str, romaji: str) -> str | None:
    """The span of `text` that transliterates the surname `romaji` into Han characters
    ('你对萨卡莫托有什么看法?' -> '萨卡莫托'), or None. Every required mora must match in order;
    a name of 4+ morae may have one written otherwise (太卡哈西 for Takahashi) or lose its last
    mora (萨卡莫)."""
    slots = _morae(romaji)
    required = sum(1 for _, opt in slots if not opt)
    if required < 2:
        return None
    misses = 1 if required >= 4 else 0
    trunc = required >= 4  # 'Arai' would match the 阿拉 of 阿拉伯
    best: tuple[int, int, int] | None = None  # (length, -start, start)
    for run in re.finditer(r"[㐀-䶿一-鿿豈-﫿]+", text):
        s = run.group()
        for a in range(len(s)):
            end = _match_at(s, a, slots, misses, trunc)
            if end is not None and end - a >= 2:
                cand = (end - a, -(run.start() + a), run.start() + a)
                if best is None or cand > best:
                    best = cand
    if best is None:
        return None
    return text[best[2] : best[2] + best[0]]


# honorifics as Chinese scanlations write them after a name
HAN_HONORIFICS = {"kun": "君", "chan": "酱", "san": "桑", "sama": "大人", "senpai": "前辈",
                  "sensei": "老师", "dono": "殿", "tan": "碳", "shi": "氏"}
# what MT makes of a romaji honorific: Latin, transliterated (昆, 库恩, 森西), or a real
# Chinese one (先生, 君), which is kept as written
_HON_TRANSLIT = {"昆": "kun", "坤": "kun", "库恩": "kun", "庫恩": "kun", "森西": "sensei",
                 "森赛": "sensei", "森派": "senpai", "森帕伊": "senpai", "仙派": "senpai",
                 "萨马": "sama", "薩馬": "sama", "多诺": "dono", "多諾": "dono", "醬": "chan"}
_HON_CHINESE = ("先生", "老师", "老師", "前辈", "前輩", "学长", "學長", "大人", "同学", "同學",
                "小姐", "女士", "君", "桑", "酱", "醬", "殿", "氏", "様", "样")
_HON_AFTER = re.compile(
    r"[ \t\-‐–—·・]*(" + "|".join(sorted(
        [*map(re.escape, _HON_TRANSLIT), *map(re.escape, _HON_CHINESE), *HAN_HONORIFICS],
        key=len, reverse=True)) + r")(?![A-Za-z])",
    re.IGNORECASE,
)
_HANT_HON = str.maketrans("酱辈师", "醬輩師")


def _honorific_key(word: str) -> str | None:
    w = word.lower()
    if w in HAN_HONORIFICS:
        return w
    return _HON_TRANSLIT.get(word)


def restore_kanji(text: str, kanji: dict[str, str], source: str = "", hant: bool = False) -> str:
    """Chinese output: the source's kanji for each romanised surname, whether MT kept the
    romaji ('Sakamoto-kun?' -> '坂本君?') or transliterated it ('你对萨卡莫托有什么看法?' ->
    '你对坂本有什么看法?', '萨卡莫 - 昆?' -> '坂本君?'). `source` is the text MT was given:
    an honorific it had ('Sakamoto-kun') and MT dropped is put back after the kanji."""
    for romaji, k in kanji.items():
        latin = rf"(?<![A-Za-z]){re.escape(romaji)}(?![A-Za-z])"
        if re.search(latin, text, re.IGNORECASE):
            text = re.sub(latin, k, text, flags=re.IGNORECASE)
        elif k not in text and (span := find_transliteration(text, romaji)):
            text = text.replace(span, k)
        if k not in text:
            continue

        def hon(m: re.Match[str], k: str = k) -> str:
            key = _honorific_key(m.group(1))
            word = HAN_HONORIFICS[key] if key else m.group(1)
            return k + (word.translate(_HANT_HON) if hant else word)

        text = re.sub(re.escape(k) + _HON_AFTER.pattern, hon, text, flags=re.IGNORECASE)
        src = re.findall(rf"(?<![A-Za-z]){re.escape(romaji)}-([a-z]+)", source, re.IGNORECASE)
        keys = {s.lower() for s in src if s.lower() in HAN_HONORIFICS}
        if len(keys) == 1 and text.count(k) == 1 and not _HON_AFTER.match(text, text.index(k) + len(k)):
            word = HAN_HONORIFICS[keys.pop()]
            text = text.replace(k, k + (word.translate(_HANT_HON) if hant else word))
    return text


def kanji_kept(text: str, kanji: dict[str, str]) -> bool:
    """Every romanised surname is still in the output, in any form `restore_kanji` repairs:
    False when MT dropped a name ('Kondo-san が言った' -> '刚刚说:')."""
    fixed = restore_kanji(text, kanji)
    return all(k in fixed for k in kanji.values())


def apply_glossary(text: str, glossary: Iterable[GlossaryEntry]) -> tuple[str, list[str]]:
    """Forced replacement: the source term becomes its target spelling inside the source text.

    NLLB/M2M100 copy Latin-script spans verbatim, so names stay consistent across bubbles.
    Longest terms first so "坂本くん" beats "坂本".
    """
    terms: list[str] = []
    for g in sorted(glossary, key=lambda g: len(g.source), reverse=True):
        if not g.source or not g.target.strip():  # an empty target would delete the word
            continue
        if g.source in text:
            text = text.replace(g.source, f" {g.target} ")
            terms.append(g.target)
        elif _LETTER.search(g.target) and has_term(text, g.target):
            terms.append(g.target)  # already substituted (pivot hop): still must survive MT
    text = re.sub(r"\s+([、。！？…])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip(), terms


_PUNCT_CHARS = re.compile(r"[\s!?.・･。、,~〜～ー\-―—]+")


def punctuation_only(text: str) -> str | None:
    """'？' -> '?', '……' / '・・・' -> '...', '！！' -> '!!', '！？' -> '!?': a bubble of
    punctuation alone is kept as punctuation. Translated, it came back empty, and the bubble was
    erased and left blank. None when the text has letters, digits or other symbols."""
    s = unicodedata.normalize("NFKC", text).strip()
    if not s or not _PUNCT_CHARS.fullmatch(s) or not re.search(r"[!?.・･。]", s):
        return None
    s = re.sub(r"[\s、,~〜～ー\-―—]+", "", s)
    s = re.sub(r"[・･。]", ".", s)
    return re.sub(r"\.{2,}", "...", s)


def prepare(
    text: str,
    source: str,
    target: str,
    glossary: Iterable[GlossaryEntry] = (),
    interjections: bool = True,
    names: bool = True,
    standardize: bool = True,
) -> Prepared:
    source, target = iso(source), iso(target)
    raw = text.strip()
    punct = punctuation_only(raw)
    if punct is not None:
        return Prepared(text="", verbatim=punct, end=_end_punct(punct))
    if source != "ja":
        out, terms = apply_glossary(unicodedata.normalize("NFKC", raw), glossary)
        return Prepared(text=out, end=_end_punct(out), terms=terms)

    hedge = False
    text = raw
    if standardize:
        text, hedge = standardize_ja(raw)
    text = normalize_ja(text)
    end = _end_punct(text)
    # checked on the raw text: "あっ、あと" is an exclamation, only a bare "あ、あと" stutters
    stutter_on = _stutter_on("".join(unicodedata.normalize("NFKC", raw).split()))
    stutter = bool(stutter_on)
    if stutter:
        text = _STUTTER.sub("", text)
    text, terms = apply_glossary(text, glossary)
    kanji: dict[str, str] = {}
    if names:
        before = text
        text, found = romanize_surnames(text)
        text, kana = romanize_kana_names(text)
        text, voc = romanize_vocative(text)
        kana += voc
        if target in HAN_TARGETS:  # written back in kanji by finish(), wherever MT kept it
            kanji = surname_kanji(before, found)
        if target not in CJK_LANGS:  # Latin output must keep the name; Hangul/Han need not
            terms += found + kana
    text = map_honorifics(text)

    lead: list[str] = []
    trail: list[str] = []
    table = INTERJECTIONS.get(target) if interjections else None
    if table:
        segs = _SEG.findall(text)
        while segs:
            seg = segs[0]
            word = _interjection(seg, table)
            if word is None:
                pre = next((p for p in _PREFIX_OK if seg.startswith(p) and len(seg) > len(p)), None)
                if pre:
                    lead.append(f"{_interjection(pre, table)},")
                    segs[0] = seg[len(pre):]
                break
            lead.append(word + _latin_punct(seg[len(seg.rstrip("、。！？…")):] or ","))
            segs.pop(0)
        while segs and (word := _interjection(segs[-1], table)) is not None:
            seg = segs.pop()
            trail.insert(0, word + _latin_punct(seg[len(seg.rstrip("、。！？…")):]))
        text = "".join(segs)
    if hedge and text and HEDGES.get(target):
        trail.insert(0, f", {HEDGES[target]}")
    m = re.match(r"([A-Z][A-Za-z']*(?:-[a-z]+)?)\s*[、，,]", text.strip())
    addressee = m.group(1) if m and m.group(1).split("-")[0] in terms else ""
    return Prepared(
        text=text.strip(),
        addressee=addressee,
        lead=" ".join(lead),
        trail=" ".join(trail),
        stutter=stutter,
        stutter_on=stutter_on,
        end=end,
        terms=terms,
        kanji=kanji,
    )


_SENT_CJK = re.compile(r"[^。！？!?]+[。！？!?]*|[。！？!?]+")
_SENT_LATIN = re.compile(r"(?<=[.!?…])\s+(?=\S)")


def split_sentences(text: str, lang: str) -> list[str]:
    """'逃げろ！ここはもう持たない！' -> ['逃げろ！', 'ここはもう持たない！'] (ja/zh/ko split on
    。！？ only, so 'そうだな…優しいよね' stays whole); Latin text splits after .!?… + space."""
    text = text.strip()
    if not text:
        return [text]
    if iso(lang) in ("ja", "zh", "zh-hant", "ko"):
        parts = [p.strip() for p in _SENT_CJK.findall(text) if p.strip()]
    else:
        parts = [p.strip() for p in _SENT_LATIN.split(text) if p.strip()]
    out: list[str] = []
    for p in parts:  # a piece without letters ('！', '...') belongs to the previous one
        if out and not _LETTER.search(p):
            out[-1] += p
        else:
            out.append(p)
    return out or [text]


# --- output checks and clean-up ---------------------------------------------------------------

def repeating_unit(s: str) -> str:
    """'abcabca' -> 'abc' (shortest unit whose repetition rebuilds s)."""
    for i in range(1, len(s) // 2 + 1):
        unit = s[:i]
        if (unit * (len(s) // i + 1))[: len(s)] == s:
            return unit
    return s


_WORD = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)


def is_repetitive(text: str, max_run: int = 3, min_unique: float = 0.35) -> bool:
    """Repetition hallucination: a word/phrase looping, or a char run like 'aaaaaaa'."""
    if re.search(r"(.)\1{5,}", text):
        return True
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < 4:
        return False
    for n in (1, 2, 3):
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        run = 1
        for a, b in zip(grams, grams[n:]):
            run = run + 1 if a == b else 1
            if run >= max_run + (1 if n == 1 else 0):
                return True
    return len(words) >= 8 and len(set(words)) / len(words) < min_unique


def collapse_repetition(text: str, keep: int = 2) -> str:
    """Cut looping words/phrases down to `keep` copies ('Hei, Hei, Hei, ...' -> 'Hei, Hei')."""
    toks = re.findall(r"\S+\s*", text)

    def key(ts: list[str]) -> list[str]:
        return [re.sub(r"\W", "", t.lower()) for t in ts]

    for n in (3, 2, 1):
        out: list[str] = []
        i = 0
        while i < len(toks):
            unit = key(toks[i : i + n])
            reps = 1
            while len(unit) == n and key(toks[i + reps * n : i + (reps + 1) * n]) == unit:
                reps += 1
            if reps > keep:
                out += toks[i : i + keep * n]
                i += reps * n
            else:
                out.append(toks[i])
                i += 1
        toks = out
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", "".join(toks))
    return re.sub(r"[\s,]+$", "", text)


def is_bad(source: str, output: str, target: str) -> bool:
    """Post-translation check: empty, looping, still in the source script, or absurd length."""
    if not output.strip():
        return bool(source.strip())
    if _LETTER.search(source) and not _LETTER.search(output):
        return True  # punctuation for words ('うんうん' -> ',.'): finish() would leave nothing
    if is_repetitive(output) or foreign_ratio(output, target) > 0.2:
        return True
    return len(output) > 12 * max(len(source), 4)


def clean_output(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TO_LATIN_PUNCT)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\.\s*\.(?:\s*\.)+", "...", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"(?<![.,;!?])([,;!?])(?=[^\W\d_])", r"\1 ", text)
    text = re.sub(r"([!?])\1+", r"\1", text)
    return re.sub(r",{2,}", ",", text).strip(" ,")


def _cap(s: str) -> str:
    for i, c in enumerate(s):
        if c.isalpha():
            return s[:i] + c.upper() + s[i + 1 :]
    return s


def _decap(s: str, terms: list[str]) -> str:
    """Lower the first word unless it is a name/glossary term or an acronym ('Aku' -> 'aku')."""
    m = _WORD.match(s)
    if not m or m.group().isupper() and len(m.group()) > 1 or any(s.startswith(t) for t in terms):
        return s
    return s[0].lower() + s[1:]


def _set_end(text: str, end: str) -> str:
    """Manga bubbles end on the source's punctuation (or none), not on MT's default period."""
    body = re.sub(r"[.!?…]+$", "", text).rstrip(" ,")
    tail = text[len(body):].strip()
    if end in ("?", "!", "?!", "..."):
        return body + end
    if end == "":
        return body if tail in ("", ".") else text
    return text


# MT defaults to formal Indonesian (Saya/Anda); manga dialogue is casual (aku/kamu).
CASUAL: dict[str, list[tuple[str, str]]] = {
    "id": [(r"\bAnda\b", "kamu"), (r"\banda\b", "kamu"), (r"\bSaya\b", "Aku"), (r"\bsaya\b", "aku"),
           (r"\bKalian semua\b", "Kalian"), (r"\btidak apa-apa\b", "nggak apa-apa"),
           (r"\b([Nn])ama aku\b", r"\1amaku")],  # "Nama aku Sakura" -> "Namaku Sakura"
}


_LATIN_CHAR = r"[0-9A-Za-z\u00c0-\u024f]"


def has_term(text: str, term: str) -> bool:
    """Whole-word match against Latin neighbours only: 'Meiは' contains 'Mei'."""
    pat = rf"(?<!{_LATIN_CHAR}){re.escape(term)}(?!{_LATIN_CHAR})"
    return re.search(pat, text, re.IGNORECASE) is not None


def restore_terms(text: str, terms: Iterable[str], cutoff: float = 0.7) -> str:
    """Glossary enforcement after MT: a term the model respelled ('Mesia' -> 'Messias') is put
    back by fuzzy-matching output spans of the same word count."""
    import difflib

    for term in terms:
        if has_term(text, term):
            continue
        n = len(term.split())
        spans = list(re.finditer(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", text))
        best, best_ratio = None, cutoff
        for i in range(len(spans) - n + 1):
            a, b = spans[i].start(), spans[i + n - 1].end()
            r = difflib.SequenceMatcher(None, text[a:b].lower(), term.lower()).ratio()
            if r > best_ratio:
                best, best_ratio = (a, b), r
        if best:
            text = text[: best[0]] + term + text[best[1] :]
    return text


_TO_HAN_PUNCT = [(r"\.\.\.", "……"), (r"(?<![0-9A-Za-z]),\s*", "，"), (r"\?", "？"),
                  (r"!", "！"), (r"(?<![0-9A-Za-z.])\.$", "。")]


def han_punct(text: str) -> str:
    """Chinese text takes fullwidth punctuation ('你好?' -> '你好？')."""
    if not _HAN.search(text):
        return text
    for pat, repl in _TO_HAN_PUNCT:
        text = re.sub(pat, repl, text)
    return re.sub(r"\s*([，。？！])\s*", r"\1", text).replace("…… ", "……")


def finish(prep: Prepared, output: str, target: str, casual: bool = True) -> str:
    """Re-attach deterministic parts, enforce glossary spelling, fix punctuation and casing."""
    if prep.verbatim:
        return prep.verbatim
    text = _finish(prep, output, target, casual)
    if iso(target) in HAN_TARGETS:
        return han_punct(restore_kanji(text, prep.kanji, prep.text, iso(target) == "zh-hant"))
    return text


def _finish(prep: Prepared, output: str, target: str, casual: bool) -> str:
    out = clean_output(output) if output else ""
    if out and prep.terms:
        out = restore_terms(out, prep.terms)
    if out and casual:
        out = apply_dict(out, CASUAL.get(iso(target)))
    if out and prep.addressee and target not in CJK_LANGS:
        # "Senpai、好きです！" -> "Senpai aku suka!": the vocative keeps its comma
        out = re.sub(rf"^({re.escape(prep.addressee)}(?:-[a-z]+)?)(?=\s+[^\W\d_])", r"\1,", out)
    if prep.text and out and prep.lead:
        out = _decap(out, prep.terms)
    if out and prep.trail and re.search(r"[、，,]\s*$", prep.text) and not re.search(r"[,.!?…]$", out):
        out += ","  # "Yui-chan、" + "おはよう！" -> "Yui-chan, Selamat pagi!"
    if out and prep.trail.startswith(","):
        out = re.sub(r"\.+$", "", out)  # "lucu., gitu" -> "lucu, gitu"
    parts = [p for p in (prep.lead, out, prep.trail) if p]
    text = clean_output(" ".join(parts))
    if not text:
        return ""
    if prep.text or prep.end:
        text = _set_end(text, prep.end)
    if prep.stutter:
        text = stutter(text, target, prep.stutter_on in _PRONOUN_KANJI)
    return _cap(text)


def stutter(text: str, target: str, pronoun: bool = False) -> str:
    """'aku tidak' -> 'A-aku tidak'. A pronoun stutter (わ、私) goes on the pronoun when it is
    among the first words ('Maksudku, a-aku...'), otherwise on the first word."""
    words = list(_WORD.finditer(text))
    if not words:
        return text
    if iso(target) in CJK_LANGS and _CJK.match(words[0].group()):
        # no Latin 'X-X' in Chinese/Korean: '我、我不是' / '나, 나는'
        m = words[0]
        sep = ", " if iso(target) == "ko" else "、"
        return f"{text[: m.start()]}{m.group()[0]}{sep}{text[m.start():]}"
    m = words[0]
    if pronoun:
        names = PRONOUNS.get(iso(target), ())
        m = next((w for w in words[:3] if w.group().lower() in names), m)
    w = m.group()
    if not w[0].isalpha():
        return text
    first = w[0].upper() if m.start() == 0 or w == "I" else w[0].lower()
    rest = w if w == "I" else w[0].lower() + w[1:]
    return f"{text[: m.start()]}{first}-{rest}{text[m.end():]}"


# --- pre/post replacement dictionaries (reference: manga-image-translator pre_dict/post_dict) ---

def apply_dict(text: str, rules: dict[str, str] | Iterable[tuple[str, str]] | None) -> str:
    """Regex replacements, applied in order. An empty replacement deletes the match."""
    if not rules:
        return text
    items = rules.items() if isinstance(rules, dict) else rules
    for pattern, repl in items:
        text = re.sub(pattern, repl, text)
    return text


# --- chrF (Popovic 2015), corpus level, char n-grams 1..6, beta=2 -----------------------------

def _ngrams(s: str, n: int) -> Counter[str]:
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def chrf(hyps: list[str], refs: list[str], order: int = 6, beta: float = 2.0) -> float:
    """Corpus chrF in [0, 100]: n-gram counts summed over segments, P and R averaged over orders."""
    match = [0] * order
    hyp_total = [0] * order
    ref_total = [0] * order
    for h, r in zip(hyps, refs, strict=True):
        h, r = "".join(h.split()), "".join(r.split())
        for n in range(1, order + 1):
            hc, rc = _ngrams(h, n), _ngrams(r, n)
            match[n - 1] += sum((hc & rc).values())
            hyp_total[n - 1] += sum(hc.values())
            ref_total[n - 1] += sum(rc.values())
    precs = [m / t for m, t in zip(match, hyp_total) if t]
    recs = [m / t for m, t in zip(match, ref_total) if t]
    if not precs or not recs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return 100 * (1 + b2) * p * r / (b2 * p + r)
