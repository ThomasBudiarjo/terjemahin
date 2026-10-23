"""Prompts shared by every LLM translator.

The model always answers with one JSON object `{"items": [{"index", "source", "target",
"speaker"}]}`, indices 1..N in the order the lines (text mode) or crops (vision mode) were given.
The instructions are written in Indonesian when translating into Indonesian (the model then
"thinks" in the register it has to produce) and in English for every other target language.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ..models import GlossaryEntry, PageContext

Mode = Literal["text", "vision"]

_NAMES = {
    "ja": "Japanese", "id": "Indonesian", "en": "English", "zh": "Chinese (Simplified)",
    "zh-hant": "Chinese (Traditional)", "ko": "Korean", "ms": "Malay", "vi": "Vietnamese",
    "th": "Thai", "tl": "Tagalog", "fr": "French", "de": "German", "es": "Spanish",
    "pt": "Portuguese", "it": "Italian", "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
}
_NAMES_ID = {"ja": "Jepang", "id": "Indonesia", "en": "Inggris", "zh": "Mandarin",
             "zh-hant": "Mandarin (Tradisional)", "ko": "Korea", "ms": "Melayu"}


def lang_name(code: str) -> str:
    code = code.strip().lower()
    try:
        from .textproc import LANGS
    except ImportError:
        LANGS = {}
    if code in LANGS:
        return LANGS[code][2]
    return _NAMES.get(code, code)


def _indonesian(target: str) -> bool:
    return target.strip().lower() in ("id", "ind", "in")


# --- system prompts --------------------------------------------------------------------------

_ID_ROLE = """\
Kamu adalah penerjemah manga profesional {src} -> Indonesia{ocr_role}.

Pedoman terjemahan:
- Bahasa Indonesia yang natural, seperti manga terbitan resmi Indonesia. Jangan harfiah; \
utamakan maksud, emosi, dan ritme dialog.
- Ikuti register bicara karakter: santai/gaul (aku, kamu, nggak, sih, kok, dong) untuk \
obrolan teman, sopan/baku untuk situasi formal, kasar untuk karakter kasar.
- Pertahankan honorifik yang lazim (-san, -kun, -chan, -senpai, -sensei); nama ditulis \
dalam romaji (坂本 -> Sakamoto).
- Gunakan glossary secara konsisten. Gunakan konteks halaman sebelumnya untuk kesinambungan.
- Teks harus muat di balon: ringkas, hindari kalimat bertele-tele.
- Tanda baca ekspresif dipertahankan (…, !, ?, ?!). Gagap ditulis "A-aku".
- Onomatope/SFX: ekuivalen Indonesia jika ada (ドキドキ -> DEG DEG), jika tidak, \
transliterasi dalam huruf kapital.
- Jangan menambahkan penjelasan, catatan, atau teks sumber ke dalam terjemahan.
"""

_ID_TEXT = """\
Masukan: baris bernomor <|n|> berisi teks hasil OCR, satu baris per balon, urut sesuai urutan \
baca. OCR bisa salah baca satu-dua karakter; pahami maksudnya dari konteks.{image_note}
Untuk setiap baris, isi "target" dengan terjemahan, "speaker" dengan nama/deskripsi singkat \
pembicara jika jelas (selain itu null), dan "source" dengan teks sumber (perbaiki jika OCR \
jelas salah).
Jika sebuah baris bukan teks bermakna (sampah OCR), isi "target" dengan string kosong.
"""

_ID_VISION = """\
Masukan: gambar satu halaman manga utuh (konteks visual: ekspresi, situasi, siapa yang \
bicara), lalu crop setiap balon teks bernomor #1..#N sesuai urutan baca.
Untuk SETIAP crop: baca teks {src} apa adanya ke "source" (abaikan furigana), terjemahkan ke \
"target", dan isi "speaker" jika jelas (selain itu null).
Jika crop bukan teks (salah deteksi), isi "source" dan "target" dengan string kosong.
"""

_ID_FORMAT = """\
Keluarkan HANYA JSON valid, tanpa markdown, persis dengan bentuk:
{{"items": [{{"index": 1, "source": "...", "target": "...", "speaker": null}}, ...]}}
Satu item untuk setiap nomor masukan, index sama dengan nomor masukan, jangan digabung atau \
dilewati.
"""

_EN_ROLE = """\
You are a professional manga translator from {src} into {tgt}{ocr_role}.

Translation guidelines:
- Natural, fluent {tgt} as in an official localized release. Do not translate literally; \
keep the intent, emotion and rhythm of the dialogue.
- Match each character's register (casual, polite, rude, childish).
- Keep common honorifics (-san, -kun, -chan, -senpai, -sensei); write names in romaji.
- Use the glossary consistently. Use previous pages for continuity.
- Text must fit in the balloon: be concise.
- Keep expressive punctuation (..., !, ?, ?!). Write stutters as "I-I".
- Sound effects: use a natural {tgt} equivalent if one exists, otherwise transliterate in \
capitals.
- Never add notes, explanations or the source text to the translation.
"""

_EN_TEXT = """\
Input: numbered lines <|n|> with OCR text, one line per balloon, in reading order. OCR may \
misread a character or two; infer the meaning from context.{image_note}
For each line fill "target" with the translation, "speaker" with a short name/description of \
the speaker when it is clear (otherwise null), and "source" with the source text (corrected if \
the OCR is clearly wrong).
If a line is not meaningful text (OCR garbage), set "target" to an empty string.
"""

_EN_VISION = """\
Input: an image of the whole manga page (visual context: expressions, situation, who is \
speaking), followed by a crop of every text balloon numbered #1..#N in reading order.
For EVERY crop: transcribe the {src} text verbatim into "source" (ignore furigana), translate it \
into "target", and fill "speaker" when it is clear (otherwise null).
If a crop contains no text (false detection), set "source" and "target" to empty strings.
"""

_EN_FORMAT = """\
Output ONLY valid JSON, no markdown, exactly in this shape:
{{"items": [{{"index": 1, "source": "...", "target": "...", "speaker": null}}, ...]}}
One item for every input number, index equal to the input number; never merge or skip items.
"""


def system_prompt(
    mode: Mode, source_lang: str = "ja", target_lang: str = "id", page_image: bool = False
) -> str:
    if _indonesian(target_lang):
        src = _NAMES_ID.get(source_lang, lang_name(source_lang))
        role = _ID_ROLE.format(src=src, ocr_role=" sekaligus pembaca teks (OCR)" if mode == "vision" else "")
        note = (" Gambar halaman disertakan untuk konteks; posisi (x%, y%) tiap balon diberikan."
                if page_image else "")
        task = _ID_VISION.format(src=src) if mode == "vision" else _ID_TEXT.format(image_note=note)
        return f"{role}\n{task}\n{_ID_FORMAT.format()}"
    src, tgt = lang_name(source_lang), lang_name(target_lang)
    role = _EN_ROLE.format(src=src, tgt=tgt, ocr_role=" and text reader (OCR)" if mode == "vision" else "")
    note = (" The page image is attached for context; each line carries its balloon position "
            "(x%, y%)." if page_image else "")
    task = _EN_VISION.format(src=src) if mode == "vision" else _EN_TEXT.format(image_note=note)
    return f"{role}\n{task}\n{_EN_FORMAT.format()}"


# Kept for callers of the old single-prompt API.
SYSTEM_PROMPT = system_prompt("vision")


# --- user prompt -----------------------------------------------------------------------------

@dataclass
class Line:
    """One numbered input: OCR text (text mode) and an optional position / kind hint."""

    index: int
    text: str = ""
    pos: tuple[int, int] | None = None  # balloon centre in % of page width/height
    sfx: bool = False


def relevant_glossary(glossary: Sequence[GlossaryEntry], texts: Sequence[str], limit: int = 80) -> list[GlossaryEntry]:
    """Only the terms that occur in this page (all of them when there is no source text yet)."""
    joined = "".join(texts)
    if not joined.strip():
        return list(glossary)[:limit]
    hits = [g for g in glossary if g.source and g.source in joined]
    return hits[:limit]


def context_block(
    context: PageContext, texts: Sequence[str], target_lang: str, max_prev_pages: int = 2,
    max_prev_lines: int = 40,
) -> str:
    ind = _indonesian(target_lang)

    def L(id_text: str, en_text: str) -> str:
        return id_text if ind else en_text

    parts: list[str] = []
    if context.series_title:
        parts.append(f"{L('Judul seri', 'Series title')}: {context.series_title}")
    if context.series_summary:
        parts.append(f"{L('Ringkasan seri', 'Series summary')}: {context.series_summary}")
    if context.style_notes:
        parts.append(f"{L('Catatan gaya', 'Style notes')}: {context.style_notes}")
    gloss = relevant_glossary(context.glossary, texts)
    if gloss:
        lines = [f"- {g.source} -> {g.target}" + (f" ({g.note})" if g.note else "") for g in gloss]
        parts.append("Glossary:\n" + "\n".join(lines))
    if context.previous_pages and max_prev_pages > 0:
        recent = context.previous_pages[-max_prev_pages:]
        blocks = []
        for i, page in enumerate(recent):
            label = L("Halaman", "Page")
            blocks.append(f"[{label} -{len(recent) - i}]\n" + "\n".join(page[-max_prev_lines:]))
        head = L("Teks halaman sebelumnya (sudah diterjemahkan)", "Previous pages (already translated)")
        parts.append(f"{head}:\n" + "\n\n".join(blocks))
    return "\n\n".join(parts)


def text_task(lines: Sequence[Line], target_lang: str, source_lang: str = "ja") -> str:
    n = len(lines)
    if _indonesian(target_lang):
        head = (f"Terjemahkan {n} baris berikut dari bahasa {_NAMES_ID.get(source_lang, lang_name(source_lang))} "
                f"ke bahasa Indonesia. Kembalikan TEPAT {n} item dengan index "
                f"{_index_list(lines)}.")
    else:
        head = (f"Translate the following {n} lines from {lang_name(source_lang)} into "
                f"{lang_name(target_lang)}. Return EXACTLY {n} items with index {_index_list(lines)}.")
    body = []
    for ln in lines:
        tags = []
        if ln.pos:
            tags.append(f"{ln.pos[0]}%,{ln.pos[1]}%")
        if ln.sfx:
            tags.append("SFX")
        tag = f"({'; '.join(tags)}) " if tags else ""
        body.append(f"<|{ln.index}|> {tag}{' '.join(ln.text.split())}")
    tail = ("Tulis semua \"target\" dalam bahasa Indonesia." if _indonesian(target_lang)
            else f"Write every \"target\" in {lang_name(target_lang)}.")
    return head + "\n\n" + "\n".join(body) + "\n\n" + tail


def vision_task(indices: Sequence[int], target_lang: str, has_page: bool = True) -> str:
    n = len(indices)
    idx = _index_list([Line(i) for i in indices])
    if _indonesian(target_lang):
        page = "Gambar pertama adalah halaman utuh, diikuti" if has_page else "Berikut"
        return (f"Halaman ini memiliki {n} balon teks. {page} crop balon {idx}. "
                f"Kembalikan TEPAT {n} item dengan index {idx}.")
    page = "The first image is the whole page, followed by" if has_page else "Here are"
    return (f"This page has {n} text balloons. {page} the crops {idx}. "
            f"Return EXACTLY {n} items with index {idx}.")


def language_note(target_lang: str) -> str:
    if _indonesian(target_lang):
        return ("PENTING: jawaban sebelumnya bukan bahasa Indonesia. Tulis SEMUA \"target\" dalam "
                "bahasa Indonesia sehari-hari, bukan bahasa Inggris.")
    return (f"IMPORTANT: the previous answer was in the wrong language. Write EVERY \"target\" in "
            f"{lang_name(target_lang)}.")


def repair_note(missing: Sequence[int], target_lang: str) -> str:
    idx = ", ".join(map(str, missing))
    if _indonesian(target_lang):
        return (f"PENTING: jawaban sebelumnya tidak lengkap. Kembalikan TEPAT {len(missing)} item, "
                f"satu untuk setiap index berikut: {idx}.")
    return (f"IMPORTANT: the previous answer was incomplete. Return EXACTLY {len(missing)} items, "
            f"one for each of these indices: {idx}.")


def crop_label(index: int, target_lang: str) -> str:
    return f"{'Crop balon' if _indonesian(target_lang) else 'Balloon crop'} #{index}:"


def _index_list(lines: Sequence[Line]) -> str:
    idx = [ln.index for ln in lines]
    if idx and idx == list(range(idx[0], idx[0] + len(idx))) and len(idx) > 2:
        return f"{idx[0]}..{idx[-1]}"
    return ", ".join(map(str, idx))


def build_user_prompt(context: PageContext, n_regions: int, target_lang: str = "id") -> str:
    """Old single-call API (vision mode, indices 1..N)."""
    ctx = context_block(context, [], target_lang)
    task = vision_task(list(range(1, n_regions + 1)), target_lang)
    return f"{ctx}\n\n{task}" if ctx else task
