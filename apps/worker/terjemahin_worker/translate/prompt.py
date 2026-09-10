from __future__ import annotations

from ..models import PageContext

SYSTEM_PROMPT = """\
Kamu adalah penerjemah manga profesional Jepang -> Indonesia sekaligus pembaca teks (OCR).

Kamu menerima:
1. Gambar satu halaman manga utuh (untuk konteks visual: ekspresi, situasi, siapa yang bicara).
2. Potongan (crop) setiap balon teks, diberi nomor sesuai urutan baca (kanan ke kiri, atas ke bawah).
3. Glossary istilah/nama dan teks halaman sebelumnya, jika ada.

Untuk SETIAP crop, lakukan:
- Baca teks Jepang apa adanya (source). Jika ada furigana, abaikan furigana-nya.
- Terjemahkan ke bahasa Indonesia yang natural, seperti manga terbitan resmi Indonesia:
  * Ikuti register bicara karakter (kasar/sopan/anak-anak/formal).
  * Pertahankan honorifik hanya jika lazim (-san, -kun, -chan, senpai); selain itu adaptasi.
  * Jangan terjemahkan harfiah; utamakan maksud dan ritme dialog.
  * Gunakan glossary secara konsisten. Gunakan konteks halaman sebelumnya untuk kesinambungan.
  * Teks harus muat di balon: ringkas, hindari kalimat bertele-tele.
  * Onomatope/SFX: transliterasi ke ekuivalen Indonesia jika ada (misal ドキドキ -> DEG DEG), \
jika tidak, tulis dalam huruf latin kapital.
- Jika crop bukan teks (salah deteksi), isi source dan target dengan string kosong.

Keluarkan JSON persis dengan skema yang diminta, satu item per crop, urut sesuai nomor.
"""


def build_user_prompt(context: PageContext, n_regions: int) -> str:
    parts: list[str] = []
    if context.series_title:
        parts.append(f"Judul seri: {context.series_title}")
    if context.series_summary:
        parts.append(f"Ringkasan seri: {context.series_summary}")
    if context.style_notes:
        parts.append(f"Catatan gaya: {context.style_notes}")
    if context.glossary:
        lines = [
            f"- {g.source} -> {g.target}" + (f" ({g.note})" if g.note else "")
            for g in context.glossary
        ]
        parts.append("Glossary:\n" + "\n".join(lines))
    if context.previous_pages:
        recent = context.previous_pages[-2:]
        blocks = []
        for i, page in enumerate(recent, start=1):
            blocks.append(f"[Halaman -{len(recent) - i + 1}]\n" + "\n".join(page))
        parts.append("Teks halaman sebelumnya (Indonesia):\n" + "\n\n".join(blocks))
    parts.append(
        f"Halaman ini memiliki {n_regions} balon teks. Gambar pertama adalah halaman utuh, "
        f"diikuti crop balon #0 sampai #{n_regions - 1}."
    )
    return "\n\n".join(parts)
