"""Indonesian / Malay syllabification for line-end hyphenation.

Indonesian spelling is close to phonemic, so the KBBI rules can be applied directly:
V-CV (a-nak), VC-CV (ban-tu), VC-CCV (in-stru-men), V-V (sa-at), with the digraphs ng, ny,
kh, sy treated as one consonant and the diphthongs ai, au, oi, ei never split. That is good
enough for loanwords and names too, and needs no dictionary.
"""

from __future__ import annotations

import re
from itertools import pairwise

VOWELS = set("aeiouAEIOUàáâäèéêëìíîïòóôöùúûüÀÁÂÄÈÉÊËÌÍÎÏÒÓÔÖÙÚÛÜ")
DIGRAPHS = ("ng", "ny", "kh", "sy")
DIPHTHONGS = ("ai", "au", "oi", "ei")
MIN_PART = 3  # letters on each side of a hyphen: "re-aksinya" and "gima-na" look broken
MIN_WORD = 6

_WORD = re.compile(r"^([^\w]*)(\w+?)([^\w]*)$", re.UNICODE)


def _units(word: str) -> list[str]:
    """Letters grouped so that digraphs count as one consonant."""
    out: list[str] = []
    i = 0
    while i < len(word):
        pair = word[i : i + 2].lower()
        if pair in DIGRAPHS and i + 2 <= len(word):
            out.append(word[i : i + 2])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def _is_v(unit: str) -> bool:
    return len(unit) == 1 and unit in VOWELS


def syllables(word: str) -> list[str]:
    """Split one alphabetic word into syllables; ''.join(result) == word."""
    units = _units(word)
    if not any(_is_v(u) for u in units):
        return [word]
    vowel_idx = [i for i, u in enumerate(units) if _is_v(u)]
    cuts: list[int] = []  # unit index where a new syllable starts
    for a, b in pairwise(vowel_idx):
        n_cons = b - a - 1
        if n_cons == 0:
            if (units[a] + units[b]).lower() not in DIPHTHONGS:
                cuts.append(b)
        elif n_cons == 1:
            cuts.append(a + 1)
        else:
            cuts.append(a + 2)
    parts, prev = [], 0
    for c in cuts:
        parts.append("".join(units[prev:c]))
        prev = c
    parts.append("".join(units[prev:]))
    return [p for p in parts if p]


def split_points(word: str) -> list[str]:
    """Pieces of `word` (with its punctuation) that may be broken across lines with a '-'.

    Keeps the leading/trailing punctuation attached to the first/last piece and refuses pieces
    shorter than MIN_PART letters, so "anak" never becomes "a-nak".
    """
    m = _WORD.match(word)
    if not m or len(m.group(2)) < MIN_WORD or not m.group(2).isalpha():
        return [word]
    lead, core, tail = m.groups()
    sy = syllables(core)
    merged: list[str] = []
    for s in sy:
        if merged and len(merged[-1]) < MIN_PART:
            merged[-1] += s
        else:
            merged.append(s)
    if len(merged) > 1 and len(merged[-1]) < MIN_PART:
        last = merged.pop()
        merged[-1] += last
    merged[0] = lead + merged[0]
    merged[-1] = merged[-1] + tail
    return merged


def hyphenate(word: str) -> str:
    """Debug helper: 'menyenangkan' -> 'me-nye-nang-kan'."""
    return "-".join(split_points(word))
