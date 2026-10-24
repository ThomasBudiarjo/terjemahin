"""Stage metrics for the benchmark: pure functions over texts, images and masks.

Ground truth has text per bubble but no boxes, so detections are matched to GT bubbles by what
OCR read in them (Hungarian assignment on CER). Inpainting and typesetting are judged from
pixels only, so every registered method is scored the same way whatever it does inside:
rendered text is read back from the drawn ink by an independent OCR, never taken from the
renderer's own account of what it drew.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise

import cv2
import numpy as np

from ..models import BBox, Bubble

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿ｦ-ﾟ]")
ELLIPSIS = re.compile(r"[・･…‥.．。]{2,}|[…‥]")
PUNCT = re.compile(r"[\s\W_ー〜～]+")

# -- text ---------------------------------------------------------------------------------


def normalize_ja(text: str, punct: bool = True) -> str:
    """NFKC, no whitespace, every ellipsis spelling ("・・", "…", "...") as one "…".

    With `punct=False` all punctuation and long-vowel marks go too: hand transcriptions and OCR
    disagree on those far more than on the words.
    """
    t = unicodedata.normalize("NFKC", text)
    t = ELLIPSIS.sub("…", t)
    t = "".join(t.split())
    return t if punct else PUNCT.sub("", t)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    """Character error rate of normalised strings (may exceed 1 for long hallucinations)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(hyp, ref) / len(ref)


def cjk_ratio(text: str) -> float:
    letters = [c for c in text if c.isalnum()]
    return sum(bool(CJK.match(c)) for c in letters) / len(letters) if letters else 0.0


def is_japanese(text: str) -> bool:
    return cjk_ratio(text) >= 0.3


PUNCTS = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _words(text: str) -> list[str]:
    """chrF++ word tokens: one leading or trailing ASCII punctuation mark split off a word."""
    out: list[str] = []
    for w in text.split():
        if len(w) > 1 and w[-1] in PUNCTS:
            out += [w[:-1], w[-1]]
        elif len(w) > 1 and w[0] in PUNCTS:
            out += [w[0], w[1:]]
        else:
            out.append(w)
    return out


def _ngrams(seq: str | list[str], n: int) -> Counter:
    return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))


def _match(hc: Counter, rc: Counter) -> list[int]:
    # as sacrebleu: hypothesis n-grams of an order the reference lacks are not counted
    nr = sum(rc.values())
    return [sum(hc.values()) if nr else 0, nr, sum((hc & rc).values())]


def chrf_stats(hyp: str, ref: str, char_order: int = 6, word_order: int = 0) -> list[int]:
    """[hyp n-grams, ref n-grams, matches] per order (chars without spaces, then words)."""
    stats: list[int] = []
    h, r = "".join(hyp.split()), "".join(ref.split())
    for n in range(1, char_order + 1):
        stats += _match(_ngrams(h, n), _ngrams(r, n))
    hw, rw = _words(hyp), _words(ref)
    for n in range(1, word_order + 1):
        stats += _match(_ngrams(hw, n), _ngrams(rw, n))
    return stats


def chrf_score(stats: list[int], beta: float = 2.0) -> float:
    """sacrebleu's default chrF (as chrF++.py): precision and recall averaged over the n-gram
    orders both sides have, then one F-beta."""
    prec = rec = 0.0
    orders = 0
    for i in range(0, len(stats), 3):
        nh, nr, nm = stats[i : i + 3]
        if nh == 0 or nr == 0:
            continue
        orders += 1
        prec += nm / nh
        rec += nm / nr
    if not orders:
        return 0.0
    prec, rec = prec / orders, rec / orders
    b2 = beta * beta
    return 100.0 * (1 + b2) * prec * rec / (b2 * prec + rec) if prec + rec > 0 else 0.0


def chrf(hyps: list[str], refs: list[str], word_order: int = 0, beta: float = 2.0) -> float:
    """Corpus chrF (word_order=2 gives chrF++): statistics are summed before scoring."""
    acc: list[int] | None = None
    for h, r in zip(hyps, refs, strict=True):
        s = chrf_stats(h, r, word_order=word_order)
        acc = s if acc is None else [a + b for a, b in zip(acc, s)]
    return chrf_score(acc, beta) if acc else 0.0


# -- meaning conflicts (Indonesian) -------------------------------------------------------
#
# Sentence embeddings are close to bags of words: "Kau tidak akan pernah memaafkanku" and
# "Aku nggak akan pernah memaafkanmu" score 0.94, "Aku marah pada guru" and "Aku dimarahi
# guru" 0.84. These checks catch what they miss: a flipped negation, swapped or invented
# participants, and an active verb where the reference has its passive (or the reverse).

NEGATION = frozenset(
    [
        "tidak", "tak", "nggak", "ngga", "enggak", "engga", "gak", "ga", "kagak", "ndak", "bukan",
        "bukannya", "jangan", "belum", "tanpa", "takkan", "tiada"
    ]
)
FIRST = frozenset(["aku", "saya", "gue", "gua", "gw", "kami", "kita", "daku"])
SECOND = frozenset(["kau", "kamu", "anda", "engkau", "lo", "lu", "elo", "elu", "kalian", "dikau"])
THIRD = frozenset(["dia", "ia", "beliau", "mereka"])
# words ending in -ku / -mu that carry no pronoun
_KU_WORDS = frozenset(
    [
        "buku", "laku", "kuku", "saku", "suku", "paku", "baku", "taku", "liku", "siku"
    ]
)
_MU_WORDS = frozenset(["ilmu", "jamu", "tamu", "temu", "jemu", "ramu", "kamu", "semu"])
# di- words that are no passive verbs
_DI_WORDS = frozenset(
    [
        "dia", "diri", "dini", "dingin", "dimana", "disini", "disana", "disitu", "dikit", "diam",
        "diamlah", "dinding", "dirinya", "dirimu", "diriku", "dinas", "dina", "dia-dia", "dijah",
        "digital", "dilema", "dimensi", "dinasti", "diet", "direktur", "diskon", "diskusi", "dosa"
    ]
)
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")
_VERB_PREFIXES = ("me", "di", "ber", "ter", "ng")


def _id_words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def persons(text: str, standalone: bool = False) -> list[int]:
    """Grammatical persons in order of first mention: 1 (aku, -ku, ku-), 2 (kau, kamu, -mu,
    kau-), 3 (dia, mereka, -nya on a verb). `standalone`: pronoun words only, the usual
    subjects (an object suffix such as memaafkan-mu is often only implied in Japanese)."""
    seen: list[int] = []

    def add(p: int) -> None:
        if p not in seen:
            seen.append(p)

    for w in _id_words(text):
        base = w.split("-")[0]
        if base in FIRST:
            add(1)
        elif base in SECOND:
            add(2)
        elif base in THIRD:
            add(3)
        elif standalone:
            continue
        elif len(w) > 4 and w.startswith("ku") and w[2] not in "aiueo" and w not in _KU_WORDS:
            add(1)  # kubilang, kuaduin
        elif len(w) > 5 and w.startswith("kau"):
            add(2)  # kaulihat
        elif len(w) > 3 and w.endswith("ku") and w not in _KU_WORDS:
            add(1)  # namaku, memaafkanku
        elif len(w) > 3 and w.endswith("mu") and w not in _MU_WORDS:
            add(2)  # menurutmu, membuatmu
        elif len(w) > 6 and w.endswith("nya") and w.startswith(_VERB_PREFIXES):
            add(3)  # menyukainya, memukulnya (on a noun, -nya is mostly "the": skipped)
    return seen


def _negated(text: str) -> bool:
    return any(w in NEGATION for w in _id_words(text))


def _passive_roots(words: list[str]) -> set[str]:
    out = set()
    for w in words:
        if len(w) >= 5 and w.startswith("di") and w not in _DI_WORDS and "-" not in w:
            root = w[2:]
            for suf in ("kannya", "nya", "kan", "in", "i"):
                if root.endswith(suf) and len(root) - len(suf) >= 3:
                    root = root[: -len(suf)]
                    break
            out.add(root)
    return out


def _active_use(root: str, words: list[str]) -> bool:
    """`root` used without the passive di- (bare or with an active prefix)."""
    for w in words:
        if w.startswith(("di", "ter")):
            continue
        if w == root or (w.startswith(root) and len(w) - len(root) <= 3):
            return True
        for pre in ("me", "mem", "men", "meng", "meny", "ng", "nge"):
            if w.startswith((pre + root, pre + root[1:])):
                return True
    return False


def meaning_conflicts(hyp: str, ref: str) -> list[str]:
    """Reasons a translation means something else than its (Indonesian) reference, beyond
    what an embedding sees; [] when none is found. Heuristic and deliberately narrow."""
    out = []
    if not hyp.strip() or not ref.strip():
        return out
    if _negated(hyp) != _negated(ref):
        out.append("negation")
    ph, pr = persons(hyp), persons(ref)
    sh, sr = set(ph), set(pr)
    if (sh - sr) and (sr - sh):
        out.append("participants")  # someone else does or undergoes it
    elif (set(persons(hyp, standalone=True)) - sr) & {1, 2} and sr & {1, 2}:
        out.append("participants")  # an "I" or "you" the reference does not have
    elif len(sh & sr) >= 2:
        common = [p for p in ph if p in sr]
        if common != [p for p in pr if p in sh]:
            out.append("roles")  # the same people, the other way round
    wh, wr = _id_words(hyp), _id_words(ref)
    if any(_active_use(r, wh) for r in _passive_roots(wr) - _passive_roots(wh)) or any(
        _active_use(r, wr) for r in _passive_roots(wh) - _passive_roots(wr)
    ):
        out.append("voice")
    return out


# -- assignment ---------------------------------------------------------------------------


def hungarian(cost: np.ndarray | list[list[float]]) -> list[tuple[int, int]]:
    """Minimum-cost one-to-one assignment for a rectangular matrix, as sorted (row, col) pairs.

    Jonker-Volgenant style shortest augmenting paths, O(n^2 m); scipy is not a dependency and
    pages have tens of regions at most.
    """
    c = np.asarray(cost, dtype=np.float64)
    if c.size == 0:
        return []
    flip = c.shape[0] > c.shape[1]
    if flip:
        c = c.T
    n, m = c.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)  # p[j]: row assigned to column j (1-based, 0 = free)
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = c[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j]]
    return sorted((b, a) for a, b in pairs) if flip else sorted(pairs)


@dataclass
class Match:
    det: int  # index into the detections
    gt: int  # index into the GT bubbles
    cer: float  # normalised, punctuation kept
    cer_text: float  # punctuation and long-vowel marks ignored
    coverage: float = 1.0  # share of the GT text (no punctuation) the read contains, in order


@dataclass
class DetectionScore:
    matches: list[Match]
    n_gt: int
    n_det: int
    sfx: list[int] = field(default_factory=list)  # detections that read a listed SFX
    fragments: list[int] = field(default_factory=list)  # pieces of an already matched bubble
    duplicates: list[int] = field(default_factory=list)  # the same bubble detected again
    false_positives: list[int] = field(default_factory=list)
    missed: list[int] = field(default_factory=list)  # GT bubbles nothing matched

    @property
    def partial(self) -> list[Match]:
        """Matches whose box read less than PARTIAL_COVERAGE of the bubble's text: the
        detector boxed part of it ("クン" for "ドクン", "はー" for "はーい"), so the rest of the
        original stays on the page next to the translation."""
        return [m for m in self.matches if m.coverage < PARTIAL_COVERAGE]

    @property
    def mean_cer(self) -> float | None:
        return float(np.mean([m.cer for m in self.matches])) if self.matches else None

    @property
    def mean_cer_text(self) -> float | None:
        return float(np.mean([m.cer_text for m in self.matches])) if self.matches else None


PARTIAL_COVERAGE = 0.75


def lcs(a: str, b: str) -> int:
    """Length of the longest common subsequence."""
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def coverage(hyp: str, ref: str) -> float:
    """Share of `ref` (normalised, no punctuation) found in order in `hyp`; 1 when the read
    is not shorter than the reference (a misread, not a cut-off box)."""
    h, r = normalize_ja(hyp, punct=False), normalize_ja(ref, punct=False)
    if not r or len(h) >= len(r):
        return 1.0
    return lcs(h, r) / len(r)


def _contained(part: str, whole: str) -> bool:
    """Most of `part` appears in order inside `whole` (a split bubble's other half)."""
    if len(part) < 2:
        return False
    return len(part) - (levenshtein(part, whole) - (len(whole) - len(part))) >= 0.8 * len(part)


def overlap_small(a: BBox, b: BBox) -> float:
    """Intersection over the smaller box: 1 when one box lies inside the other."""
    iw = min(a.x2, b.x2) - max(a.x1, b.x1)
    ih = min(a.y2, b.y2) - max(a.y1, b.y1)
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(1, min(a.w * a.h, b.w * b.h))


def match_detections(
    det_texts: list[str],
    gt_texts: list[str],
    sfx: list[str] = (),
    max_cer: float = 0.5,
    boxes: list[BBox] | None = None,
) -> DetectionScore:
    """Assign detections to GT bubbles by OCR text; a pair counts when its CER <= max_cer.

    Unmatched detections are SFX hits (if they read a listed SFX), duplicates (they read the
    same text as, or mostly overlap, the detection matched to that bubble), fragments (a bubble
    split in two, the other half matched), or false positives.
    """
    hyp = [normalize_ja(t, punct=False) for t in det_texts]
    ref = [normalize_ja(t, punct=False) for t in gt_texts]
    cost = np.array([[min(cer(h, r), 2.0) for r in ref] for h in hyp]).reshape(len(hyp), len(ref))
    matches = [
        Match(
            d, g, cer(normalize_ja(det_texts[d]), normalize_ja(gt_texts[g])), float(cost[d, g]),
            coverage(det_texts[d], gt_texts[g]),
        )
        for d, g in hungarian(cost)
        if cost[d, g] <= max_cer
    ]
    score = DetectionScore(matches, len(gt_texts), len(det_texts))
    matched_d = {m.det for m in matches}
    matched_g = {m.gt for m in matches}
    sfx_n = [normalize_ja(s, punct=False) for s in sfx]

    def duplicate(d: int) -> bool:
        for m in matches:
            if cer(hyp[d], hyp[m.det]) <= 0.2 or (
                boxes is not None and overlap_small(boxes[d], boxes[m.det]) >= 0.5
            ):
                return True
        return False

    for d, h in enumerate(hyp):
        if d in matched_d:
            continue
        if any(cer(h, s) <= max_cer for s in sfx_n):
            score.sfx.append(d)
        elif duplicate(d):
            score.duplicates.append(d)
        elif any(_contained(h, ref[g]) for g in matched_g):
            score.fragments.append(d)
        else:
            score.false_positives.append(d)
    score.missed = [g for g in range(len(gt_texts)) if g not in matched_g]
    return score


# -- inpainting ---------------------------------------------------------------------------


def outside_mask_change(
    original: np.ndarray, inpainted: np.ndarray, mask: np.ndarray, ring: int = 8
) -> dict:
    """How much the inpainter touched pixels it was not asked to: all must be ~0.

    `far_changed` counts changes more than `ring` px from the mask, which no edge feathering or
    hole dilation explains (a whole-page resize or colour shift would show up there).
    """
    keep = mask == 0
    if not keep.any():
        return {"mad": 0.0, "max": 0, "changed_frac": 0.0, "far_changed": 0}
    diff = np.abs(original.astype(np.int16) - inpainted.astype(np.int16)).max(axis=2)
    k = np.ones((2 * ring + 1,) * 2, np.uint8)
    far = cv2.dilate((mask > 0).astype(np.uint8), k) == 0
    d = diff[keep]
    return {
        "mad": float(d.mean()),
        "max": int(d.max()),
        "changed_frac": float((d > 0).mean()),
        "far_changed": int((diff[far] > 0).sum()),
    }


def box_overlap(mask: np.ndarray, b: BBox) -> float:
    """Share of the box's pixels that are set in `mask`."""
    H, W = mask.shape[:2]
    c = b.clamp(W, H)
    win = mask[c.y1 : c.y2, c.x1 : c.x2]
    return float((win > 0).mean()) if win.size else 0.0


def dark_residue(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
    boxes: list[BBox],
    dark: int = 110,
    ghost: int = 40,
) -> dict:
    """Glyph remains in the erased part of closed light balloons.

    The balloon interior is re-found on the inpainted page; where text was erased nothing dark
    (`inpainted`) and nothing visibly off the interior tone (`ghost`: faint grey ghosts that the
    detector can still read) should remain. `original` is the "nothing erased" baseline.
    """
    from ..render.balloon import find_balloon

    g_in = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY).astype(np.int16)
    g_or = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    n = n_dark = n_ghost = n_orig = 0
    worst = 0.0
    seen = np.zeros(mask.shape, bool)  # balloons shared by two boxes count once
    for b in boxes:
        bal = find_balloon(inpainted, original, b)
        if not bal.closed or sum(bal.color) / 3 < 180:
            continue
        ys, xs = bal.page_slice()
        inner = cv2.erode(bal.mask, np.ones((7, 7), np.uint8)) > 0
        if not inner.any():
            continue
        gi = g_in[ys, xs]
        tone = int(np.median(gi[inner]))
        zone = inner & (mask[ys, xs] > 0) & ~seen[ys, xs]
        seen[ys, xs] |= zone
        k = int(zone.sum())
        if not k:
            continue
        g = int((np.abs(gi[zone] - tone) > ghost).sum())
        n, n_ghost = n + k, n_ghost + g
        n_dark += int((gi[zone] < dark).sum())
        n_orig += int((g_or[ys, xs][zone] < dark).sum())
        worst = max(worst, g / k)
    if not n:
        return {"pixels": 0, "inpainted": 0.0, "ghost": 0.0, "ghost_max": 0.0, "original": 0.0}
    return {
        "pixels": n,
        "inpainted": n_dark / n,
        "ghost": n_ghost / n,
        "ghost_max": worst,  # worst balloon: one ghosted bubble must not hide in the page mean
        "original": n_orig / n,
    }


# -- rendering ----------------------------------------------------------------------------

INK_THRESHOLD = 40  # max channel difference between final and inpainted that counts as ink
TOLERANCE_PX = 3  # anti-aliasing / outline slack around a container
LOST_CER = 0.5  # read-back CER above which a bubble's text counts as lost

# reads the page's rendered text back: (image, per bubble a list of line quads) -> texts
Reader = Callable[[np.ndarray, list[list[np.ndarray]]], list[str]]


@dataclass
class RenderBox:
    bubble: int
    closed: bool  # container is a closed balloon (else the padded detector box)
    ink_px: int
    outside_px: int
    ink_bbox: BBox | None
    bbox_inside: bool
    target_chars: int
    rendered_chars: int | None  # from the renderer's layout when it exposes one
    font_px: float | None
    font_source: str  # "layout" or "estimate"
    fits: bool | None
    read_text: str | None = None  # what an independent OCR reads in the drawn ink
    read_cer: float | None = None  # of read_text against the translation, letters only

    @property
    def inside_frac(self) -> float:
        return 1.0 - self.outside_px / self.ink_px if self.ink_px else 1.0

    @property
    def read_chars(self) -> int | None:
        return count_chars(self.read_text) if self.read_text is not None else None

    @property
    def lost(self) -> bool:
        return self.target_chars > 0 and (
            self.ink_px == 0
            or (self.rendered_chars is not None and self.rendered_chars < self.target_chars)
            or (self.read_cer is not None and self.read_cer > LOST_CER)
        )


def count_chars(text: str) -> int:
    """Letters and digits only: renderers legitimately respell punctuation and hyphenate."""
    return sum(c.isalnum() for c in unicodedata.normalize("NFKC", text))


def letters(text: str) -> str:
    """Case-folded letters and digits: what read-back is compared on (renderers may hyphenate,
    upper-case or respell punctuation)."""
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def _runs(on: np.ndarray) -> list[list[int]]:
    d = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(np.int8), [0]])))
    return [[int(a), int(b)] for a, b in zip(d[::2], d[1::2])]


def ink_lines(ink: np.ndarray, pad: int = 2) -> list[np.ndarray]:
    """Horizontal text-line quads of a 0/1 ink mask, from its row profile.

    Thin bands (i-dots, accents, commas cut off by the ink threshold) join their nearest line.
    Tightly set lines touch (descenders meet ascenders), so a band is also cut wherever the
    profile has two dense x-height cores with a near-empty valley between them. Used to read
    rendered text back with our own line boxes, not the renderer's.
    """
    prof = ink.sum(1).astype(np.float64)
    bands = _runs(prof > 0)
    if not bands:
        return []
    hs = np.array([e - s for s, e in bands])
    line_h = float(np.median(hs[hs >= 0.5 * hs.max()]))
    while len(bands) > 1:
        k = min(range(len(bands)), key=lambda i: bands[i][1] - bands[i][0])
        if bands[k][1] - bands[k][0] >= 0.45 * line_h:
            break
        gaps = [
            bands[k][0] - bands[k - 1][1] if k > 0 else np.inf,
            bands[k + 1][0] - bands[k][1] if k + 1 < len(bands) else np.inf,
        ]
        j = k - 1 if gaps[0] <= gaps[1] else k + 1
        a, b = sorted((j, k))
        bands[a : b + 1] = [[bands[a][0], bands[b][1]]]
    out: list[list[int]] = []
    for s, e in bands:
        p = prof[s:e]
        cores = _runs(p >= 0.4 * p.max())
        tall = max(b - a for a, b in cores)
        cores = [c for c in cores if c[1] - c[0] >= 0.3 * tall]
        cuts = [s]
        for (_, z), (a, _) in pairwise(cores):
            if a > z and p[z:a].min() <= 0.25 * p.max():
                cuts.append(s + z + int(np.argmin(p[z:a])))
        out += [[a, b] for a, b in zip(cuts, cuts[1:] + [e])]
    H, W = ink.shape
    quads = []
    for s, e in out:
        cols = np.flatnonzero(ink[s:e].any(0))
        if not len(cols):
            continue
        x1, y1 = max(0, int(cols[0]) - pad), max(0, s - pad)
        x2, y2 = min(W, int(cols[-1]) + 1 + pad), min(H, e + pad)
        quads.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32))
    return quads


def open_container(b: BBox, W: int, H: int) -> BBox:
    """Where text without a closed balloon may go: the box, padded, and at least as wide as
    tall (a vertical Japanese column becomes horizontal Latin lines)."""
    pad = max(10, round(0.2 * max(b.w, b.h)))
    grow = max(0, b.h - b.w) // 2
    return BBox(x1=b.x1 - pad - grow, y1=b.y1 - pad, x2=b.x2 + pad + grow, y2=b.y2 + pad).clamp(
        W, H
    )


def containers(
    inpainted: np.ndarray, original: np.ndarray, bubbles: list[Bubble]
) -> tuple[list[np.ndarray], list[bool]]:
    """Per bubble, a page-sized bool mask of where its text may be drawn."""
    from ..render.balloon import find_balloon, split_shared

    H, W = inpainted.shape[:2]
    boxes = [b.region.bbox for b in bubbles]
    balloons = [find_balloon(inpainted, original, bx) for bx in boxes]
    split_shared(balloons, boxes)
    k = np.ones((2 * TOLERANCE_PX + 1,) * 2, np.uint8)
    out, closed = [], []
    for bx, bal in zip(boxes, balloons):
        m = np.zeros((H, W), np.uint8)
        if bal.closed:
            ys, xs = bal.page_slice()
            m[ys, xs] = bal.mask
        else:
            c = open_container(bx, W, H)
            m[c.y1 : c.y2, c.x1 : c.x2] = 255
        out.append(cv2.dilate(m, k) > 0)
        closed.append(bal.closed)
    return out, closed


def _estimate_font(ink: np.ndarray) -> float | None:
    """Median text-line height from the row profile: about one em for Latin text."""
    rows = ink.any(axis=1).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], rows, [0]])))
    runs = edges[1::2] - edges[::2]
    runs = runs[runs >= 4]
    return float(np.median(runs)) if len(runs) else None


def reading_canvas(inpainted: np.ndarray, final: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Only what the renderer drew, as dark-on-white: pixels it darkened near the ink keep their
    final value (glyphs and their anti-aliased edges), everything else is white, so neither the
    art behind the text nor a light outline (halo) over it can help or hurt the read."""
    near = cv2.dilate(ink, np.ones((3, 3), np.uint8)) > 0
    g_in = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY).astype(np.int16)
    g_fi = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY).astype(np.int16)
    keep = near & (g_fi < g_in - 16)
    out = np.full_like(final, 255)
    out[keep] = (255 - (g_in - g_fi)[keep, None].clip(0, 255)).astype(np.uint8)
    return out


def render_check(
    inpainted: np.ndarray,
    final: np.ndarray,
    original: np.ndarray,
    bubbles: list[Bubble],
    plans: list | None = None,
    reader: Reader | None = None,
) -> tuple[list[RenderBox], np.ndarray]:
    """Attribute every drawn pixel to a bubble and check it stays in that bubble's container.

    Ink = pixels the renderer changed. Each 8-connected ink component goes to the bubble whose
    container holds most of it (else the nearest box). `reader` reads each bubble's ink back
    (on a blank canvas, so neither art nor the neighbours' text can help or hurt) to prove the
    whole translation was drawn legibly. `plans` (renderers that expose their layouts, like the
    default one) add the layout's characters and font sizes.
    Returns the per-bubble results and a debug overlay.
    """
    diff = np.abs(final.astype(np.int16) - inpainted.astype(np.int16)).max(axis=2)
    ink = (diff > INK_THRESHOLD).astype(np.uint8)
    conts, closed = containers(inpainted, original, bubbles)
    n, labels, _, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
    owner = np.full(n, -1, np.int64)
    if bubbles and n > 1:
        flat = labels.ravel()
        cover = np.stack(
            [np.bincount(flat, weights=c.ravel(), minlength=n) for c in conts], axis=1
        )
        centers = np.array([[b.region.bbox.cx, b.region.bbox.cy] for b in bubbles])
        for lab in range(1, n):
            if cover[lab].max() > 0:
                owner[lab] = int(np.argmax(cover[lab]))
            else:
                owner[lab] = int(np.argmin(np.hypot(*(centers - cents[lab]).T)))
    by_bubble = {
        id(p.bubble): p for p in (plans or []) if getattr(p, "bubble", None) is not None
    }
    owner_map = owner[labels]
    mines = [owner_map == i for i in range(len(bubbles))]
    reads: list[str | None] = [None] * len(bubbles)
    if reader is not None and bubbles:
        canvas = reading_canvas(inpainted, final, ink)
        latin = [not is_japanese(b.target_text) for b in bubbles]  # the reader reads lines
        quads = [ink_lines(m.astype(np.uint8)) if ok else [] for m, ok in zip(mines, latin)]
        texts = reader(canvas, quads)
        reads = [(t if q else "") if ok else None for t, q, ok in zip(texts, quads, latin)]
        # text the renderer turned a quarter (a margin note set down its column) is read
        # turned back upright
        for i, b in enumerate(bubbles):
            angle = getattr(by_bubble.get(id(b)), "angle", 0.0)
            if not latin[i] or abs(abs(angle) - 90) > 1 or not mines[i].any():
                continue
            ys, xs = np.nonzero(mines[i])
            y1, y2 = max(0, ys.min() - 4), ys.max() + 5
            x1, x2 = max(0, xs.min() - 4), xs.max() + 5
            turn = cv2.ROTATE_90_COUNTERCLOCKWISE if angle < 0 else cv2.ROTATE_90_CLOCKWISE
            sub = np.ascontiguousarray(cv2.rotate(canvas[y1:y2, x1:x2], turn))
            m = np.ascontiguousarray(cv2.rotate(mines[i][y1:y2, x1:x2].astype(np.uint8), turn))
            q = ink_lines(m)
            reads[i] = reader(sub, [q])[0] if q else ""
    overlay = final.copy()
    results = []
    for i, (b, cont) in enumerate(zip(bubbles, conts)):
        mine = mines[i]
        ink_px = int(mine.sum())
        outside = mine & ~cont
        ink_bbox, bbox_inside = None, True
        if ink_px:
            ys, xs = np.nonzero(mine)
            ink_bbox = BBox(x1=int(xs.min()), y1=int(ys.min()), x2=int(xs.max()) + 1, y2=int(ys.max()) + 1)
            cy, cx = np.nonzero(cont)
            bbox_inside = bool(
                ink_bbox.x1 >= cx.min() and ink_bbox.y1 >= cy.min()
                and ink_bbox.x2 <= cx.max() + 1 and ink_bbox.y2 <= cy.max() + 1
            )
        plan = by_bubble.get(id(b))
        lay = getattr(plan, "layout", None)
        rendered = count_chars(lay.text) if lay is not None and hasattr(lay, "text") else None
        if lay is not None and getattr(lay, "size", 0):
            font, src = float(lay.size), "layout"
        else:
            font, src = (_estimate_font(mine) if ink_px else None), "estimate"
        results.append(
            RenderBox(
                bubble=i,
                closed=closed[i],
                ink_px=ink_px,
                outside_px=int(outside.sum()),
                ink_bbox=ink_bbox,
                bbox_inside=bbox_inside,
                target_chars=count_chars(b.target_text),
                rendered_chars=rendered,
                font_px=font,
                font_source=src,
                fits=getattr(lay, "fits", None),
                read_text=reads[i],
                read_cer=cer(letters(reads[i]), letters(b.target_text)) if reads[i] is not None else None,
            )
        )
        edge = cont & ~cv2.erode(cont.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        overlay[edge] = (0, 170, 0) if closed[i] else (0, 140, 255)
        overlay[outside] = (0, 0, 255)
        if ink_bbox:
            cv2.rectangle(overlay, (ink_bbox.x1, ink_bbox.y1), (ink_bbox.x2, ink_bbox.y2), (255, 0, 0), 1)
    return results, overlay
