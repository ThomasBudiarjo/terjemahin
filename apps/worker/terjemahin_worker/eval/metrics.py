"""Stage metrics for the benchmark: pure functions over texts, images and masks.

Ground truth has text per bubble but no boxes, so detections are matched to GT bubbles by what
OCR read in them (Hungarian assignment on CER). Inpainting and typesetting are judged from
pixels only, so every registered method is scored the same way whatever it does inside.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

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


@dataclass
class DetectionScore:
    matches: list[Match]
    n_gt: int
    n_det: int
    sfx: list[int] = field(default_factory=list)  # detections that read a listed SFX
    fragments: list[int] = field(default_factory=list)  # pieces of an already matched bubble
    false_positives: list[int] = field(default_factory=list)
    missed: list[int] = field(default_factory=list)  # GT bubbles nothing matched

    @property
    def mean_cer(self) -> float | None:
        return float(np.mean([m.cer for m in self.matches])) if self.matches else None

    @property
    def mean_cer_text(self) -> float | None:
        return float(np.mean([m.cer_text for m in self.matches])) if self.matches else None


def _contained(part: str, whole: str) -> bool:
    """Most of `part` appears in order inside `whole` (a split bubble's other half)."""
    if len(part) < 2:
        return False
    return len(part) - (levenshtein(part, whole) - (len(whole) - len(part))) >= 0.8 * len(part)


def match_detections(
    det_texts: list[str], gt_texts: list[str], sfx: list[str] = (), max_cer: float = 0.5
) -> DetectionScore:
    """Assign detections to GT bubbles by OCR text; a pair counts when its CER <= max_cer.

    Unmatched detections are SFX hits (if they read a listed SFX), fragments (a bubble split in
    two, the other half matched), or false positives.
    """
    hyp = [normalize_ja(t, punct=False) for t in det_texts]
    ref = [normalize_ja(t, punct=False) for t in gt_texts]
    cost = np.array([[min(cer(h, r), 2.0) for r in ref] for h in hyp]).reshape(len(hyp), len(ref))
    matches = [
        Match(d, g, cer(normalize_ja(det_texts[d]), normalize_ja(gt_texts[g])), float(cost[d, g]))
        for d, g in hungarian(cost)
        if cost[d, g] <= max_cer
    ]
    score = DetectionScore(matches, len(gt_texts), len(det_texts))
    matched_d = {m.det for m in matches}
    matched_g = {m.gt for m in matches}
    sfx_n = [normalize_ja(s, punct=False) for s in sfx]
    for d, h in enumerate(hyp):
        if d in matched_d:
            continue
        if any(cer(h, s) <= max_cer for s in sfx_n):
            score.sfx.append(d)
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

    @property
    def inside_frac(self) -> float:
        return 1.0 - self.outside_px / self.ink_px if self.ink_px else 1.0

    @property
    def lost(self) -> bool:
        return self.target_chars > 0 and (
            self.ink_px == 0
            or (self.rendered_chars is not None and self.rendered_chars < self.target_chars)
        )


def count_chars(text: str) -> int:
    """Letters and digits only: renderers legitimately respell punctuation and hyphenate."""
    return sum(c.isalnum() for c in unicodedata.normalize("NFKC", text))


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


def render_check(
    inpainted: np.ndarray,
    final: np.ndarray,
    original: np.ndarray,
    bubbles: list[Bubble],
    plans: list | None = None,
) -> tuple[list[RenderBox], np.ndarray]:
    """Attribute every drawn pixel to a bubble and check it stays in that bubble's container.

    Ink = pixels the renderer changed. Each 8-connected ink component goes to the bubble whose
    container holds most of it (else the nearest box). `plans` (renderers that expose their
    layouts, like the default one) add exact rendered characters and font sizes.
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
    overlay = final.copy()
    results = []
    for i, (b, cont) in enumerate(zip(bubbles, conts)):
        mine = owner_map == i
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
            )
        )
        edge = cont & ~cv2.erode(cont.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        overlay[edge] = (0, 170, 0) if closed[i] else (0, 140, 255)
        overlay[outside] = (0, 0, 255)
        if ink_bbox:
            cv2.rectangle(overlay, (ink_bbox.x1, ink_bbox.y1), (ink_bbox.x2, ink_bbox.y2), (255, 0, 0), 1)
    return results, overlay
