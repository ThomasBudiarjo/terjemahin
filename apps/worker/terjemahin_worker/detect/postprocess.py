"""Detector-independent clean-up of text regions, from the page pixels.

Two failure modes seen on real B/W scans (Manga109 ARMS_015, Arisa_010) that every detector
shares, because they come from boxes predicted at reduced resolution:

- **one box over two balloons.** Two short columns in two touching balloons ("ドクン" above,
  "ドクン" below and to the left) come back as one block, read "ドクンドクシ" and get one
  translation drawn across both. Columns of one Japanese text block run side by side and share
  most of their height (rows of a horizontal block share their width); lines that do not are
  separate blocks, so `split_unaligned` cuts the region there.
- **a box that stops short of its text.** "ドクン" boxed without its ド, "なに…？" without the
  ？, "はーい" without the い, "じゃあいつは……？" without the じゃ and ……？ columns: OCR reads
  less than is there and the glyphs outside the box stay on the final page under the
  translation. `complete_region` grows the box over the ink islands next to it in the same
  balloon: dark connected components, wholly surrounded by the paper the box's text sits on,
  that continue a line of it (along the reading axis, any size: dots, ？, small kana) or form
  a further full-size column (row) beside it. Smaller glyphs beside a column are furigana:
  they are left out of the box (they only confuse OCR) and erased by the mask refiner.

Only light paper (balloons, white margins) is handled; text over art or screentone is left as
the detector boxed it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import cv2
import numpy as np

from ..models import BBox, TextRegion

log = logging.getLogger(__name__)

ALONG_GAP = 0.8  # x font: largest gap to an island that continues a line
BESIDE_GAP = 0.6  # x font: largest gap to a further column / row
FULL_SIZE = 0.65  # x font: smaller islands beside the text are furigana
MAX_GLYPH = 2.2  # x font: larger islands are art, outlines or another block
MAX_DASH = 6.0  # x font: a long-vowel dash (は―――い) may be this long when thin
MAX_ROUNDS = 8
FLAT = 0.75  # share of the paper pixels within 25 levels of the paper's own level
MIN_ALIGN = 0.3  # share of the shorter of two lines they must have side by side


def _axis_aligned(q: Sequence[Sequence[int]]) -> bool:
    p = np.asarray(q, np.float32)
    if len(p) != 4:
        return False
    xs, ys = np.sort(p[:, 0]), np.sort(p[:, 1])
    return bool(xs[1] - xs[0] <= 2 and xs[3] - xs[2] <= 2 and ys[1] - ys[0] <= 2 and ys[3] - ys[2] <= 2)


def _qbox(q: Sequence[Sequence[int]]) -> tuple[int, int, int, int]:
    p = np.asarray(q, np.int64)
    return int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())


def _quad(x1: int, y1: int, x2: int, y2: int) -> list[tuple[int, int]]:
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _vertical(r: TextRegion) -> bool:
    if r.lines:
        return r.direction == "v"
    return r.bbox.h >= 0.8 * r.bbox.w  # box-only detectors leave direction at its default


def _font(r: TextRegion) -> float:
    if r.font_size:
        return float(r.font_size)
    if r.lines:
        sides = []
        for q in r.lines:
            x1, y1, x2, y2 = _qbox(q)
            sides.append(min(x2 - x1, y2 - y1))
        if sides:
            return float(max(4, np.median(sides)))
    return float(max(4, min(r.bbox.w, r.bbox.h)))


def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


# -- splitting ------------------------------------------------------------------------------


def split_unaligned(r: TextRegion, min_align: float = MIN_ALIGN) -> list[TextRegion]:
    """The region cut into groups of lines that run side by side (see module doc). Needs the
    detector's line quads (ctd): glyph islands are too unreliable a stand-in for lines."""
    if len(r.lines) < 2 or abs(r.angle) >= 3:
        return [r]
    v = r.direction == "v"
    boxes = [_qbox(q) for q in r.lines]
    font = _font(r)
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            # along = the reading axis of the lines, across = the axis they stack on
            al = (a[1], a[3], b[1], b[3]) if v else (a[0], a[2], b[0], b[2])
            ac = (a[0], a[2], b[0], b[2]) if v else (a[1], a[3], b[1], b[3])
            shorter = max(1.0, min(al[1] - al[0], al[3] - al[2]))
            side_by_side = _overlap(*al) >= min_align * shorter
            gap_across = max(ac[2] - ac[1], ac[0] - ac[3], 0)
            # one line broken in two by the detector: same column, small gap along it
            thinner = max(1.0, min(ac[1] - ac[0], ac[3] - ac[2]))
            same_line = _overlap(*ac) >= 0.5 * thinner and max(al[2] - al[1], al[0] - al[3]) <= font
            if (side_by_side and gap_across <= 1.5 * font) or same_line:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    if len(groups) < 2:
        return [r]
    out = []
    for idx in groups.values():
        xs1, ys1, xs2, ys2 = zip(*(boxes[i] for i in idx))
        bb = BBox(x1=min(xs1), y1=min(ys1), x2=max(xs2) + 1, y2=max(ys2) + 1)
        out.append(r.model_copy(update={"bbox": bb, "lines": [r.lines[i] for i in idx]}))
    log.info("region at %s: lines in %d unaligned groups, split", r.bbox, len(out))
    return out


def split_by_paper(r: TextRegion, gray: np.ndarray) -> list[TextRegion]:
    """The region cut along balloon boundaries: its glyphs grouped by the paper (balloon
    interior) each sits on; one group per balloon. Works without line quads, so box-only
    detectors (ctd_blk, yolo) get the same split as `split_unaligned`."""
    if abs(r.angle) >= 3:
        return [r]
    H, W = gray.shape[:2]
    font = _font(r)
    pad = round(font) + 4
    b = r.bbox
    x0, y0, x3, y3 = max(0, b.x1 - pad), max(0, b.y1 - pad), min(W, b.x2 + pad), min(H, b.y2 + pad)
    win = gray[y0:y3, x0:x3]
    box = (b.x1 - x0, b.y1 - y0, b.x2 - x0, b.y2 - y0)
    found = _paper(win, box)
    if found is None:
        return [r]
    paper, _ = found
    _, plab, pst, _ = cv2.connectedComponentsWithStats(paper.astype(np.uint8), connectivity=4)
    n, lab, st, _ = cv2.connectedComponentsWithStats((~paper).astype(np.uint8), connectivity=8)
    bx1, by1, bx2, by2 = box
    k = np.ones((5, 5), np.uint8)
    groups: dict[int, list[tuple[int, int, int, int]]] = {}
    for i in range(1, n):
        x, y, w, h, area = (int(t) for t in st[i])
        inside = _overlap(x, x + w, bx1, bx2) * _overlap(y, y + h, by1, by2)
        if inside < 0.5 * w * h or area < 3 or max(w, h) > MAX_GLYPH * font:
            continue
        sy, sx = slice(max(0, y - 3), y + h + 3), slice(max(0, x - 3), x + w + 3)
        comp = (lab[sy, sx] == i).astype(np.uint8)
        filled = comp.copy()
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled, cnts, -1, 1, -1)
        ring = (cv2.dilate(filled, k) > 0) & (filled == 0)
        labs = plab[sy, sx][ring]
        labs = labs[labs > 0]
        if labs.size < 4:
            continue
        home = int(np.bincount(labs).argmax())
        if (labs == home).mean() < 0.7 or pst[home][4] < 4 * font * font:
            continue
        groups.setdefault(home, []).append((x, y, x + w, y + h))
    big = {
        g: v for g, v in groups.items()
        if sum((c - a) * (d - bb) for a, bb, c, d in v) >= 0.25 * font * font
    }
    if len(big) < 2:
        return [r]
    out = []
    for pieces in big.values():
        nx1 = min(p[0] for p in pieces) + x0
        ny1 = min(p[1] for p in pieces) + y0
        nx2 = max(p[2] for p in pieces) + x0
        ny2 = max(p[3] for p in pieces) + y0
        nb = BBox(x1=max(b.x1, nx1), y1=max(b.y1, ny1), x2=min(b.x2, nx2), y2=min(b.y2, ny2))
        if nb.w < 2 or nb.h < 2:
            continue
        lines = []
        for q in r.lines:
            qx1, qy1, qx2, qy2 = _qbox(q)
            if _overlap(qx1, qx2, nb.x1, nb.x2) * _overlap(qy1, qy2, nb.y1, nb.y2) > 0.5 * max(1, (qx2 - qx1) * (qy2 - qy1)):
                lines.append(q)
        out.append(r.model_copy(update={"bbox": nb, "lines": lines or [_quad(nb.x1, nb.y1, nb.x2, nb.y2)] if r.lines else []}))
    if len(out) < 2:
        return [r]
    log.info("region at %s: text on %d separate balloons, split", r.bbox, len(out))
    return out


# -- completion -----------------------------------------------------------------------------


def _paper(gray: np.ndarray, b: tuple[int, int, int, int]) -> tuple[np.ndarray, float] | None:
    """Paper mask of the window and the ink threshold, or None when the text is not dark on
    light paper. `b` is the box in window coordinates."""
    x1, y1, x2, y2 = b
    inside = gray[y1:y2, x1:x2]
    if inside.size < 16:
        return None
    paper_level = float(np.percentile(inside, 90))
    ink_level = float(np.percentile(inside, 3))
    if paper_level < 170 or paper_level - ink_level < 80:
        return None
    t = (paper_level + ink_level) / 2
    paper = gray > t
    # balloon paper is flat; light, textured art (a washed-out scan's furniture, sky
    # gradients) is not, and its line fragments are no glyphs
    if (gray[paper] >= paper_level - 25).mean() < FLAT:
        return None
    return paper, t


def complete_region(
    r: TextRegion, gray: np.ndarray, others: Sequence[BBox] = ()
) -> TextRegion:
    """The region with its box (and line quads) grown over the ink islands that belong to its
    text (see module doc). Unchanged when nothing qualifies."""
    if abs(r.angle) >= 3:
        return r
    H, W = gray.shape[:2]
    font = _font(r)
    v = _vertical(r)
    pad = round(4 * font) + 4
    bx = [r.bbox.x1, r.bbox.y1, r.bbox.x2, r.bbox.y2]
    x0, y0 = max(0, bx[0] - pad), max(0, bx[1] - pad)
    x3, y3 = min(W, bx[2] + pad), min(H, bx[3] + pad)
    win = gray[y0:y3, x0:x3]
    box = (bx[0] - x0, bx[1] - y0, bx[2] - x0, bx[3] - y0)
    found = _paper(win, box)
    if found is None:
        return r
    paper, _ = found
    _, plab = cv2.connectedComponents(paper.astype(np.uint8), connectivity=4)
    # the paper the text sits on: the label most common just inside / around the box
    band = np.zeros(paper.shape, bool)
    bx1, by1, bx2, by2 = box
    band[max(0, by1 - 2) : by2 + 2, max(0, bx1 - 2) : bx2 + 2] = True
    lab_in = plab[band & paper]
    if lab_in.size == 0:
        return r
    home = int(np.bincount(lab_in).argmax())
    if home == 0:
        return r
    ink = ~paper
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    hh, ww = ink.shape
    blocked = [
        (o.x1 - x0, o.y1 - y0, o.x2 - x0, o.y2 - y0)
        for o in others
        if _overlap(o.x1, o.x2, x0, x3) > 0 and _overlap(o.y1, o.y2, y0, y3) > 0
    ]
    cand: list[int] = []
    for i in range(1, n):
        x, y, w, h, area = (int(t) for t in st[i])
        if x == 0 or y == 0 or x + w >= ww or y + h >= hh:
            continue  # runs out of the window: art or an outline
        long_, short_ = max(w, h), min(w, h)
        dash = short_ <= 0.45 * font and long_ <= MAX_DASH * font and (h > w) == v
        if long_ > MAX_GLYPH * font and not dash:
            continue
        if long_ < max(2, 0.12 * font) or area < 3:
            continue  # a speck (scan noise, screentone dot)
        if any(_overlap(x, x + w, o[0], o[2]) * _overlap(y, y + h, o[1], o[3]) > 0.3 * w * h for o in blocked):
            continue
        cand.append(i)
    if not cand:
        return r
    # each island must sit on the text's paper all around
    on_home = []
    k = np.ones((5, 5), np.uint8)
    for i in cand:
        x, y, w, h = (int(t) for t in st[i][:4])
        sy, sx = slice(max(0, y - 3), y + h + 3), slice(max(0, x - 3), x + w + 3)
        comp = (lab[sy, sx] == i).astype(np.uint8)
        holes_filled = comp.copy()
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(holes_filled, cnts, -1, 1, -1)
        ring = (cv2.dilate(holes_filled, k) > 0) & (holes_filled == 0)
        if ring.sum() < 4:
            continue
        if (plab[sy, sx][ring] == home).mean() >= 0.7:
            on_home.append(i)
    if not on_home:
        return r
    islands = _chain_dots([tuple(int(t) for t in st[i]) for i in on_home], font, v)
    cur = list(box)
    added: list[tuple[int, int, int, int]] = []
    left = set(range(len(islands)))
    for _ in range(MAX_ROUNDS):
        grew = False
        for i in sorted(left):
            x, y, w, h = islands[i]
            ix1, iy1, ix2, iy2 = x, y, x + w, y + h
            if ix1 >= cur[0] and iy1 >= cur[1] and ix2 <= cur[2] and iy2 <= cur[3]:
                left.discard(i)  # already inside
                continue
            gx = max(cur[0] - ix2, ix1 - cur[2], 0)
            gy = max(cur[1] - iy2, iy1 - cur[3], 0)
            if v:
                along = _overlap(ix1, ix2, cur[0], cur[2]) >= 0.5 * w and gy <= ALONG_GAP * font and gx == 0
                beside = _overlap(iy1, iy2, cur[1], cur[3]) >= 0.5 * h and gx <= BESIDE_GAP * font and gy == 0
                size = max(w, h)
            else:
                along = _overlap(iy1, iy2, cur[1], cur[3]) >= 0.5 * h and gx <= ALONG_GAP * font and gy == 0
                beside = _overlap(ix1, ix2, cur[0], cur[2]) >= 0.5 * w and gy <= BESIDE_GAP * font and gx == 0
                size = max(w, h)
            if along or (beside and size >= FULL_SIZE * font):
                cur = [min(cur[0], ix1), min(cur[1], iy1), max(cur[2], ix2), max(cur[3], iy2)]
                added.append((ix1, iy1, ix2, iy2))
                left.discard(i)
                grew = True
        if not grew:
            break
    if not added:
        return r
    nb = BBox(x1=cur[0] + x0, y1=cur[1] + y0, x2=cur[2] + x0, y2=cur[3] + y0).clamp(W, H)
    if (nb.x1, nb.y1, nb.x2, nb.y2) == (r.bbox.x1, r.bbox.y1, r.bbox.x2, r.bbox.y2):
        return r
    lines = _grow_lines(r, [(a + x0, b + y0, c + x0, d + y0) for a, b, c, d in added], v)
    log.info("region at %s: box grown to %s over %d glyph piece(s) outside it", r.bbox, nb, len(added))
    return r.model_copy(update={"bbox": nb, "lines": lines})


def _chain_dots(
    stats: list[tuple[int, int, int, int, int]], font: float, v: bool
) -> list[tuple[int, int, int, int]]:
    """Islands (x, y, w, h) from (x, y, w, h, area), with runs of dots in reading direction
    ("…" set as a column of its own beside vertical text) joined into one: a lone dot beside
    a column may be part of a furigana glyph, a run of them is punctuation. A dot is small
    and solid; furigana strokes are as small but thin."""
    boxes = [(x, y, w, h) for x, y, w, h, _ in stats]
    dot = [max(w, h) <= 0.3 * font and a >= 0.5 * w * h for _, _, w, h, a in stats]
    idx = [i for i, d in enumerate(dot) if d]
    parent = {i: i for i in idx}

    def find(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for a in idx:
        for b in idx:
            if a >= b:
                continue
            xa, ya, wa, ha = boxes[a]
            xb, yb, wb, hb = boxes[b]
            if v:
                ok = _overlap(xa, xa + wa, xb, xb + wb) > 0 and max(yb - ya - ha, ya - yb - hb) <= font
            else:
                ok = _overlap(ya, ya + ha, yb, yb + hb) > 0 and max(xb - xa - wa, xa - xb - wb) <= font
            if ok:
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    out = [b for b, d in zip(boxes, dot) if not d]
    for g in groups.values():
        if len(g) == 1:
            out.append(boxes[g[0]])
            continue
        x1 = min(boxes[i][0] for i in g)
        y1 = min(boxes[i][1] for i in g)
        x2 = max(boxes[i][0] + boxes[i][2] for i in g)
        y2 = max(boxes[i][1] + boxes[i][3] for i in g)
        out.append((x1, y1, x2 - x1, y2 - y1))
    return out


def _home_paper(gray: np.ndarray, boxes: Sequence[BBox], font: float) -> list[int] | None:
    """Label of the paper each box's text sits on, from one shared window (0 = none)."""
    H, W = gray.shape[:2]
    pad = round(2 * font) + 4
    x0 = max(0, min(b.x1 for b in boxes) - pad)
    y0 = max(0, min(b.y1 for b in boxes) - pad)
    x3 = min(W, max(b.x2 for b in boxes) + pad)
    y3 = min(H, max(b.y2 for b in boxes) + pad)
    win = gray[y0:y3, x0:x3]
    b0 = boxes[0]
    found = _paper(win, (b0.x1 - x0, b0.y1 - y0, b0.x2 - x0, b0.y2 - y0))
    if found is None:
        return None
    paper, _ = found
    _, plab = cv2.connectedComponents(paper.astype(np.uint8), connectivity=4)
    out = []
    for b in boxes:
        band = np.zeros(paper.shape, bool)
        band[max(0, b.y1 - y0 - 2) : b.y2 - y0 + 2, max(0, b.x1 - x0 - 2) : b.x2 - x0 + 2] = True
        lab = plab[band & paper]
        out.append(int(np.bincount(lab).argmax()) if lab.size else 0)
    return out


def merge_adjacent(regions: list[TextRegion], gray: np.ndarray) -> list[TextRegion]:
    """Join two regions that are one block cut in two: same direction and font size, side by
    side with (almost) no gap, on the same paper (one balloon)."""
    regions = list(regions)
    merged = True
    while merged:
        merged = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                a, b = regions[i], regions[j]
                if abs(a.angle) >= 3 or abs(b.angle) >= 3:
                    continue
                v = _vertical(a)
                if v != _vertical(b):
                    continue
                fa, fb = _font(a), _font(b)
                font = max(fa, fb)
                A, B = a.bbox, b.bbox
                small = A if fa < fb else B
                # a sliver beside a block (its "……" column found on its own) has no font
                # size worth comparing
                sliver = (small.w if v else small.h) <= 0.6 * font
                if max(fa, fb) > 1.6 * min(fa, fb) and not sliver:
                    continue
                if v:
                    along = _overlap(A.y1, A.y2, B.y1, B.y2) / max(1, min(A.h, B.h))
                    gap = max(B.x1 - A.x2, A.x1 - B.x2)
                else:
                    along = _overlap(A.x1, A.x2, B.x1, B.x2) / max(1, min(A.w, B.w))
                    gap = max(B.y1 - A.y2, A.y1 - B.y2)
                if along < 0.5 or gap > BESIDE_GAP * font:
                    continue
                homes = _home_paper(gray, [A, B], font)
                if homes is None or homes[0] == 0 or homes[0] != homes[1]:
                    continue
                bb = BBox(x1=min(A.x1, B.x1), y1=min(A.y1, B.y1), x2=max(A.x2, B.x2), y2=max(A.y2, B.y2))
                lines = [*a.lines, *b.lines] if a.lines and b.lines else []
                key = (lambda q: -_qbox(q)[0]) if v else (lambda q: _qbox(q)[1])
                regions[i] = a.model_copy(update={
                    "bbox": bb,
                    "lines": sorted(lines, key=key),
                    "score": max(a.score, b.score),
                    "font_size": min(x for x in (a.font_size, b.font_size) if x) if (a.font_size or b.font_size) else None,
                })
                del regions[j]
                log.info("regions at %s and %s: one block on one paper, merged", A, B)
                merged = True
                break
            if merged:
                break
    return regions


def _grow_lines(
    r: TextRegion, pieces: list[tuple[int, int, int, int]], v: bool
) -> list[list[tuple[int, int]]]:
    """Line quads extended over the added pieces: a piece in line with a quad (same column /
    row) stretches it, any other piece becomes a quad of its own, merged with the pieces in
    line with it. Tilted quads are kept as they are."""
    if not r.lines:
        return []
    quads = [list(map(tuple, q)) for q in r.lines]
    boxes: list[list[int] | None] = [list(_qbox(q)) if _axis_aligned(q) else None for q in quads]
    for p in sorted(pieces, key=lambda p: (p[0], p[1])):
        best, best_ov = None, 0.0
        for j, b in enumerate(boxes):
            if b is None:
                continue
            ov = _overlap(p[0], p[2], b[0], b[2]) / max(1, p[2] - p[0]) if v else (
                _overlap(p[1], p[3], b[1], b[3]) / max(1, p[3] - p[1])
            )
            if ov > best_ov:
                best, best_ov = j, ov
        if best is not None and best_ov >= 0.5:
            b = boxes[best]
            boxes[best] = [min(b[0], p[0]), min(b[1], p[1]), max(b[2], p[2]), max(b[3], p[3])]
        else:
            boxes.append(list(p))
            quads.append(_quad(*p))
    out = []
    for q, b in zip(quads, boxes):
        out.append(q if b is None else _quad(*b))
    return [[(int(x), int(y)) for x, y in q] for q in out]


def refine_regions(regions: Sequence[TextRegion], image_bgr: np.ndarray) -> list[TextRegion]:
    """split_unaligned, then complete_region against the other boxes. Ids are kept in order
    (the caller sorts and renumbers)."""
    if not regions:
        return []
    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr[..., :3], cv2.COLOR_BGR2GRAY)
    split = [p for r in regions for s in split_unaligned(r) for p in split_by_paper(s, gray)]
    split = merge_adjacent(split, gray)
    out: list[TextRegion] = []
    for i, r in enumerate(split):
        others = [o.bbox for j, o in enumerate(split) if j != i]
        out.append(complete_region(r, gray, others))
    return [r.model_copy(update={"id": i}) for i, r in enumerate(out)]
