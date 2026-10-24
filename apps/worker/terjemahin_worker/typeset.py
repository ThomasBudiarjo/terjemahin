from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .models import BBox

MIN_FONT = 9
LINE_SPACING = 1.12
INNER_PAD_RATIO = 0.06

FontT = ImageFont.FreeTypeFont | ImageFont.ImageFont


@dataclass
class Layout:
    lines: list[str]
    font_size: int
    line_height: int


def _words(text: str) -> list[str]:
    """Words, with "\\n" tokens where the text has line breaks (kept, blank lines too)."""
    paras = text.replace("\r\n", "\n").strip().split("\n")
    out: list[str] = []
    for i, para in enumerate(paras):
        out += para.split()
        if i < len(paras) - 1:
            out.append("\n")
    return out


class Typesetter:
    def __init__(self, font_path: Path | None) -> None:
        self.font_path = Path(font_path) if font_path and Path(font_path).exists() else None

    @lru_cache(maxsize=256)
    def _font(self, size: int) -> FontT:
        if self.font_path:
            return ImageFont.truetype(str(self.font_path), size)
        return ImageFont.load_default(size=size)

    def layout(self, draw: ImageDraw.ImageDraw, text: str, box_w: int, box_h: int) -> Layout:
        """Largest font size whose word-wrapped text fits inside (box_w, box_h)."""
        words = _words(text)
        if not words:
            return Layout([], MIN_FONT, MIN_FONT)

        lo, hi = MIN_FONT, max(MIN_FONT, box_h)
        best: Layout | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            lay = self._try(draw, words, mid, box_w, box_h)
            if lay:
                best, lo = lay, mid + 1
            else:
                hi = mid - 1
        return best or self._force(draw, words, MIN_FONT, box_w)

    def _try(
        self, draw: ImageDraw.ImageDraw, words: list[str], size: int, box_w: int, box_h: int
    ) -> Layout | None:
        font = self._font(size)
        line_h = round(size * LINE_SPACING)
        lines: list[str] = []
        cur = ""
        for w in words:
            if w == "\n":  # a line break typed in the text
                lines.append(cur)
                cur = ""
                continue
            cand = f"{cur} {w}".strip()
            if draw.textlength(cand, font=font) <= box_w:
                cur = cand
            else:
                if not cur:  # single word wider than the box at this size
                    return None
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        if any(draw.textlength(ln, font=font) > box_w for ln in lines) or len(lines) * line_h > box_h:
            return None
        return Layout(lines, size, line_h)

    def _force(self, draw: ImageDraw.ImageDraw, words: list[str], size: int, box_w: int) -> Layout:
        """Minimum size still overflows: break long words by character so nothing is lost."""
        font = self._font(size)
        line_h = round(size * LINE_SPACING)
        lines: list[str] = []
        cur = ""
        for w in words:
            if w == "\n":
                lines.append(cur.strip())
                cur = ""
                continue
            for ch in w:
                cand = cur + ch
                if draw.textlength(cand, font=font) <= box_w or not cur:
                    cur = cand
                else:
                    lines.append(cur)
                    cur = ch
            cur += " "
        if cur.strip():
            lines.append(cur.strip())
        return Layout(lines, size, line_h)

    def render(
        self,
        image: Image.Image,
        bbox: BBox,
        text: str,
        bg_rgb: tuple[int, int, int] = (255, 255, 255),
    ) -> Layout:
        """Draw `text` centred inside `bbox`. Text colour is chosen by background luminance."""
        draw = ImageDraw.Draw(image)
        pad = round(min(bbox.w, bbox.h) * INNER_PAD_RATIO)
        box_w, box_h = max(1, bbox.w - 2 * pad), max(1, bbox.h - 2 * pad)
        lay = self.layout(draw, text, box_w, box_h)
        if not lay.lines:
            return lay

        font = self._font(lay.font_size)
        lum = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
        fill = (0, 0, 0) if lum > 128 else (255, 255, 255)

        total_h = len(lay.lines) * lay.line_height
        y = bbox.y1 + pad + (box_h - total_h) / 2
        for line in lay.lines:
            w = draw.textlength(line, font=font)
            x = bbox.x1 + pad + (box_w - w) / 2
            draw.text((x, y), line, font=font, fill=fill)
            y += lay.line_height
        return lay
