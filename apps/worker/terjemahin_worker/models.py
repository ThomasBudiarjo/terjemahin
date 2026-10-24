from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RGB = tuple[int, int, int]
Point = tuple[int, int]


class BBox(BaseModel):
    """Axis-aligned box in pixel coords, inclusive-exclusive (x2, y2 not included)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def clamp(self, width: int, height: int) -> BBox:
        return BBox(
            x1=max(0, min(self.x1, width - 1)),
            y1=max(0, min(self.y1, height - 1)),
            x2=max(1, min(self.x2, width)),
            y2=max(1, min(self.y2, height)),
        )

    def pad(self, px: int) -> BBox:
        return BBox(x1=self.x1 - px, y1=self.y1 - px, x2=self.x2 + px, y2=self.y2 + px)


class TextRegion(BaseModel):
    """A text block (usually one speech bubble's worth of text).

    `lines` holds the individual text-line quads (4 points each, clockwise from top-left) when
    the detector produces them; renderers and OCR may use them, but `bbox` is always valid.
    """

    id: int
    bbox: BBox
    score: float = 1.0
    lines: list[list[Point]] = Field(default_factory=list)
    direction: Literal["h", "v"] = "v"
    angle: float = 0.0
    font_size: int | None = None  # estimated source glyph size in px
    fg_color: RGB | None = None
    bg_color: RGB | None = None
    kind: Literal["text", "bubble", "sfx"] = "text"


class Bubble(BaseModel):
    """A text region after the LLM has read and translated it."""

    region: TextRegion
    source_text: str = ""
    target_text: str = ""
    speaker: str | None = None
    ocr_confidence: float | None = None
    erase: bool | None = None
    """Whether the original text under the box is erased. None (a bubble added by hand):
    erased when it has source_text. The pipeline sets it on every bubble it produces; it is
    False when a translation came back empty, so the original text stays instead of a blank."""

    def erases(self) -> bool:
        return bool(self.source_text.strip()) if self.erase is None else self.erase


class GlossaryEntry(BaseModel):
    source: str
    target: str
    note: str | None = None


class PageContext(BaseModel):
    """Series-level and neighbouring-page context that makes translations consistent."""

    series_title: str | None = None
    series_summary: str | None = None
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    previous_pages: list[list[str]] = Field(
        default_factory=list,
        description="Indonesian text of previous pages, one list per page, in reading order.",
    )
    style_notes: str | None = None


class PipelineConfig(BaseModel):
    """Which method runs at each stage, plus per-stage options.

    `options` is keyed by stage name, e.g. {"inpainter": {"size": 1024}}. Unset method names
    fall back to Settings defaults (env vars DETECTOR, OCR, TRANSLATOR, ...).
    """

    detector: str | None = None
    ocr: str | None = None
    translator: str | None = None
    mask_refiner: str | None = None
    inpainter: str | None = None
    renderer: str | None = None
    source_lang: str = "ja"
    target_lang: str = "id"
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    min_ocr_confidence: float | None = None
    """Bubbles whose OCR confidence is below this are left untouched (Settings default)."""


class PageResult(BaseModel):
    bubbles: list[Bubble]
    width: int
    height: int
    pipeline: PipelineConfig | None = None
    timings: dict[str, float] = Field(default_factory=dict)


class JobMessage(BaseModel):
    job_id: str
    page_id: str
    input_path: str
    output_path: str
    result_path: str
    context: PageContext = Field(default_factory=PageContext)
    pipeline: PipelineConfig | None = None
    debug_dir: str | None = None
