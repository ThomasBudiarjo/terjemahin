"""Shared core of the LLM translators (OpenAI-compatible, Anthropic, Gemini).

A provider only turns a `ChatRequest` (system prompt + text/image parts + JSON schema) into its
HTTP body and pulls the answer text out of its response. Everything that decides translation
quality and robustness lives here once:

- text mode (numbered OCR lines + context, optional page image) or vision mode (page + crops,
  the model also reads the text); `mode=auto` picks vision only when OCR gave nothing;
- tolerant parsing: strict JSON, JSON buried in prose/code fences/<think>, any list-shaped key,
  0-based indices, and finally `<|n|>` / `n.` numbered lines;
- repair: items that are missing, empty or still in the source script are re-requested with an
  explicit index list; whatever is still missing goes to an optional fallback translator;
- HTTP retries with exponential backoff (honouring Retry-After) on 429/5xx/timeouts, and
  automatic removal of optional request fields an endpoint rejects with 400;
- the result always has exactly one item per region.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import re
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np

from ..models import PageContext, TextRegion
from . import prompt as P
from .base import TranslationItem, TranslationRequest

log = logging.getLogger(__name__)

ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "source": {"type": "string"},
        "target": {"type": "string"},
        "speaker": {"type": ["string", "null"]},
    },
    "required": ["index", "source", "target", "speaker"],
    "additionalProperties": False,
}
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": ITEM_SCHEMA}},
    "required": ["items"],
    "additionalProperties": False,
}

RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529})


class LLMError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class Image:
    jpeg: bytes
    label: str = ""

    @property
    def b64(self) -> str:
        return base64.b64encode(self.jpeg).decode("ascii")


Part = str | Image


@dataclass
class ChatRequest:
    system: str
    parts: list[Part]
    schema: dict[str, Any] = field(default_factory=lambda: RESPONSE_SCHEMA)
    max_tokens: int = 4096
    temperature: float | None = 0.3

    @property
    def has_images(self) -> bool:
        return any(isinstance(p, Image) for p in self.parts)

    def text(self) -> str:
        return "\n\n".join(p for p in self.parts if isinstance(p, str))


class Backend(Protocol):
    name: str
    vision: bool

    def complete(self, req: ChatRequest) -> str:
        """Raw answer text of the model."""
        ...


# --- HTTP ------------------------------------------------------------------------------------

class Http:
    """One httpx client plus the retry policy every provider shares."""

    def __init__(
        self,
        timeout: float = 120.0,
        max_retries: int = 4,
        backoff: float = 1.5,
        max_wait: float = 60.0,
        transport: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        import httpx

        self.httpx = httpx
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(timeout, 15.0)), transport=transport
        )
        self.max_retries = max_retries
        self.backoff = backoff
        self.max_wait = max_wait
        self.sleep = sleep

    def _wait(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(self.max_wait, max(0.0, float(retry_after)))
            except ValueError:
                pass
        return min(self.max_wait, self.backoff * (2**attempt) * (0.75 + 0.5 * random.random()))

    def post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        last: LLMError | None = None
        for attempt in range(self.max_retries + 1):
            retry_after = None
            try:
                r = self.client.post(url, headers=headers, json=body)
            except self.httpx.TimeoutException as e:
                last = LLMError(f"timeout: {e}", retryable=True)
            except self.httpx.TransportError as e:
                last = LLMError(f"connection error: {e}", retryable=True)
            else:
                if r.status_code < 400:
                    try:
                        return r.json()
                    except ValueError as e:
                        last = LLMError(f"invalid JSON body from {url}: {e}", r.status_code, True)
                else:
                    msg = _error_message(r)
                    last = LLMError(f"HTTP {r.status_code}: {msg}", r.status_code,
                                    r.status_code in RETRY_STATUS)
                    retry_after = r.headers.get("retry-after")
                    if not last.retryable:
                        raise last
            if attempt < self.max_retries:
                wait = self._wait(attempt, retry_after)
                log.warning("LLM request failed (%s); retry %d/%d in %.1fs",
                            last, attempt + 1, self.max_retries, wait)
                self.sleep(wait)
        assert last is not None
        raise last

    def post_adaptive(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        optional: Sequence[str] = (),
    ) -> dict[str, Any]:
        """POST; on a 400 that names an optional field, drop that field and try again.

        OpenAI-compatible servers disagree on `response_format`, `temperature` and
        `max_tokens` (reasoning models want `max_completion_tokens`); this keeps one code path.
        """
        body = dict(body)
        for _ in range(len(optional) + 2):
            try:
                return self.post(url, headers, body)
            except LLMError as e:
                if e.status not in (400, 422):
                    raise
                text = str(e).lower()
                if "max_completion_tokens" in text and "max_tokens" in body:
                    body["max_completion_tokens"] = body.pop("max_tokens")
                    continue
                drop = [k for k in optional if k in body and k.lower() in text]
                if not drop:
                    raise
                log.info("endpoint rejected %s; retrying without it", ", ".join(drop))
                for k in drop:
                    body.pop(k)
        return self.post(url, headers, body)


def _error_message(r: Any) -> str:
    try:
        data = r.json()
    except ValueError:
        return r.text[:500]
    err = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(err, dict):
        return str(err.get("message") or err)[:500]
    return str(err)[:500]


# --- images ----------------------------------------------------------------------------------

def encode_jpeg(image_bgr: np.ndarray, max_side: int, quality: int = 88) -> bytes:
    h, w = image_bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        image_bgr = cv2.resize(image_bgr, (max(1, round(w * s)), max(1, round(h * s))),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("failed to encode image")
    return buf.tobytes()


def crop_region(page_bgr: np.ndarray, region: TextRegion, pad: int = 8, min_side: int = 48) -> np.ndarray:
    H, W = page_bgr.shape[:2]
    b = region.bbox.pad(pad).clamp(W, H)
    crop = page_bgr[b.y1 : b.y2, b.x1 : b.x2]
    h, w = crop.shape[:2]
    if 0 < min(h, w) < min_side:  # tiny crops are unreadable for vision encoders
        s = min_side / min(h, w)
        crop = cv2.resize(crop, (round(w * s), round(h * s)), interpolation=cv2.INTER_CUBIC)
    return crop


# --- parsing ---------------------------------------------------------------------------------

_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_NUMBERED = re.compile(r"^\s*(?:<\|(\d+)\|>|\[(\d+)\]|#(\d+)[:.)]?|(\d+)\s*[.):：、])\s*(.*)$")
_INDEX_KEYS = ("index", "idx", "id", "ID", "i", "no", "number", "n")
_TARGET_KEYS = ("target", "translation", "translated", "tgt", "text", "output")
_SOURCE_KEYS = ("source", "original", "src", "ocr", "japanese")


@dataclass
class Parsed:
    target: str
    source: str = ""
    speaker: str | None = None


def clean_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r", "")
    s = re.sub(r"^\s*<\|\d+\|>\s*", "", s)
    s = " ".join(s.split())
    if len(s) >= 2 and (s[0], s[-1]) in {('"', '"'), ("“", "”"), ("「", "」"), ("'", "'")}:
        s = s[1:-1].strip()
    return s


def _json_candidates(text: str) -> list[str]:
    out = [text]
    out += [m.group(1) for m in _FENCE.finditer(text)]
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    end = max(text.rfind("}"), text.rfind("]"))
    if starts and end > min(starts):
        out.append(text[min(starts) : end + 1])
    out += [_TRAILING_COMMA.sub(r"\1", c) for c in list(out)]
    return out


def _find_list(data: Any) -> list[Any] | None:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "translations", "TextList", "results", "data", "bubbles", "lines"):
            if isinstance(data.get(key), list):
                return data[key]
        for v in data.values():
            if isinstance(v, list) and v:
                return v
        for v in data.values():
            found = _find_list(v) if isinstance(v, dict) else None
            if found is not None:
                return found
        if any(k in data for k in _TARGET_KEYS):
            return [data]
        if data and all(re.fullmatch(r"\d+", str(k)) for k in data):  # {"1": "...", "2": "..."}
            return [{"index": int(k), "target": v} for k, v in data.items()]
    return None


def _first(d: dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _row(row: Any) -> tuple[int | None, Parsed] | None:
    if isinstance(row, dict):
        idx = _first(row, _INDEX_KEYS)
        try:
            idx = int(str(idx).strip("#<|> ")) if idx is not None else None
        except ValueError:
            idx = None
        return idx, Parsed(
            clean_text(_first(row, _TARGET_KEYS)),
            clean_text(_first(row, _SOURCE_KEYS)),
            clean_text(row.get("speaker")) or None,
        )
    if isinstance(row, str | int | float):
        return None, Parsed(clean_text(row))
    return None


def _from_json(text: str, expected: Sequence[int]) -> dict[int, Parsed] | None:
    for cand in _json_candidates(text):
        try:
            data = json.loads(cand)
        except (ValueError, TypeError):
            continue
        rows = _find_list(data)
        if rows is not None:
            return _assign([r for r in map(_row, rows) if r is not None], expected)
    return None


_FLAT_OBJECT = re.compile(r"\{[^{}]*\}")


def _from_objects(text: str, expected: Sequence[int]) -> dict[int, Parsed]:
    """Salvage item objects from a broken document (bad brackets, truncation, stray prose)."""
    rows = []
    for m in _FLAT_OBJECT.finditer(text):
        try:
            obj = json.loads(_TRAILING_COMMA.sub(r"\1", m.group()))
        except ValueError:
            continue
        if isinstance(obj, dict) and any(k in obj for k in _TARGET_KEYS):
            rows.append(_row(obj))
    return _assign([r for r in rows if r is not None], expected)


def _assign(rows: list[tuple[int | None, Parsed]], expected: Sequence[int]) -> dict[int, Parsed]:
    exp = list(expected)
    given = [i for i, _ in rows if i is not None]
    shift = 0
    if given and 0 in given and 0 not in exp and all(i + 1 in exp for i in given):
        shift = 1  # model counted from 0
    out: dict[int, Parsed] = {}
    pending = iter(exp)
    for idx, item in rows:
        if idx is None:
            idx = next((i for i in pending if i not in out), None)
            if idx is None:
                continue
        else:
            idx += shift
        if idx in exp and idx not in out:
            out[idx] = item
    return out


def _from_lines(text: str, expected: Sequence[int]) -> dict[int, Parsed]:
    rows: list[tuple[int | None, Parsed]] = []
    for line in text.splitlines():
        m = _NUMBERED.match(line)
        if m:
            idx = int(next(g for g in m.groups()[:4] if g is not None))
            rows.append((idx, Parsed(clean_text(m.group(5)))))
        elif rows and line.strip():
            prev = rows[-1][1]
            prev.target = clean_text(f"{prev.target} {line}")
    return _assign(rows, expected)


def parse_items(text: str, expected: Sequence[int]) -> dict[int, Parsed]:
    """Best-effort mapping index -> item. Indices outside `expected` are dropped."""
    text = _THINK.sub("", text or "").strip()
    if not text:
        return {}
    return _from_json(text, expected) or _from_objects(text, expected) or _from_lines(text, expected)


# --- the translator --------------------------------------------------------------------------

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯ｦ-ﾟ]")
_LETTER = re.compile(r"[^\W\d_]")
_CJK_TARGETS = ("ja", "zh", "zh-hant", "ko")


def untranslated(target: str, source: str, target_lang: str) -> bool:
    """Target still mostly in the source's CJK script (the model echoed instead of translating)."""
    if target_lang in _CJK_TARGETS or not _CJK.search(source):
        return False
    letters = _LETTER.findall(target)
    return bool(letters) and sum(1 for c in letters if _CJK.match(c)) / len(letters) > 0.3


def _words(s: str) -> frozenset[str]:
    return frozenset(s.split())


_STOPWORDS: dict[str, frozenset[str]] = {
    "id": _words(
        "aku kamu dia itu ini yang dan apa tidak nggak gak sih kan gimana juga saja aja sudah udah ada "
        "dengan untuk ke di dari lagi terus kok dong banget kalau tapi mau bisa sangat selalu saya kita "
        "mereka begitu gitu deh lho cuma sendiri bagaimana kenapa"
    ),
    "en": _words(
        "the you i i'm is are was what that this and to of it he she so just my your do don't not very "
        "about how with yeah wow it's that's can't me his her they we"
    ),
}


def wrong_language(targets: Sequence[str], target_lang: str) -> bool:
    """Page-level check that the answer is in the requested language (small models drift to English)."""
    own_words = _STOPWORDS.get(target_lang)
    if own_words is None:
        return False
    words = re.findall(r"[a-z']+", " ".join(targets).lower())
    if len(words) < 6:
        return False
    own = sum(w in own_words for w in words)
    other = max(sum(w in sw for w in words) for lang, sw in _STOPWORDS.items() if lang != target_lang)
    return other >= 3 and other > 2 * own


@dataclass
class _Slot:
    pos: int  # index into req.regions
    number: int  # 1-based index used in the prompt
    region: TextRegion
    source: str


class LLMTranslator:
    """Provider-agnostic LLM translator; see the module docstring."""

    def __init__(
        self,
        backend: Backend,
        *,
        mode: str = "auto",
        page_image: bool | None = None,
        crops: bool = True,
        positions: bool = True,
        max_items: int = 40,
        repair: int = 1,
        parse_retries: int = 1,
        page_max_side: int = 1536,
        crop_max_side: int = 768,
        crop_pad: int = 8,
        max_tokens: int = 4096,
        temperature: float | None = 0.3,
        prev_pages: int = 2,
        remember_pages: int = 0,
        fallback: Callable[[TranslationRequest], list[TranslationItem]] | None = None,
    ) -> None:
        if mode not in ("auto", "text", "vision"):
            raise ValueError(f"mode must be auto|text|vision, got {mode!r}")
        if mode == "vision" and not backend.vision:
            raise ValueError(f"{backend.name}: model is not vision-capable (set vision=true if it is)")
        self.backend = backend
        self.mode = mode
        self.page_image = backend.vision if page_image is None else bool(page_image and backend.vision)
        self.crops = crops
        self.positions = positions
        self.max_items = max(1, max_items)
        self.repair = repair
        self.parse_retries = parse_retries
        self.page_max_side = page_max_side
        self.crop_max_side = crop_max_side
        self.crop_pad = crop_pad
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prev_pages = prev_pages
        self.memory: deque[list[str]] = deque(maxlen=max(0, remember_pages))
        self.fallback = fallback
        self.calls = 0

    @property
    def vision(self) -> bool:
        return self.backend.vision

    def resolve_mode(self, req: TranslationRequest) -> str:
        if self.mode != "auto":
            return self.mode
        empty = not any(s.strip() for s in req.sources)
        return "vision" if empty and self.backend.vision else "text"

    def translate(self, req: TranslationRequest) -> list[TranslationItem]:
        n = len(req.regions)
        if n == 0:
            return []
        sources = list(req.sources) + [""] * (n - len(req.sources))
        mode = self.resolve_mode(req)
        positions = range(n) if mode == "vision" else [i for i in range(n) if sources[i].strip()]
        if not positions:
            log.warning("%s: no OCR text and no vision; nothing to translate", self.backend.name)
            return [TranslationItem(source=s) for s in sources[:n]]

        context = req.context
        if not context.previous_pages and self.memory:
            context = context.model_copy(update={"previous_pages": list(self.memory)})
        page_jpeg = None
        if mode == "vision" or self.page_image:
            page_jpeg = encode_jpeg(req.page_bgr, self.page_max_side)

        results: dict[int, Parsed] = {}
        pos_list = list(positions)
        for start in range(0, len(pos_list), self.max_items):
            chunk = pos_list[start : start + self.max_items]
            slots = [_Slot(p, k + 1, req.regions[p], sources[p]) for k, p in enumerate(chunk)]
            got = self._run(req, mode, slots, context, page_jpeg)
            results.update({s.pos: got[s.number] for s in slots if s.number in got})

        missing = [p for p in pos_list if p not in results]
        if missing and self.fallback and mode == "text":
            log.warning("%s: %d item(s) still missing, using fallback", self.backend.name, len(missing))
            sub = TranslationRequest(
                page_bgr=req.page_bgr, regions=[req.regions[p] for p in missing],
                sources=[sources[p] for p in missing], context=req.context,
                source_lang=req.source_lang, target_lang=req.target_lang,
            )
            for p, it in zip(missing, self.fallback(sub)):
                if it.target.strip():
                    results[p] = Parsed(it.target, sources[p])
        elif missing:
            log.warning("%s: %d item(s) left untranslated", self.backend.name, len(missing))

        items: list[TranslationItem] = []
        for i in range(n):
            got_item = results.get(i)
            if got_item is None:
                items.append(TranslationItem(source=sources[i]))
                continue
            src = got_item.source if mode == "vision" else sources[i]
            items.append(TranslationItem(source=src, target=got_item.target, speaker=got_item.speaker))
        if self.memory.maxlen:
            self.memory.append([it.target for it in items if it.target])
        return items

    # one chunk: request, parse, repair
    def _run(
        self, req: TranslationRequest, mode: str, slots: list[_Slot], context: PageContext,
        page_jpeg: bytes | None,
    ) -> dict[int, Parsed]:
        expected = [s.number for s in slots]
        system = P.system_prompt(mode, req.source_lang, req.target_lang, page_image=page_jpeg is not None and mode == "text")
        ctx = P.context_block(context, [s.source for s in slots], req.target_lang, self.prev_pages)

        got: dict[int, Parsed] = {}
        for attempt in range(self.parse_retries + 1):
            text = self._complete(system, self._parts(req, mode, slots, ctx, page_jpeg, None))
            got = self._accept(parse_items(text, expected), slots, mode, req.target_lang)
            if got:
                break
            log.warning("%s: unparseable answer (attempt %d): %.200r", self.backend.name, attempt + 1, text)

        if self.repair and wrong_language([g.target for g in got.values()], req.target_lang):
            log.warning("%s: answer is not in '%s', asking again", self.backend.name, req.target_lang)
            note = P.language_note(req.target_lang)
            text = self._complete(system, self._parts(req, mode, slots, ctx, page_jpeg, None, note))
            again = self._accept(parse_items(text, expected), slots, mode, req.target_lang)
            if len(again) >= len(got) and not wrong_language([g.target for g in again.values()], req.target_lang):
                got = again

        for _ in range(self.repair):
            missing = [n for n in expected if n not in got]
            if not missing:
                break
            log.warning("%s: %d/%d items missing (%s), asking again", self.backend.name,
                        len(missing), len(expected), missing)
            text = self._complete(system, self._parts(req, mode, slots, ctx, page_jpeg, missing))
            fixed = self._accept(parse_items(text, missing), slots, mode, req.target_lang)
            got.update({k: v for k, v in fixed.items() if k in missing})
        return got

    def _accept(self, parsed: dict[int, Parsed], slots: list[_Slot], mode: str, target_lang: str) -> dict[int, Parsed]:
        """Drop items that need a retry: empty targets for real text, or echoed source."""
        by_num = {s.number: s for s in slots}
        out: dict[int, Parsed] = {}
        for k, it in parsed.items():
            src = by_num[k].source if mode == "text" else it.source
            if not it.target and (mode == "text" and src.strip()):
                continue
            if it.target and untranslated(it.target, src or it.target, target_lang):
                continue
            out[k] = it
        return out

    def _complete(self, system: str, parts: list[Part]) -> str:
        self.calls += 1
        t = time.perf_counter()
        text = self.backend.complete(ChatRequest(system, parts, RESPONSE_SCHEMA, self.max_tokens, self.temperature))
        log.info("%s: answer in %.1fs (%d chars)", self.backend.name, time.perf_counter() - t, len(text))
        return text

    def _parts(
        self, req: TranslationRequest, mode: str, slots: list[_Slot], ctx: str,
        page_jpeg: bytes | None, missing: list[int] | None, note: str | None = None,
    ) -> list[Part]:
        parts: list[Part] = []
        if mode == "text":
            H, W = req.page_bgr.shape[:2]
            lines = [
                P.Line(
                    s.number, s.source,
                    pos=(round(100 * s.region.bbox.cx / W), round(100 * s.region.bbox.cy / H))
                    if page_jpeg is not None and self.positions else None,
                    sfx=s.region.kind == "sfx",
                )
                for s in slots
            ]
            task = P.text_task(lines, req.target_lang, req.source_lang)
            if missing:
                task += "\n\n" + P.repair_note(missing, req.target_lang)
            if note:
                task += "\n\n" + note
            parts.append(f"{ctx}\n\n{task}" if ctx else task)
            if page_jpeg is not None:
                parts.append(Image(page_jpeg, "page"))
            return parts

        want = [s for s in slots if missing is None or s.number in missing]
        task = P.vision_task([s.number for s in want], req.target_lang, has_page=page_jpeg is not None)
        if missing:
            task += "\n\n" + P.repair_note(missing, req.target_lang)
        if note:
            task += "\n\n" + note
        parts.append(f"{ctx}\n\n{task}" if ctx else task)
        if page_jpeg is not None:
            parts.append(Image(page_jpeg, "page"))
        if self.crops:
            for s in want:
                parts.append(P.crop_label(s.number, req.target_lang))
                crop = crop_region(req.page_bgr, s.region, self.crop_pad)
                parts.append(Image(encode_jpeg(crop, self.crop_max_side, 92), f"crop {s.number}"))
        return parts


# --- factory helpers -------------------------------------------------------------------------

CORE_OPTS = (
    "mode", "page_image", "crops", "positions", "max_items", "repair", "parse_retries",
    "page_max_side", "crop_max_side", "crop_pad", "max_tokens", "temperature", "prev_pages",
    "remember_pages",
)
HTTP_OPTS = ("timeout", "max_retries", "backoff", "max_wait", "transport", "sleep")


def split_opts(opts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(core translator opts, Http opts, provider opts)."""
    core = {k: opts.pop(k) for k in CORE_OPTS if k in opts}
    http = {k: opts.pop(k) for k in HTTP_OPTS if k in opts}
    return core, http, opts


def make_translator(backend: Backend, settings: Any, core: dict[str, Any], fallback: str | None) -> LLMTranslator:
    fb = None
    if fallback:
        from ..registry import TRANSLATORS

        holder: list[Any] = []

        def fb(sub: TranslationRequest) -> list[TranslationItem]:
            if not holder:
                holder.append(TRANSLATORS.build(fallback, settings))
            return holder[0].translate(sub)

    return LLMTranslator(backend, fallback=fb, **core)


def as_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)
