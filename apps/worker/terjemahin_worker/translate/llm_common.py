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
- HTTP retries with exponential backoff (honouring Retry-After) on 429/5xx/connection errors,
  read timeouts retried at most `timeout_retries` times (a slow model would only be asked to
  generate the whole page again), and automatic removal of optional request fields an endpoint
  rejects with 400; an endpoint that rejects images falls back to text-only requests;
- a request that still fails goes to the fallback translator (text mode) instead of failing the
  page, and source-script characters never leak into a non-CJK target;
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

from ..models import GlossaryEntry, PageContext, TextRegion
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


class ImagesRejected(LLMError):
    """The endpoint/model does not accept image input (text-only model behind a vision API)."""


_IMAGE_ERROR = re.compile(
    r"image|vision|multi-?modal|image_url|content.{0,40}(?:array|list|type|string)", re.IGNORECASE
)


def images_rejected(e: LLMError) -> bool:
    return e.status in (400, 404, 415, 422) and bool(_IMAGE_ERROR.search(str(e)))


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

CONNECT_RETRIES = 1  # connection refused / connect timeout: retried once, not max_retries times


class Http:
    """One httpx client plus the retry policy every provider shares."""

    def __init__(
        self,
        timeout: float = 120.0,
        max_retries: int = 4,
        timeout_retries: int = 1,
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
        self.timeout_retries = timeout_retries
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
        read_timeouts = connect_fails = 0
        for attempt in range(self.max_retries + 1):
            retry_after = None
            try:
                r = self.client.post(url, headers=headers, json=body)
            except self.httpx.ReadTimeout as e:
                read_timeouts += 1
                last = LLMError(f"no answer within {self.client.timeout.read}s (timeout option): {e}",
                                retryable=True)
                if read_timeouts > self.timeout_retries:
                    raise last from e
            except (self.httpx.UnsupportedProtocol, self.httpx.InvalidURL) as e:
                raise LLMError(f"bad URL {url!r} (check base_url): {e}") from e
            except (self.httpx.ConnectError, self.httpx.ConnectTimeout) as e:
                # a server that is not running stays down for the next 30 s of backoff too
                connect_fails += 1
                last = LLMError(f"cannot connect to {url}: {e} (is the server running and "
                                f"base_url right?)", retryable=True)
                if connect_fails > CONNECT_RETRIES:
                    raise last from e
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
            idx = int(float(str(idx).strip("#<|> "))) if idx is not None else None
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


def _objects(text: str) -> list[str]:
    """Every balanced {...} span, innermost first; braces inside JSON strings are ignored."""
    out: list[str] = []
    stack: list[int] = []
    in_str = esc = False
    for i, c in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            stack.append(i)
        elif c == "}" and stack:
            out.append(text[stack.pop() : i + 1])
    return out


def _from_objects(text: str, expected: Sequence[int]) -> dict[int, Parsed]:
    """Salvage item objects from a broken document (bad brackets, truncation, stray prose)."""
    rows = []
    for span in _objects(text):
        try:
            obj = json.loads(_TRAILING_COMMA.sub(r"\1", span))
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


_CJK_RUN = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯ｦ-ﾟ々〆ー]+")
_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]|「\s*」|『\s*』")
NO_SOURCE = "[?]"  # vision model translated a balloon without transcribing it


def untranslated(target: str, source: str, target_lang: str) -> bool:
    """Target still contains source-script (CJK) characters although the target language has none.

    Any CJK counts, not only a mostly-CJK echo: a single name left in kanji renders as tofu boxes
    with a Latin font.
    """
    if target_lang in _CJK_TARGETS or not _CJK.search(source):
        return False
    return bool(_CJK.search(target))


def apply_glossary(target: str, glossary: Sequence[GlossaryEntry]) -> str:
    """Replace glossary source terms the model left untranslated (坂本 -> Sakamoto)."""
    if not _CJK.search(target):
        return target
    for g in sorted(glossary, key=lambda g: -len(g.source)):
        if g.source and g.target and g.source in target:
            target = target.replace(g.source, f" {g.target} " if g.target.isascii() else g.target)
    return clean_text(re.sub(r"\s+([?!.,…、。？！])", r"\1", target))


def strip_cjk(target: str) -> str:
    """Last resort for a non-CJK target: drop leftover CJK runs rather than render tofu."""
    out = _EMPTY_BRACKETS.sub("", _CJK_RUN.sub(" ", target))
    return clean_text(re.sub(r"\s+([?!.,…])", r"\1", out))


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
        output: str = "json",
    ) -> None:
        if mode not in ("auto", "text", "vision"):
            raise ValueError(f"mode must be auto|text|vision, got {mode!r}")
        if output not in ("json", "lines"):
            raise ValueError(f"output must be json|lines, got {output!r}")
        self.output = output  # "lines": `<|n|> target` answers in text mode (fewer tokens)
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
        self.images_ok = True  # cleared when the endpoint rejects image input
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
        tainted: dict[int, Parsed] = {}
        pos_list = list(positions)
        can_fall_back = self.fallback is not None and mode == "text"
        for start in range(0, len(pos_list), self.max_items):
            chunk = pos_list[start : start + self.max_items]
            slots = [_Slot(p, k + 1, req.regions[p], sources[p]) for k, p in enumerate(chunk)]
            chunk_tainted: dict[int, Parsed] = {}
            try:
                got = self._run(req, mode, slots, context, page_jpeg, chunk_tainted)
            except LLMError as e:
                if not can_fall_back:
                    raise
                log.warning("%s failed (%s); %d item(s) go to the fallback", self.backend.name, e, len(slots))
                got = {}
            results.update({s.pos: got[s.number] for s in slots if s.number in got})
            tainted.update({s.pos: chunk_tainted[s.number] for s in slots if s.number in chunk_tainted})

        missing = [p for p in pos_list if p not in results]
        if missing and can_fall_back:
            log.warning("%s: %d item(s) still missing, using fallback", self.backend.name, len(missing))
            sub = TranslationRequest(
                page_bgr=req.page_bgr, regions=[req.regions[p] for p in missing],
                sources=[sources[p] for p in missing], context=req.context,
                source_lang=req.source_lang, target_lang=req.target_lang,
            )
            for p, it in zip(missing, self.fallback(sub)):
                if it.target.strip():
                    results[p] = Parsed(it.target, sources[p])
        for p in [p for p in pos_list if p not in results and p in tainted]:
            it = tainted[p]
            if not _LETTER.search(strip_cjk(it.target)):
                continue  # a full echo: nothing usable once the CJK is gone
            log.warning("%s: item %d kept CJK %r; stripping it", self.backend.name, p + 1, it.target)
            results[p] = Parsed(strip_cjk(it.target), it.source, it.speaker)
        left = sum(p not in results for p in pos_list)
        if left:
            log.warning("%s: %d item(s) left untranslated", self.backend.name, left)

        items: list[TranslationItem] = []
        for i in range(n):
            got_item = results.get(i)
            if got_item is None:
                items.append(TranslationItem(source=sources[i]))
                continue
            src = sources[i]
            if mode == "vision":
                # the pipeline drops bubbles without source text; keep translated ones
                src = got_item.source or (NO_SOURCE if got_item.target else "")
            items.append(TranslationItem(source=src, target=got_item.target, speaker=got_item.speaker))
        if self.memory.maxlen:
            self.memory.append([it.target for it in items if it.target])
        return items

    # one chunk: request, parse, repair
    def _run(
        self, req: TranslationRequest, mode: str, slots: list[_Slot], context: PageContext,
        page_jpeg: bytes | None, tainted: dict[int, Parsed],
    ) -> dict[int, Parsed]:
        expected = [s.number for s in slots]
        glossary = context.glossary
        leaks: dict[int, list[str]] = {}  # index -> CJK runs the model left in its target

        def accept(text: str, want: Sequence[int]) -> dict[int, Parsed]:
            got = self._accept(parse_items(text, want), slots, mode, req.target_lang, glossary, tainted)
            for k in want:
                if k in got:
                    leaks.pop(k, None)
                elif k in tainted:
                    leaks[k] = _CJK_RUN.findall(tainted[k].target)
            return got

        system = P.system_prompt(mode, req.source_lang, req.target_lang,
                                 page_image=page_jpeg is not None and mode == "text", output=self.output)
        ctx = P.context_block(context, [s.source for s in slots], req.target_lang, self.prev_pages)

        got: dict[int, Parsed] = {}
        for attempt in range(self.parse_retries + 1):
            text = self._complete(system, self._parts(req, mode, slots, ctx, page_jpeg, None), mode)
            got = accept(text, expected)
            if got:
                break
            log.warning("%s: unparseable answer (attempt %d): %.200r", self.backend.name, attempt + 1, text)

        try:
            if self.repair and wrong_language([g.target for g in got.values()], req.target_lang):
                log.warning("%s: answer is not in '%s', asking again", self.backend.name, req.target_lang)
                note = P.language_note(req.target_lang)
                text = self._complete(system, self._parts(req, mode, slots, ctx, page_jpeg, None, note), mode)
                again = accept(text, expected)
                if len(again) >= len(got) and not wrong_language([g.target for g in again.values()], req.target_lang):
                    got = again

            for _ in range(self.repair):
                missing = [n for n in expected if n not in got]
                if not missing:
                    break
                log.warning("%s: %d/%d items missing (%s), asking again", self.backend.name,
                            len(missing), len(expected), missing)
                parts = self._parts(req, mode, slots, ctx, page_jpeg, missing, leaks=leaks)
                text = self._complete(system, parts, mode)
                fixed = accept(text, missing)
                got.update({k: v for k, v in fixed.items() if k in missing})
        except LLMError as e:  # a failed follow-up call must not throw away what we already have
            if not got:
                raise
            log.warning("%s: follow-up request failed (%s); keeping %d item(s)", self.backend.name, e, len(got))
        return got

    def _accept(
        self, parsed: dict[int, Parsed], slots: list[_Slot], mode: str, target_lang: str,
        glossary: Sequence[GlossaryEntry], tainted: dict[int, Parsed],
    ) -> dict[int, Parsed]:
        """Drop items that need a retry: empty targets for real text, or source script left in.

        A mostly-translated item that only kept some CJK is remembered in `tainted` so it can
        still be used (CJK stripped) if repair and fallback do not produce anything better.
        """
        by_num = {s.number: s for s in slots}
        out: dict[int, Parsed] = {}
        for k, it in parsed.items():
            src = by_num[k].source if mode == "text" else it.source
            if not it.target and (mode == "text" and src.strip()):
                continue
            if it.target and untranslated(it.target, src or it.target, target_lang):
                it.target = apply_glossary(it.target, glossary)
                if untranslated(it.target, src or it.target, target_lang):
                    if k not in tainted or _LETTER.search(strip_cjk(it.target)):
                        tainted[k] = it
                    continue
            tainted.pop(k, None)
            out[k] = it
        return out

    def _complete(self, system: str, parts: list[Part], mode: str) -> str:
        self.calls += 1
        t = time.perf_counter()
        if mode == "text" and not self.images_ok:
            parts = [p for p in parts if isinstance(p, str)]
        req = ChatRequest(system, parts, RESPONSE_SCHEMA, self.max_tokens, self.temperature)
        try:
            text = self.backend.complete(req)
        except ImagesRejected as e:
            if mode != "text" or not req.has_images:
                raise  # vision mode cannot work without images
            log.warning("%s rejects images (%s); continuing text-only", self.backend.name, e)
            self.images_ok = False
            req.parts = [p for p in parts if isinstance(p, str)]
            text = self.backend.complete(req)
        log.info("%s: answer in %.1fs (%d chars)", self.backend.name, time.perf_counter() - t, len(text))
        return text

    def _parts(
        self, req: TranslationRequest, mode: str, slots: list[_Slot], ctx: str,
        page_jpeg: bytes | None, missing: list[int] | None, note: str | None = None,
        leaks: dict[int, list[str]] | None = None,
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
            task = P.text_task(lines, req.target_lang, req.source_lang, self.output)
            if missing:
                task += "\n\n" + P.repair_note(missing, req.target_lang, leaks)
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

def as_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def as_int(v: Any) -> int:
    return int(float(v)) if isinstance(v, str) else int(v)


def _optional(conv: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def f(v: Any) -> Any:
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "none", "null", "auto")):
            return None
        return conv(v)

    return f


# Options may arrive as strings (web/API job payloads), so every known one is coerced.
CORE_OPTS: dict[str, Callable[[Any], Any]] = {
    "mode": lambda v: str(v).strip().lower(), "page_image": _optional(as_bool), "crops": as_bool,
    "positions": as_bool, "max_items": as_int, "repair": as_int, "parse_retries": as_int,
    "page_max_side": as_int, "crop_max_side": as_int, "crop_pad": as_int, "max_tokens": as_int,
    "temperature": _optional(float), "prev_pages": as_int, "remember_pages": as_int,
    "output": lambda v: str(v).strip().lower(),
}
HTTP_OPTS: dict[str, Callable[[Any], Any]] = {
    "timeout": float, "max_retries": as_int, "timeout_retries": as_int, "backoff": float,
    "max_wait": float, "transport": lambda v: v, "sleep": lambda v: v,
}


def split_opts(opts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(core translator opts, Http opts, provider opts), known ones coerced to their types."""
    core = {k: conv(opts.pop(k)) for k, conv in CORE_OPTS.items() if k in opts}
    http = {k: conv(opts.pop(k)) for k, conv in HTTP_OPTS.items() if k in opts}
    return core, http, opts


def make_translator(backend: Backend, settings: Any, core: dict[str, Any], fallback: str | None) -> LLMTranslator:
    fb = None
    if fallback:
        from ..registry import TRANSLATORS

        TRANSLATORS.get(fallback)  # a typo fails at build, not after the primary has failed
        holder: list[Any] = []

        def fb(sub: TranslationRequest) -> list[TranslationItem]:
            if not holder:
                holder.append(TRANSLATORS.build(fallback, settings))
            return holder[0].translate(sub)

    return LLMTranslator(backend, fallback=fb, **core)

