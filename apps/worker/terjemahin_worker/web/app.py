"""Local web UI + JSON API for hands-on use of the pipeline.

One `Pipeline` instance serves every request (it caches loaded models) and runs on a single
worker thread: models are neither thread-safe nor small, and a second concurrent page would
only halve the speed of both. The event loop never blocks, so the UI keeps polling progress.

Per job everything lives under `<storage_dir>/web/<job_id>/`: the input page, the detector's
raw text mask, stage images (detect, mask, inpainted, final), result.json and job.json.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
import shutil
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, ValidationError

from ..config import Settings, get_settings
from ..imageio import ImageError, decode_image
from ..models import Bubble, PageContext, PageResult, PipelineConfig
from ..pipeline import Pipeline, lang_code
from ..registry import REGISTRIES, STAGES, load_all

log = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 25 << 20
STAGE_IMAGES = ("detect", "mask", "inpainted", "final")
COMPOSE_STAGES = ("mask_refiner", "inpainter", "renderer")
JOB_ID = re.compile(r"^[0-9a-f]{16}$")
FILE_NAME = re.compile(r"^[a-z_]+\.png$")

_progress = threading.local()


class Job(BaseModel):
    id: str
    status: str = "queued"  # queued | running | done | failed
    stage: str | None = None  # stage currently running
    error: str | None = None
    created_at: float = Field(default_factory=time.time)
    elapsed: float | None = None
    version: int = 0  # bumps on every (re)compose so image URLs change
    pipeline: PipelineConfig | None = None
    context: PageContext = Field(default_factory=PageContext)
    result: PageResult | None = None
    dropped: list[Bubble] = Field(default_factory=list)
    """Bubbles the user excluded from typesetting (left as in the original), kept for the UI."""
    erase_key: str | None = None  # what the stored inpainted.png was made from

    def public(self) -> dict[str, Any]:
        out = self.model_dump(exclude={"erase_key"})
        base = f"/api/jobs/{self.id}/files"
        out["images"] = {"original": f"{base}/input.png"}
        if self.status == "done":
            out["images"].update({k: f"{base}/{k}.png?v={self.version}" for k in STAGE_IMAGES})
            out["download"] = f"/api/jobs/{self.id}/download?v={self.version}"
        return out


class ComposeRequest(BaseModel):
    job_id: str
    bubbles: list[Bubble]
    dropped: list[Bubble] = Field(default_factory=list)
    """Bubbles left untouched; only stored so the UI can offer them again."""
    pipeline: PipelineConfig | None = None
    """Only mask_refiner / inpainter / renderer and their options are taken from it."""


class WebPipeline(Pipeline):
    """Pipeline that reports the running stage and keeps the detector's raw mask.

    Both go to a thread-local sink so the shared instance needs no per-job state.
    """

    def stage(self, stage: str, cfg: PipelineConfig) -> Any:
        sink = getattr(_progress, "job", None)
        if sink is not None:
            sink(stage)
        inst = super().stage(stage, cfg)
        return _MaskTap(inst) if stage == "detector" else inst

    def check(self, cfg: PipelineConfig, stages: tuple[str, ...] = STAGES) -> None:
        sink = getattr(_progress, "job", None)
        _progress.job = None  # building the stages up front is not running them
        try:
            super().check(cfg, stages)
        finally:
            _progress.job = sink


class _MaskTap:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def detect(self, image: np.ndarray) -> Any:
        det = self._inner.detect(image)
        _progress.raw_mask = det.mask
        return det

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def erase_key(bubbles: list[Bubble], cfg: PipelineConfig) -> str:
    """Identity of a mask+inpaint result: if unchanged, re-typesetting only re-renders."""
    erase = [b.region.model_dump(include={"bbox", "lines"}) for b in bubbles if b.erases()]
    parts = [erase] + [[getattr(cfg, s), cfg.options.get(s, {})] for s in COMPOSE_STAGES[:2]]
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def list_methods(settings: Settings) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage, reg in load_all().items():
        rows = []
        for name in reg.names():
            info = reg.get(name)
            ok, reason = info.status(settings)
            rows.append(
                {
                    "name": name,
                    "available": ok,
                    "reason": reason,
                    "offline": info.offline,
                    "description": info.description,
                }
            )
        stages[stage] = {"default": getattr(settings, stage), "methods": rows}
    return {
        "stages": stages,
        "source_lang": settings.source_lang,
        "target_lang": settings.target_lang,
    }


class JobStore:
    """In-memory jobs mirrored to job.json, so results survive a server restart."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def dir(self, job_id: str) -> Path:
        return self.root / job_id

    def save(self, job: Job) -> None:
        with self.lock:
            self.jobs[job.id] = job
            d = self.dir(job.id)
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "job.json.tmp"
            tmp.write_text(job.model_dump_json())
            tmp.replace(d / "job.json")

    def prune(self, keep: int) -> list[str]:
        """Delete the oldest finished jobs beyond `keep` (each holds ~5 full-size PNGs)."""
        if keep <= 0 or not self.root.exists():
            return []
        dirs = [d for d in self.root.iterdir() if d.is_dir() and JOB_ID.match(d.name)]
        if len(dirs) <= keep:
            return []

        def mtime(d: Path) -> float:
            try:
                return (d / "job.json").stat().st_mtime
            except OSError:
                return d.stat().st_mtime

        dirs.sort(key=mtime)
        removed = []
        for d in dirs[: len(dirs) - keep]:
            with self.lock:
                live = self.jobs.get(d.name)
                if live is not None and live.status in ("queued", "running"):
                    continue
                self.jobs.pop(d.name, None)
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
        if removed:
            log.info("removed %d old web jobs (keeping %d)", len(removed), keep)
        return removed

    def get(self, job_id: str) -> Job:
        if not JOB_ID.match(job_id):
            raise HTTPException(404, "job not found")
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            f = self.dir(job_id) / "job.json"
            if not f.exists():
                raise HTTPException(404, "job not found")
            job = Job.model_validate_json(f.read_text())
            if job.status in ("queued", "running"):
                # only this process runs jobs: one on disk but not in memory was cut short by a
                # restart and would otherwise poll as "running" forever
                job.status, job.stage = "failed", None
                job.error = "interrupted: the server stopped before the job finished; run it again"
                self.save(job)
            with self.lock:
                self.jobs[job_id] = job
        return job


def _json_field(raw: str | None, model: type[BaseModel], what: str) -> Any:
    if not raw or not raw.strip():
        return model()
    try:
        return model.model_validate_json(raw)
    except ValidationError as e:
        raise HTTPException(400, f"{what}: {e.errors(include_url=False)}") from None


def _check_methods(cfg: PipelineConfig, settings: Settings, stages: tuple[str, ...]) -> None:
    for stage in stages:
        name = getattr(cfg, stage)
        try:
            info = REGISTRIES[stage].get(name)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        ok, reason = info.status(settings)
        if not ok:
            raise HTTPException(400, f"{stage} '{name}' unavailable: {reason}")
    for stage, opts in cfg.options.items():
        if stage not in STAGES or not isinstance(opts, dict):
            raise HTTPException(400, f"options must be {{stage: {{key: value}}}}, got {stage!r}")


def _write_png(path: Path, img: np.ndarray) -> None:
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"cannot write {path}")


def create_app(settings: Settings | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    settings = settings or get_settings()
    pipe = pipeline or WebPipeline(settings)
    store = JobStore(Path(settings.storage_dir) / "web")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="terjemahin",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.store, app.state.pipeline, app.state.executor = store, pipe, executor

    def run_job(job: Job) -> None:
        d = store.dir(job.id)
        t0 = time.perf_counter()

        def on_stage(stage: str) -> None:
            job.stage = stage
            store.save(job)

        _progress.job, _progress.raw_mask = on_stage, None
        job.status = "running"
        store.save(job)
        try:
            page = cv2.imread(str(d / "input.png"))
            res = pipe.run(page, job.context, job.pipeline, debug=True)
            for name, img in {**res.debug, "final": res.image}.items():
                _write_png(d / f"{name}.png", img)
            if _progress.raw_mask is not None:
                _write_png(d / "raw_mask.png", _progress.raw_mask)
            (d / "result.json").write_text(res.result.model_dump_json(indent=2))
            cfg = res.result.pipeline or pipe.resolve(job.pipeline)
            job.result, job.pipeline = res.result, cfg
            job.erase_key = erase_key(res.result.bubbles, cfg)
            job.status, job.stage, job.version = "done", None, job.version + 1
        except Exception as e:
            log.exception("job %s failed", job.id)
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
        finally:
            _progress.job = _progress.raw_mask = None
            job.elapsed = round(time.perf_counter() - t0, 3)
            store.save(job)

    def run_compose(
        job: Job, bubbles: list[Bubble], dropped: list[Bubble], cfg: PipelineConfig
    ) -> None:
        d = store.dir(job.id)
        page = cv2.imread(str(d / "input.png"))
        key = erase_key(bubbles, cfg)
        t: dict[str, float] = {}
        if key == job.erase_key and (d / "inpainted.png").exists():
            inpainted = cv2.imread(str(d / "inpainted.png"))
            t0 = time.perf_counter()
            pipe.check(cfg, ("renderer",))
            render = [b for b in pipe.sanitize(page, bubbles) if b.target_text.strip()]
            out = pipe.stage("renderer", cfg).render(inpainted, render, page)
            t["render"] = time.perf_counter() - t0
            dbg = {"final": out}
        else:
            raw = d / "raw_mask.png"
            raw_mask = cv2.imread(str(raw), cv2.IMREAD_GRAYSCALE) if raw.exists() else None
            dbg = {}
            out = pipe.compose(page, bubbles, cfg, raw_mask=raw_mask, timings=t, debug=dbg)
        for name, img in dbg.items():
            _write_png(d / f"{name}.png", img)
        assert job.result is not None
        timings = {**job.result.timings, **{f"compose_{k}": v for k, v in t.items()}}
        job.result = job.result.model_copy(
            update={"bubbles": bubbles, "pipeline": cfg, "timings": timings}
        )
        (d / "result.json").write_text(job.result.model_dump_json(indent=2))
        job.pipeline, job.erase_key, job.dropped = cfg, key, dropped
        job.version += 1
        store.save(job)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/methods")
    def methods() -> dict[str, Any]:
        return list_methods(settings)

    @app.post("/api/translate")
    async def translate(
        file: Annotated[UploadFile, File()],
        pipeline: Annotated[str | None, Form()] = None,
        context: Annotated[str | None, Form()] = None,
        wait: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        """Multipart: file, pipeline (PipelineConfig JSON), context (PageContext JSON).

        wait=true (default) returns the finished job; wait=false returns at once and the client
        polls GET /api/jobs/{id}. A string, because FastAPI reads a missing bool form as false.
        """
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "image larger than 25 MB")
        if not data:
            raise HTTPException(400, "file is empty")
        try:
            page = decode_image(data, settings.max_pixels, name="file")
        except ImageError as e:
            raise HTTPException(413 if "limit" in str(e) else 400, str(e)) from None
        cfg = pipe.resolve(_json_field(pipeline, PipelineConfig, "pipeline"))
        _check_methods(cfg, settings, STAGES)
        for what, lang in (("source_lang", cfg.source_lang), ("target_lang", cfg.target_lang)):
            try:
                lang_code(lang)
            except ValueError as e:
                raise HTTPException(400, f"{what}: {e}") from None
        job = Job(
            id=secrets.token_hex(8),
            pipeline=cfg,
            context=_json_field(context, PageContext, "context"),
        )
        store.dir(job.id).mkdir(parents=True, exist_ok=True)
        _write_png(store.dir(job.id) / "input.png", page)
        store.save(job)
        store.prune(settings.web_keep_jobs)
        fut = asyncio.get_running_loop().run_in_executor(executor, run_job, job)
        if (wait or "true").strip().lower() not in {"0", "false", "no", "off"}:
            await fut
        return job.public()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return store.get(job_id).public()

    @app.post("/api/compose")
    async def compose(req: ComposeRequest) -> dict[str, Any]:
        """Re-typeset edited bubbles: mask -> inpaint -> render, no detector/OCR/translator.

        If the erased regions and mask/inpaint methods are unchanged (only text was edited),
        the stored inpainted page is reused and only the renderer runs.
        """
        job = store.get(req.job_id)
        if job.status != "done" or job.result is None:
            raise HTTPException(409, f"job is {job.status}")
        base = job.pipeline or pipe.resolve(None)
        over = req.pipeline or PipelineConfig()
        cfg = base.model_copy(deep=True)
        for stage in COMPOSE_STAGES:
            if getattr(over, stage):
                setattr(cfg, stage, getattr(over, stage))
            if stage in over.options:
                cfg.options[stage] = over.options[stage]
        _check_methods(cfg, settings, COMPOSE_STAGES)
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(executor, run_compose, job, req.bubbles, req.dropped, cfg)
        except Exception as e:
            log.exception("compose %s failed", job.id)
            raise HTTPException(500, f"{type(e).__name__}: {e}") from None
        return job.public()

    @app.get("/api/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str) -> FileResponse:
        store.get(job_id)
        path = store.dir(job_id) / name
        if not FILE_NAME.match(name) or not path.exists():
            raise HTTPException(404, "file not found")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-cache"})

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str) -> FileResponse:
        job = store.get(job_id)
        path = store.dir(job_id) / "final.png"
        if job.status != "done" or not path.exists():
            raise HTTPException(409, f"job is {job.status}")
        return FileResponse(path, media_type="image/png", filename=f"terjemahin-{job_id}.png")

    return app
