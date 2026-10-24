from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import typer

from .config import get_settings
from .imageio import ImageError, read_image
from .models import PageContext, PageResult, PipelineConfig
from .registry import STAGES

app = typer.Typer(no_args_is_help=True, help="terjemahin worker")

STAGE_OPTS = {
    "detector": typer.Option(None, "--detector", "-d", help="see `methods`"),
    "ocr": typer.Option(None, "--ocr"),
    "translator": typer.Option(None, "--translator", "-t"),
    "mask_refiner": typer.Option(None, "--mask"),
    "inpainter": typer.Option(None, "--inpainter", "-i"),
    "renderer": typer.Option(None, "--renderer", "-r"),
}


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# --opt stage names as the method flags spell them (--mask, -i/--inpainter, -r/--renderer)
STAGE_ALIASES = {
    "mask": "mask_refiner", "refiner": "mask_refiner", "inpaint": "inpainter",
    "render": "renderer", "detect": "detector", "translate": "translator", "tr": "translator",
}
_WORDS = {"true": True, "false": False, "yes": True, "no": False, "on": True, "off": False,
          "none": None, "null": None}


def _parse_opts(opts: list[str]) -> dict[str, dict[str, Any]]:
    """`--opt inpainter.merge=32 --opt detector.conf=0.3` -> {"inpainter": {"merge": 32}, ...}

    Values are JSON (numbers, [lists], {objects}); True/False, yes/no, on/off (any case) are
    booleans and None/null is null, so `renderer.uppercase=False` really is false. The stage
    is a registry stage or the name of its flag (mask, inpaint, render ...); anything else is
    an error, not an option silently ignored."""
    out: dict[str, dict[str, Any]] = {}
    for o in opts:
        key, _, raw = o.partition("=")
        stage, _, name = key.strip().partition(".")
        if not name or not raw:
            raise typer.BadParameter(f"--opt expects stage.key=value, got {o!r}")
        stage = STAGE_ALIASES.get(stage, stage)
        if stage not in STAGES:
            raise typer.BadParameter(
                f"--opt {o!r}: unknown stage {stage!r}, choose from: {', '.join(STAGES)} "
                f"(or {', '.join(STAGE_ALIASES)})"
            )
        word = raw.strip().lower()
        if word in _WORDS:
            val: Any = _WORDS[word]
        else:
            try:
                val = json.loads(raw)
            except json.JSONDecodeError:
                val = raw
        out.setdefault(stage, {})[name] = val
    return out


def _read(image: Path):
    try:
        return read_image(image, get_settings().max_pixels)
    except ImageError as e:
        raise typer.BadParameter(str(e)) from None


def _fail(e: Exception) -> typer.Exit:
    """A configuration error (found before the pipeline ran, or an unreachable LLM server):
    one line, not a traceback."""
    typer.echo(f"error: {e}", err=True)
    return typer.Exit(2)


def _config(
    detector, ocr, translator, mask_refiner, inpainter, renderer, opt, source_lang, target_lang
) -> PipelineConfig:
    return PipelineConfig(
        detector=detector,
        ocr=ocr,
        translator=translator,
        mask_refiner=mask_refiner,
        inpainter=inpainter,
        renderer=renderer,
        source_lang=source_lang,
        target_lang=target_lang,
        options=_parse_opts(opt),
    )


def _write_debug(debug_dir: Path, debug: dict) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, img in debug.items():
        cv2.imwrite(str(debug_dir / f"{name}.png"), img)


@app.command()
def translate(
    image: Path = typer.Argument(..., exists=True, help="Manga page (png/jpg)"),
    out: Path = typer.Option(Path("out/page.png"), "--out", "-o"),
    context: Path | None = typer.Option(None, "--context", help="PageContext JSON"),
    dump: bool = typer.Option(True, help="Also write <out>.json with bubbles"),
    debug_dir: Path | None = typer.Option(None, "--debug-dir", help="Write per-stage images"),
    detector: str | None = STAGE_OPTS["detector"],
    ocr: str | None = STAGE_OPTS["ocr"],
    translator: str | None = STAGE_OPTS["translator"],
    mask_refiner: str | None = STAGE_OPTS["mask_refiner"],
    inpainter: str | None = STAGE_OPTS["inpainter"],
    renderer: str | None = STAGE_OPTS["renderer"],
    opt: list[str] = typer.Option([], "--opt", help="stage.key=value (repeatable)"),
    source_lang: str = typer.Option("ja", "--from"),
    target_lang: str = typer.Option("id", "--to"),
) -> None:
    """Translate a single page end-to-end."""
    from .pipeline import Pipeline

    ctx = PageContext.model_validate_json(context.read_text()) if context else PageContext()
    page = _read(image)
    cfg = _config(
        detector, ocr, translator, mask_refiner, inpainter, renderer, opt, source_lang, target_lang
    )
    pipe = Pipeline(get_settings())
    cfg = pipe.resolve(cfg)
    try:
        pipe.check(cfg)  # unknown methods, bad options or languages: before any stage runs
    except (ValueError, FileNotFoundError) as e:
        raise _fail(e) from None
    from .translate.llm_common import LLMError

    try:
        res = pipe.run(page, ctx, cfg, debug=debug_dir is not None)
    except LLMError as e:  # server down, bad key or model: the message says which, one line
        raise _fail(e) from None
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), res.image)
    if dump:
        out.with_suffix(".json").write_text(res.result.model_dump_json(indent=2))
    if debug_dir:
        _write_debug(debug_dir, res.debug)
    typer.echo(f"wrote {out} ({len(res.result.bubbles)} bubbles)")


@app.command()
def compose(
    image: Path = typer.Argument(..., exists=True),
    result: Path = typer.Argument(..., exists=True, help="PageResult JSON, possibly hand-edited"),
    out: Path = typer.Option(Path("out/page.png"), "--out", "-o"),
    mask_refiner: str | None = STAGE_OPTS["mask_refiner"],
    inpainter: str | None = STAGE_OPTS["inpainter"],
    renderer: str | None = STAGE_OPTS["renderer"],
    opt: list[str] = typer.Option([], "--opt"),
) -> None:
    """Re-typeset from an edited PageResult without calling the detector, OCR or translator."""
    from .pipeline import Pipeline

    page = _read(image)
    res = PageResult.model_validate_json(result.read_text())
    if (res.width, res.height) != (page.shape[1], page.shape[0]):
        raise typer.BadParameter(
            f"{result} is for a {res.width}x{res.height} page, {image} is "
            f"{page.shape[1]}x{page.shape[0]}: the boxes would land in the wrong places"
        )
    base = res.pipeline or PipelineConfig()
    cfg = base.model_copy(
        update={
            k: v
            for k, v in {
                "mask_refiner": mask_refiner, "inpainter": inpainter, "renderer": renderer
            }.items()
            if v
        }
    )
    cfg.options.update(_parse_opts(opt))
    pipe = Pipeline(get_settings())
    cfg = pipe.resolve(cfg)
    try:
        pipe.check(cfg, ("mask_refiner", "inpainter", "renderer"))
    except (ValueError, FileNotFoundError) as e:
        raise _fail(e) from None
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), pipe.compose(page, res.bubbles, cfg))
    typer.echo(f"wrote {out}")


@app.command()
def detect(
    image: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("out/detect.png"), "--out", "-o"),
    detector: str | None = STAGE_OPTS["detector"],
    opt: list[str] = typer.Option([], "--opt"),
) -> None:
    """Debug: draw detected regions with their reading-order index."""
    from .pipeline import Pipeline, draw_regions

    pipe = Pipeline(get_settings())
    cfg = pipe.resolve(PipelineConfig(detector=detector, options=_parse_opts(opt)))
    page = _read(image)
    try:
        pipe.check(cfg, ("detector",))
    except (ValueError, FileNotFoundError) as e:
        raise _fail(e) from None
    det, regions = pipe.detect(page, cfg)  # the regions and ids translate uses
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), draw_regions(page, regions))
    if det.mask is not None:
        cv2.imwrite(str(out.with_name(out.stem + "-mask.png")), det.mask)
    typer.echo(json.dumps([r.model_dump() for r in regions], indent=2))


def _factory_options(factory: Any) -> dict[str, Any]:
    """Named options of a method factory with their defaults (methods taking **opts list
    only what their signature names; see the method's docs for the rest)."""
    import inspect

    try:
        params = list(inspect.signature(factory).parameters.values())[1:]  # after settings
    except (TypeError, ValueError):
        return {}
    out: dict[str, Any] = {}
    for p in params:
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        d = None if p.default is p.empty else p.default
        out[p.name] = d if isinstance(d, (bool, int, float, str, type(None))) else repr(d)
    return out


@app.command()
def methods(as_json: bool = typer.Option(False, "--json")) -> None:
    """List every method of every stage and whether it can run here."""
    from .registry import load_all

    settings = get_settings()
    rows = []
    for stage, reg in load_all().items():
        for name in reg.names():
            info = reg.get(name)
            ok, reason = info.status(settings)
            rows.append(
                {
                    "stage": stage,
                    "name": name,
                    "available": ok,
                    "reason": reason,
                    "offline": info.offline,
                    "description": info.description,
                    "default": getattr(settings, stage) == name,
                    "options": _factory_options(info.factory),
                }
            )
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    for r in rows:
        mark = "*" if r["default"] else " "
        avail = "ok " if r["available"] else "-- "
        typer.echo(f"{r['stage']:<13}{mark}{r['name']:<18}{avail}{r['description']}"
                   + ("" if r["available"] else f"  [{r['reason']}]"))


def _hf_models(stage: str, name: str) -> list[str]:
    """Hugging Face hub repos a method loads with from_pretrained (default options)."""
    if stage == "ocr" and name == "manga_ocr_hf":
        from .ocr.manga_ocr_hf import REPO

        return [REPO]
    if stage == "translator":
        from .translate import hf_seq2seq, local_llm

        if name in hf_seq2seq._KNOWN:
            return [hf_seq2seq._KNOWN[name][0]]
        if name == "hymt":
            from .translate.hymt import DEFAULT_MODEL

            return [DEFAULT_MODEL]
        if name == "local_llm":  # pivot en->target through nllb, per-item fallback m2m100
            return [local_llm.DEFAULT_MODEL] + _hf_models(stage, "nllb") + _hf_models(stage, "m2m100")
        if name == "chain":  # the translators of its default steps
            from .registry import TRANSLATORS
            from .translate.chain import parse_steps

            steps = _factory_options(TRANSLATORS.get("chain").factory).get("steps")
            return [r for s in parse_steps(steps) for r in _hf_models(stage, s["translator"])]
    return []


# methods whose weights are produced locally by training/ (Settings attribute of the path)
_LOCAL_WEIGHTS = {("detector", "yolo"): "yolo_weights", ("detector", "ctd_blk"): "ctd_weights"}

# files transformers needs besides the weights (config, tokenizer, generation config)
_HF_SIDE_FILES = (".json", ".txt", ".model", ".tiktoken")


def _hf_fetch(repo: str, offline: bool) -> tuple[Path, bool]:
    """(snapshot dir, downloaded). The local cache is checked first without any network call;
    a download takes the root config/tokenizer files plus one weight format (safetensors if
    the repo has it, else pytorch_model*.bin), which is what from_pretrained loads."""
    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache

    def has_weights(p: Path) -> bool:
        for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if (p / index).exists():  # sharded: every shard must be there
                shards = set(json.loads((p / index).read_text())["weight_map"].values())
                return all((p / s).exists() for s in shards)
        return any(p.glob("*.safetensors")) or any(p.glob("pytorch_model*.bin"))

    # not snapshot_download(local_files_only=True): hub 2.x calls a snapshot without README.md
    # or the unused rust_model.ot incomplete; what from_pretrained needs is config + weights
    cached = try_to_load_from_cache(repo, "config.json")
    if isinstance(cached, str) and has_weights(Path(cached).parent):
        return Path(cached).parent, False
    if offline:
        raise FileNotFoundError(f"{repo} is not in the Hugging Face cache and downloads are off")
    root = [f for f in HfApi().list_repo_files(repo) if "/" not in f]
    weights = [f for f in root if f.endswith(".safetensors")] or [
        f for f in root if f.startswith("pytorch_model") and f.endswith(".bin")
    ]
    if not weights:
        raise FileNotFoundError(f"{repo}: no safetensors or pytorch_model*.bin at the repo root")
    keep = [f for f in root if f.endswith(_HF_SIDE_FILES)] + weights
    return Path(snapshot_download(repo, allow_patterns=keep)), True


def _fetch_targets(targets: list[str], every: bool, settings: Any) -> tuple[set, set]:
    """(weight keys, (stage, method) pairs) for fetch-weights."""
    from .registry import load_all
    from .weights import WEIGHTS

    regs = load_all()
    keys: set[str] = set(WEIGHTS) if every else set()
    methods: set[tuple[str, str]] = set()
    if not targets and not every:  # the configured defaults (+ the non-Japanese OCR fallback)
        methods = {(s, getattr(settings, s)) for s in regs}
        methods.add(("ocr", settings.ocr_non_japanese))
    for t in targets:
        stage, sep, name = t.replace("=", ":").partition(":")
        if sep:
            stage = STAGE_ALIASES.get(stage, stage)
            if stage not in regs:
                raise typer.BadParameter(f"unknown stage {stage!r} in {t!r}")
            regs[stage].get(name)  # unknown method: ValueError with the choices
            methods.add((stage, name))
            continue
        if t in WEIGHTS:
            keys.add(t)
            continue
        hits = [s for s, reg in regs.items() if t in reg.methods]
        if not hits:
            raise typer.BadParameter(
                f"{t!r} is neither a weight key ({', '.join(WEIGHTS)}) nor a method name "
                "(see `methods`)"
            )
        if len(hits) > 1:
            raise typer.BadParameter(f"{t!r} is a method of {', '.join(hits)}: write stage:{t}")
        methods.add((hits[0], t))
    for stage, name in methods:
        keys.update(regs[stage].get(name).weights)
    return keys, methods


@app.command("fetch-weights")
def fetch_weights(
    targets: list[str] = typer.Argument(
        None, help="weight keys (ctd, lama_manga, ...) or methods (nllb, ocr:48px). "
        "Default: the configured default method of every stage."
    ),
    every: bool = typer.Option(False, "--all", help="every downloadable weight file"),
    hf: bool = typer.Option(True, "--hf/--no-hf", help="also cache Hugging Face models"),
) -> None:
    """Download model files now instead of on first use (e.g. while building an image).

    Files already present are left alone (no network call). With TERJEMAHIN_OFFLINE or
    HF_HUB_OFFLINE set it only checks: exit 1 if anything is missing."""
    import os

    from .registry import load_all
    from .weights import WEIGHTS, ensure

    settings = get_settings()
    offline = any(
        os.environ.get(v, "") not in ("", "0") for v in ("TERJEMAHIN_OFFLINE", "HF_HUB_OFFLINE")
    )
    try:
        keys, methods = _fetch_targets(list(targets or []), every, settings)
    except ValueError as e:
        raise _fail(e) from None
    failed = 0
    for key in sorted(keys):
        path = Path(settings.weights_dir) / WEIGHTS[key].filename
        had = path.exists() and path.stat().st_size > 0
        try:
            path = ensure(key, settings)
        except OSError as e:  # report every file, then fail (URLError is an OSError)
            failed += 1
            typer.echo(f"MISSING  weight {key:<18} {path}: {e}", err=True)
            continue
        state = "present " if had else "fetched "
        typer.echo(f"{state} weight {key:<18} {path} ({path.stat().st_size / 1e6:.1f} MB)")
    regs = load_all()
    runnable = set()
    for stage, name in sorted(methods):
        attr = _LOCAL_WEIGHTS.get((stage, name))  # made by training/, no URL to fetch from
        if attr and not Path(getattr(settings, attr)).exists():
            failed += 1
            typer.echo(f"MISSING  local  {attr:<18} {getattr(settings, attr)}: not "
                       "downloadable, see weights/README.md", err=True)
        ok, reason = regs[stage].get(name).status(settings)
        if ok:
            runnable.add((stage, name))
        else:  # e.g. torch not installed: its HF models would be useless (and need hf_hub)
            typer.echo(f"note     {stage}:{name} cannot run here ({reason}), its Hugging Face "
                       "models are skipped", err=True)
    repos = sorted({r for s, n in runnable for r in _hf_models(s, n)}) if hf else []
    for repo in repos:
        try:
            path, fetched = _hf_fetch(repo, offline)
        except OSError as e:  # missing offline, HTTP errors (HfHubHTTPError is an OSError)
            failed += 1
            typer.echo(f"MISSING  hf     {repo:<18} {e}", err=True)
            continue
        typer.echo(f"{'fetched ' if fetched else 'present '} hf     {repo} {path}")
    if failed:
        raise typer.Exit(1)


@app.command()
def serve() -> None:
    """Consume jobs from LavinMQ until interrupted."""
    from .consumer import run

    run(get_settings())


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
) -> None:
    """Serve the web UI: upload a page, pick methods per stage, edit and re-typeset bubbles."""
    import uvicorn

    from .web.app import create_app

    uvicorn.run(create_app(get_settings()), host=host, port=port, log_level="info")
