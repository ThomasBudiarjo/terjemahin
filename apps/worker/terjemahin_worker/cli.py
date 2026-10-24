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
    """A configuration error found before the pipeline ran: one line, not a traceback."""
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
    res = pipe.run(page, ctx, cfg, debug=debug_dir is not None)
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
    from .detect import sort_reading_order
    from .pipeline import Pipeline, draw_regions

    pipe = Pipeline(get_settings())
    cfg = pipe.resolve(PipelineConfig(detector=detector, options=_parse_opts(opt)))
    page = _read(image)
    try:
        pipe.check(cfg, ("detector",))
    except (ValueError, FileNotFoundError) as e:
        raise _fail(e) from None
    det = pipe.stage("detector", cfg).detect(page)
    regions = sort_reading_order(det.regions, image=page)  # the ids translate assigns
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
