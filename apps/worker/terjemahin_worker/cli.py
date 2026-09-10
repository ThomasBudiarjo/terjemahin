from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import typer

from .config import get_settings
from .models import PageContext, PageResult

app = typer.Typer(no_args_is_help=True, help="terjemahin worker")


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def translate(
    image: Path = typer.Argument(..., exists=True, help="Manga page (png/jpg)"),
    out: Path = typer.Option(Path("out/page.png"), "--out", "-o"),
    context: Path | None = typer.Option(None, "--context", help="PageContext JSON"),
    dump: bool = typer.Option(True, help="Also write <out>.json with bubbles"),
) -> None:
    """Translate a single page end-to-end."""
    from .pipeline import Pipeline

    settings = get_settings()
    ctx = PageContext.model_validate_json(context.read_text()) if context else PageContext()
    page = cv2.imread(str(image))
    if page is None:
        raise typer.BadParameter(f"cannot read {image}")

    result_img, result = Pipeline.from_settings(settings).run(page, ctx)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), result_img)
    if dump:
        out.with_suffix(".json").write_text(result.model_dump_json(indent=2))
    typer.echo(f"wrote {out} ({len(result.bubbles)} bubbles)")


@app.command()
def compose(
    image: Path = typer.Argument(..., exists=True),
    result: Path = typer.Argument(..., exists=True, help="PageResult JSON, possibly hand-edited"),
    out: Path = typer.Option(Path("out/page.png"), "--out", "-o"),
) -> None:
    """Re-typeset from an edited PageResult without calling the detector or LLM."""
    from .pipeline import Pipeline
    from .typeset import Typesetter

    settings = get_settings()
    page = cv2.imread(str(image))
    res = PageResult.model_validate_json(result.read_text())
    pipe = Pipeline(detector=None, translator=None, typesetter=Typesetter(settings.font_path))  # type: ignore[arg-type]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), pipe.compose(page, res.bubbles))
    typer.echo(f"wrote {out}")


@app.command()
def detect(
    image: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("out/detect.png"), "--out", "-o"),
) -> None:
    """Debug: draw detected regions with their reading-order index."""
    from .detect import build_detector, sort_reading_order

    settings = get_settings()
    page = cv2.imread(str(image))
    regions = sort_reading_order(build_detector(settings).detect(page))
    for r in regions:
        b = r.bbox
        cv2.rectangle(page, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), 2)
        cv2.putText(page, str(r.id), (b.x1, max(12, b.y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), page)
    typer.echo(json.dumps([r.model_dump() for r in regions], indent=2))


@app.command()
def serve() -> None:
    """Consume jobs from LavinMQ until interrupted."""
    from .consumer import run

    run(get_settings())
