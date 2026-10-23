"""python -m terjemahin_worker.eval bench --page PAGE --gt GT.json --out out/bench [--matrix]"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from ..config import get_settings
from ..models import PipelineConfig
from ..registry import STAGES

app = typer.Typer(no_args_is_help=True, help="terjemahin evaluation harness")


@app.callback()
def _setup(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


Opt = typer.Option


@app.command("bench")
def bench_cmd(
    page: Annotated[Path, Opt("--page", exists=True)],
    gt: Annotated[Path | None, Opt("--gt", exists=True, help="GT JSON: bubbles ja/id_ref")] = None,
    out: Annotated[Path, Opt("--out")] = Path("out/bench"),
    matrix: Annotated[bool, Opt("--matrix", help="vary every stage over its methods")] = False,
    stages: Annotated[str, Opt("--stages", help="comma list of stages for --matrix")] = "",
    detector: Annotated[str | None, Opt("--detector", "-d")] = None,
    ocr: Annotated[str | None, Opt("--ocr")] = None,
    translator: Annotated[str | None, Opt("--translator", "-t")] = None,
    mask_refiner: Annotated[str | None, Opt("--mask")] = None,
    inpainter: Annotated[str | None, Opt("--inpainter", "-i")] = None,
    renderer: Annotated[str | None, Opt("--renderer", "-r")] = None,
    opt: Annotated[list[str] | None, Opt("--opt", help="stage.key=value (repeatable)")] = None,
    allow_downloads: Annotated[bool, Opt(help="include methods that download GBs first")] = False,
    network: Annotated[bool, Opt(help="include network / local-server methods")] = False,
    warm: Annotated[bool, Opt(help="re-time the baseline without caches (warm latency)")] = True,
) -> None:
    """Score detect / OCR / translate / inpaint / render on a page; write report.md + JSON."""
    from ..cli import _parse_opts
    from .bench import bench, markdown

    chosen = [s.strip() for s in stages.split(",") if s.strip()]
    bad = [s for s in chosen if s not in STAGES]
    if bad:
        raise typer.BadParameter(f"unknown stage(s) {bad}, choose from {', '.join(STAGES)}")
    base = PipelineConfig(
        detector=detector,
        ocr=ocr,
        translator=translator,
        mask_refiner=mask_refiner,
        inpainter=inpainter,
        renderer=renderer,
        options=_parse_opts(opt or []),
    )
    report = bench(
        page, gt, out, get_settings(), matrix, chosen or None, base, allow_downloads, network, warm
    )
    md = markdown(report)
    typer.echo(md.split("\n## Baseline per bubble")[0])
    typer.echo(f"\nwrote {out / 'report.md'}, {out / 'report.json'}, {out / 'contact_sheet.jpg'}")
    base_run = report["runs"][0]
    if base_run.get("status") == "ok":
        typer.echo(json.dumps(base_run["checks"]))


if __name__ == "__main__":
    app()
