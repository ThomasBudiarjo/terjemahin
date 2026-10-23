"""One entry point for every training-side script: `python training/cli.py <command> [args]`.

Each command runs its script exactly as `python training/<script>.py [args]` would (same argv,
`__name__ == "__main__"`), so script flags and `--help` work unchanged. Scripts are only loaded
when their command runs: listing commands needs no torch / ultralytics.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# command -> (script, one-line help); in the order a full training run uses them
COMMANDS: dict[str, tuple[str, str]] = {
    "synth": ("synth.py", "synthetic pages -> detector / OCR / inpaint training data"),
    "download-pool": ("download_pool.py", "unlabeled manga pages for pseudo-labelling"),
    "pseudo-label": ("pseudo_label.py", "label a page pool with a teacher detector (YOLO dataset)"),
    "prepare-manga109": ("prepare_manga109.py", "Manga109 annotations -> YOLO text dataset"),
    "detector": ("train.py", "fine-tune YOLO11n text detector and export ONNX"),
    "ocr": ("train_ocr.py", "fine-tune the OCR model on text-line crops"),
    "translator": ("train_translator.py", "fine-tune NLLB / M2M100 on a parallel corpus"),
    "inpaint": ("train_inpaint.py", "fine-tune the AOT inpainter on (clean, mask, texted)"),
    "export-inpaint": ("export_inpaint_onnx.py", "AOT checkpoint -> ONNX (lama interface)"),
    "export-ocr": ("export_ocr_onnx.py", "OCR checkpoint -> ONNX"),
    "prune-ctd": ("prune_ctd.py", "strip comic-text-detector to its box head"),
}


def usage() -> str:
    w = max(map(len, COMMANDS))
    rows = []
    for name, (script, desc) in COMMANDS.items():
        missing = "" if (HERE / script).exists() else "  [not available: missing]"
        rows.append(f"  {name:<{w}}  {desc} ({script}){missing}")
    return "usage: python training/cli.py <command> [args...]\n\ncommands:\n" + "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help", "list"):
        print(usage())
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command '{cmd}'\n\n{usage()}", file=sys.stderr)
        return 2
    script = HERE / COMMANDS[cmd][0]
    if not script.exists():
        print(f"'{cmd}' needs {script.name}, which is not in this checkout", file=sys.stderr)
        return 2
    old_argv, old_path = sys.argv, list(sys.path)
    sys.argv = [str(script), *rest]
    sys.path.insert(0, str(HERE))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.argv, sys.path[:] = old_argv, old_path
    return 0


if __name__ == "__main__":
    sys.exit(main())
