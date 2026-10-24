# Model weights (not in git)

| File | Used by | Source |
|---|---|---|
| `koharu-comictextdetector.onnx` | `-d ctd` (default) | downloaded on first use (`terjemahin_worker/weights.py`) |
| `comictextdetector-blk.onnx` | `-d ctd_blk` (boxes only) | `python training/cli.py prune-ctd` from the file above |
| `manga-text-yolo11n.onnx` | `-d yolo` (boxes only) | `python training/cli.py detector ...` (see training/README.md) |
| `comictextdetector.pt.onnx` | optional input to `prune-ctd` | https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx |

Box-only CTD:

```sh
python training/cli.py prune-ctd   # first of comictextdetector.pt.onnx / koharu-comictextdetector.onnx
terjemahin-worker translate page.png -d ctd_blk
```

`ctd_blk` returns ctd's raw YOLO blocks (identical boxes to `ctd` with `--opt detector.heads=blk`;
`ctd` itself refines them with its text lines) without line quads or a glyph mask, so downstream
stages estimate those from pixels.
