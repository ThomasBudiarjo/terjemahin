# Model weights (not in git)

| File | Used by | Source |
|---|---|---|
| `manga-text-yolo11n.onnx` | `DETECTOR=yolo` (default) | `python training/train.py` on your 4060 |
| `comictextdetector-blk.onnx` | `DETECTOR=ctd` (fallback) | `python training/prune_ctd.py` from the file below |
| `comictextdetector.pt.onnx` | input to `prune_ctd.py` only | https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx |

Quick start before you have trained YOLO:

```sh
curl -L -o weights/comictextdetector.pt.onnx \
  https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx
uv pip install -p .venv/bin/python onnx
python training/prune_ctd.py            # -> weights/comictextdetector-blk.onnx (27MB)
DETECTOR=ctd terjemahin-worker detect page.jpg
```

Always run the pruned `-blk` file in the worker. The full export also carries the mask and
line heads, and their ConvTranspose layers make onnxruntime CPU take ~29s per page instead of
~0.15s, with identical boxes.
