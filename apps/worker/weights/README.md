# Model weights (not in git)

| File | Used by | Source |
|---|---|---|
| `manga-text-yolo11n.onnx` | `DETECTOR=yolo` (default) | `python training/train.py` on your 4060 |
| `comictextdetector.pt.onnx` | `DETECTOR=ctd` (fallback) | https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx |

Quick start before you have trained YOLO:

```sh
curl -L -o weights/comictextdetector.pt.onnx \
  https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx
DETECTOR=ctd terjemahin-worker detect page.jpg
```
