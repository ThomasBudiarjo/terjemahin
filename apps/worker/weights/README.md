# Model weights (not in git)

Downloadable files are fetched into this folder (`WEIGHTS_DIR`) the first time a method needs
them, or ahead of time:

```sh
terjemahin-worker fetch-weights                 # the default method of every stage + 48px OCR
terjemahin-worker fetch-weights --all --no-hf   # every file below, no Hugging Face models
terjemahin-worker fetch-weights lama ocr:48px   # weight keys or method names
TERJEMAHIN_OFFLINE=1 terjemahin-worker fetch-weights   # check only: exit 1 if anything is missing
```

## Downloaded automatically (`terjemahin_worker/weights.py`)

| key | file | size | used by | source |
|---|---|---|---|---|
| `ctd` | `koharu-comictextdetector.onnx` | 95 MB | `-d ctd` (default) | [mayocream/koharu](https://huggingface.co/mayocream/koharu) `comictextdetector.onnx` |
| `manga_ocr_onnx` | `koharu-manga-ocr.onnx` | 461 MB | `--ocr manga_ocr` (default) | mayocream/koharu `manga-ocr.onnx` |
| `manga_ocr_vocab` | `koharu-vocab.txt` | 24 KB | `--ocr manga_ocr` | mayocream/koharu `vocab.txt` |
| `lama_manga` | `koharu-lama-manga.onnx` | 207 MB | `-i lama_manga` (default) | mayocream/koharu `lama-manga.onnx` |
| `lama` | `lama-carve.onnx` | 208 MB | `-i lama` | [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) `lama_fp32.onnx` |
| `ocr_48px` | `ocr_ar_48px.ckpt` | 204 MB | `--ocr 48px` (also the non-Japanese default, and the eval harness's read-back judge) | [zyddnys/manga-image-translator](https://huggingface.co/zyddnys/manga-image-translator) |
| `ocr_48px_alphabet` | `alphabet-all-v7.txt` | 187 KB | `--ocr 48px` | zyddnys/manga-image-translator |

`-d ctd` runs all three heads of comic-text-detector (boxes, DBNet text lines, glyph mask) in
about 0.6 s per page on CPU. That export stores ~40% of its weights as float32 denormals; the
session enables onnxruntime's denormals-as-zero (`detect/ctd_full.py`), which brings one
1024x1024 forward pass from 5.1 s to 0.5 s here (4 cores; ~9 s was seen under load) with
identical outputs on page01. The old advice to prune the graph for speed is obsolete: the
box-only variants below are faster still but give up the line quads and the glyph mask that
the default mask refiner uses.

## Made locally (no URL; `fetch-weights` only reports them missing)

| file | setting | used by | how to make it |
|---|---|---|---|
| `comictextdetector-blk.onnx` | `CTD_WEIGHTS` | `-d ctd_blk` (boxes only, ~0.09 s) | `python training/cli.py prune-ctd` (from `koharu-comictextdetector.onnx`) |
| `manga-text-yolo11n.onnx` | `YOLO_WEIGHTS` | `-d yolo` (boxes only, ~0.03 s) | `python training/cli.py detector ...`, see [training/README.md](../training/README.md) |
| `aot-inpainting.ckpt` | – | `training/cli.py inpaint` / `export-inpaint` | fetched by those commands from [ogkalu/aot-inpainting](https://huggingface.co/ogkalu/aot-inpainting), SHA-256 checked |
| `comictextdetector.pt.onnx` | – | optional input to `prune-ctd` | manga-image-translator GitHub release `beta-0.3` (the koharu file works the same) |

`ctd_blk` returns ctd's raw YOLO blocks (identical boxes to `ctd` with
`--opt detector.heads=blk`; `ctd` itself refines them with its text lines). Without line quads
or a glyph mask, downstream stages estimate those from pixels. Fine-tuned exports plug in with
`--opt ocr.weights=<onnx>` / `--opt inpainter.weights=<onnx>` and need no entry here.

## Hugging Face models (in the HF cache, `HF_HOME`, not in this folder)

| repo | size | used by |
|---|---|---|
| `facebook/nllb-200-distilled-600M` | 2.5 GB | `-t nllb` (default), `local_llm`'s en->target pivot, `chain`'s default 2nd step |
| `facebook/m2m100_418M` | 1.9 GB | `-t m2m100`, `local_llm`'s per-item fallback, `chain`'s default 1st step |
| `facebook/nllb-200-distilled-1.3B`, `facebook/m2m100_1.2B` | ~5 GB each | `-t nllb_big`, `-t m2m100_big` |
| `Qwen/Qwen2.5-3B-Instruct` | 6.2 GB | `-t local_llm` |
| `kha-white/manga-ocr-base` | 0.4 GB | `--ocr manga_ocr_hf` |

`fetch-weights` caches these for the methods it is given (or the defaults). A method that
cannot run here (torch not installed) is skipped. `--opt translator.model=<id or dir>` loads
any other checkpoint of the same family on first use.
