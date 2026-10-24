# Training

Every pipeline stage can be retrained or fine-tuned from this folder, and each result plugs back
into the worker through a `--opt` (no code change). One entry point dispatches to the scripts:

```bash
cd apps/worker
python training/cli.py                 # list commands (works without torch)
python training/cli.py <command> --help
```

| command            | script                  | produces                                   | used by the worker as                                         |
|--------------------|-------------------------|--------------------------------------------|---------------------------------------------------------------|
| `synth`            | `synth.py`              | detector / OCR / inpaint training data     | input to the commands below                                   |
| `download-pool`    | `download_pool.py`      | unlabeled manga pages                      | input to `pseudo-label`                                       |
| `pseudo-label`     | `pseudo_label.py`       | YOLO dataset labelled by a teacher (ctd)   | input to `detector`                                           |
| `prepare-manga109` | `prepare_manga109.py`   | YOLO dataset from Manga109 `<text>` boxes  | input to `detector`                                           |
| `detector`         | `train.py`              | `weights/manga-text-yolo11n.onnx`          | `-d yolo`                                                     |
| `prune-ctd`        | `prune_ctd.py`          | box-head-only comic-text-detector ONNX     | `-d ctd` fast path                                            |
| `ocr`              | `train_ocr.py`          | fine-tuned manga-ocr checkpoint            | input to `export-ocr`                                         |
| `export-ocr`       | `export_ocr_onnx.py`    | single-graph manga-ocr ONNX + vocab        | `--ocr manga_ocr --opt ocr.weights=<onnx>`                    |
| `translator`       | `train_translator.py`   | Hugging Face seq2seq checkpoint dir        | `-t nllb --opt translator.model=<dir>` (`-t m2m100` for M2M)  |
| `inpaint`          | `train_inpaint.py`      | AOT inpainter checkpoint (`best.pt`)       | input to `export-inpaint`                                     |
| `export-inpaint`   | `export_inpaint_onnx.py`| ONNX with the LaMa interface               | `-i lama_manga --opt inpainter.weights=<onnx>`                |

Install the extras once: `uv sync --extra torch --extra train` (torch + transformers for OCR,
translator and inpaint; ultralytics for the YOLO detector).

## Data

### Synthetic pages (`synth`)

```bash
python training/cli.py synth --out data/synth --n 2000 --fonts /path/to/jp-fonts \
    --backgrounds data/clean-pages --corpus lines.txt
```

One run writes every stage's data with exact labels (see the `synth.py` docstring):

```
data/synth/det/{images,labels}/{train,val}/*   + det/dataset.yaml   -> detector
data/synth/ocr/crops/*.png + ocr/{train,val}.jsonl                  -> ocr
data/synth/inpaint/{clean,mask,texted}/NNNNNN.png                   -> inpaint
```

Inpaint triplets share a stem: `clean` is the page without text, `texted` the same page with
text, `mask` a uint8 0/255 image (255 = text pixel, including outline strokes). Real data in the
same layout (a scanlator's cleaned page + the raw page + a text mask) can be passed alongside.
Use `--backgrounds` with text-free crops of real pages: the inpainter learns most from holes on
real artwork, and procedural paper alone under-represents it.

### Parallel corpus (`translator`)

JSONL with `{"src": "...", "tgt": "..."}` per line, or TSV `src<TAB>tgt`. Several files can be
given; `--val FILE` fixes the held-out set (otherwise `--val-ratio` of distinct sources is held
out, so repeated lines never leak). Japanese sources are normalised with the same
`textproc.normalize_ja` the worker applies before translating (`--no-normalize` to skip).

## Detector

```bash
python training/cli.py download-pool --out data/pool
DETECTOR=ctd python training/cli.py pseudo-label --pool data/pool --out data/pseudo-text
python training/cli.py detector --data data/pseudo-text/dataset.yaml --epochs 60
# or: --data data/synth/det/dataset.yaml, or a Manga109 dataset from prepare-manga109
```

## OCR

```bash
python training/cli.py ocr --data data/synth/ocr --out training/runs/ocr-ft --steps 3000
python training/cli.py export-ocr --model training/runs/ocr-ft/best --out weights/manga-ocr-ft.onnx
terjemahin-worker translate page.png --ocr manga_ocr --opt ocr.weights=weights/manga-ocr-ft.onnx
```

`--freeze-encoder` trains only the 2-layer decoder (about 4x faster per step on CPU).

## Translator (NLLB-200 / M2M100)

```bash
python training/cli.py translator --data data/manga_ja_id.jsonl --val data/page_refs.jsonl \
    --out training/runs/nllb-manga --epochs 3 --batch 8 --lr 1e-4 --decoder-layers 2 --eval-worker
terjemahin-worker translate page.png -t nllb --opt translator.model=training/runs/nllb-manga
```

- `--model` defaults to `facebook/nllb-200-distilled-600M`; any NLLB or M2M100 id/dir works (the
  family is detected from the tokenizer and recorded in `<out>/terjemahin.json`).
- What trains, from cheapest to most expressive: `--decoder-layers N` (last N decoder layers +
  final norm: 33.6M of 615M params for N=2, fits CPU in about 4 GB), `--freeze-encoder` (whole
  decoder), default (all but the 256k-token shared embedding, which is tied to the output layer),
  `--train-embeddings` (adds the shared embedding to any of the above; alone it is a full
  fine-tune). Frozen parameters get no gradients or optimizer state.
- Schedule: AdamW, `--scheduler linear|cosine|constant` with `--warmup` (steps, or a fraction
  of all steps if < 1), label smoothing 0.1, gradient clipping 1.0, `--grad-accum` for larger
  effective batches. Batches are length-bucketed to keep padding (and CPU time) down.
- Held-out data: `--val FILE` (its sources are removed from the training data, so a page's
  reference lines can never leak into training) or `--val-ratio`, split by distinct source.
- Eval: corpus chrF (`textproc.chrf`) and a smoothed word BLEU on the held-out set before training
  (the baseline) and every `--eval-every` steps (default once per epoch); the best chrF checkpoint
  is saved. `--eval-worker` also scores the base and tuned models through the worker's own
  translator (textproc pre/post-processing and retries), which is what users see.
- The output dir is a plain `save_pretrained` checkpoint (config, safetensors, tokenizer), so it
  also loads with `AutoModelForSeq2SeqLM` / `AutoTokenizer` anywhere else.
- Small corpora overfit fast: keep `--decoder-layers` small, watch held-out chrF, and do not
  trust a gain on fewer than a few hundred held-out lines.

## Inpainter (AOT)

```bash
python training/cli.py inpaint --data data/synth --steps 2000 --size 256 --batch 8 \
    --out training/runs/inpaint --export weights/aot-ft.onnx
# or export any checkpoint later (also 'aot' = the released weights unchanged):
python training/cli.py export-inpaint training/runs/inpaint/best.pt weights/aot-ft.onnx
terjemahin-worker translate page.png -i lama_manga --opt inpainter.weights=weights/aot-ft.onnx
```

- Network: manga-image-translator's AOTGenerator (gated, scaled-weight-standardised convs and 10
  AOT dilated blocks, 5.7M params), re-implemented with the same parameter names. `--init aot`
  (default) starts from the released `inpainting.ckpt`, fetched from the MIT-licensed Hugging Face
  mirror `ogkalu/aot-inpainting`, SHA-256 checked against upstream and cached as
  `weights/aot-inpainting.ckpt`. `--init none --ch 16 --blocks 4` trains a smaller net from scratch.
- Crops (`--size`, multiple of 4 and at least 68) are centred on text clusters, `--hard-ratio`
  of them on holes that are not on a flat colour, because the worker's `lama*` inpainters fill
  holes in plain balloons exactly without any model. Crops are randomly downscaled up to
  `--max-down` (as lama.py's tiling does), masks randomly dilated up to `--max-dilate` px, and
  flipped. `--clean-only DIR` adds text-free pages with random text-shaped masks.
- Loss: L1 in the hole (x6), L1 outside (x1), L1 on Sobel gradients in the hole (keeps line art
  and screentone edges sharp). No perceptual network or GAN, so it runs on CPU.
- Eval: hole-only PSNR and MAE of the composited output on fixed held-out crops, next to the cv2
  Telea baseline; `best.pt` keeps the best PSNR, `metrics.json` the history.
- Eval: if the held-out pages have no hole pixels, crops come from training pages instead; with no
  holes at all the scores are `null` and `best.pt` follows the latest eval (never a fake 128 dB).
- ONNX contract (identical to `koharu-lama-manga.onnx`): inputs `image [B,3,H,W]` RGB 0..1 and
  `mask [B,1,H,W]` (1 = hole), output `output [B,3,H,W]` RGB 0..1; H, W dynamic (padded to a
  multiple of 4 inside the graph, at least 68; the worker uses 512). The graph zeroes the hole
  and does the -1..1 mapping itself. Weight standardisation is folded into plain conv weights
  before export, and the export checks ONNX against torch at the trace size and an odd size.
- Using it: `-i lama_manga --opt inpainter.weights=<onnx>` makes `inpaint/lama.py` load that graph
  instead of koharu's LaMa, keeping its per-cluster tiling and flat-balloon shortcut; a missing
  file is an error, not a silent fallback. Use `lama_manga`, not `lama`: the `lama` method expects
  Carve's 0..255 output. On page01 with `--opt inpainter.flat_tol=0` (every hole through the
  model) the fine-tuned AOT inpaints in 7.1 s vs 35.0 s for the stock LaMa.
- Quality: AOT is a speed option, not a LaMa replacement. It copies periodic screentone and
  hatching worse than LaMa's Fourier convolutions, and where two balloons overlap (page01's small
  bottom-left balloon) it breaks the outline and leaves grey smears that LaMa does not.

## Smoke tests

`tests/test_training_translate_inpaint.py` runs every script end to end in seconds on CPU: a
tiny NLLB-shaped model is fine-tuned for 3 steps and loaded back through the worker's `nllb`
translator; a tiny AOT net is trained for 3 steps, exported, and run through `LamaInpainter`; the
released AOT weights are exported and compared with Telea on a hatched texture.
