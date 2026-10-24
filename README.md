# terjemahin

Translate manga pages, Japanese to Indonesian by default, and typeset the translation back into
the balloons. Every step runs offline on a CPU with open models; LLM APIs (Gemini, Claude,
OpenAI-compatible, DeepSeek, Ollama) are optional translators that add series context.

Each of the six stages is a registry of interchangeable methods, chosen per page from the CLI,
the web UI, the HTTP API or `.env`.

## Architecture

```
                     ┌──────────────────── apps/worker (Python) ──────────────────────┐
 CLI  ──────────────▶│                                                                │
 web UI / JSON API ─▶│  detect ─▶ OCR ─▶ translate ─▶ mask refine ─▶ inpaint ─▶ render │──▶ page.png
 (terjemahin-worker  │   ctd      manga  m2m100       default        lama_manga  default │    + page.json
  web, :8000)        │            _ocr                                                │      (bubbles)
                     └───────────▲────────────────────────────────────────────────────┘
 apps/api (Go, :8080)            │ JobMessage (input path, PageContext, PipelineConfig)
   POST /v1/pages ──▶ LavinMQ ───┘ terjemahin-worker serve ──▶ results queue ──▶ GET /v1/jobs/{id}
```

- `terjemahin_worker/registry.py`: one `Registry` per stage; a method is a factory
  `(settings, **opts)` registered with `@REGISTRY.register(name, requires=..., weights=...,
  needs_key=...)`. Heavy imports happen inside factories, so a missing optional dependency
  only disables the methods that need it.
- `pipeline.py`: resolves a `PipelineConfig` (method per stage + `options` per stage, unset
  stages fall back to `.env`), validates it before any stage runs, caches built stages, and
  re-runs only mask, inpaint and render (`compose`) when bubbles are edited by hand.
- `models.py`: `PageContext` (series title, glossary, previous pages, style notes),
  `PipelineConfig`, `PageResult` (bubbles with box, source, translation, font size).
- Model files: `weights.py` downloads ONNX/ckpt files into `apps/worker/weights/`; Hugging
  Face models go to the HF cache. See [weights/README.md](apps/worker/weights/README.md).

## Stages and methods

Generated from `terjemahin-worker methods --json` (bold = default). `needs`: torch = the
`torch` extra, weights = downloaded on first use, HF = Hugging Face model downloaded on first
use.

| stage | method | needs | description |
|---|---|---|---|
| detector | classical | – | OpenCV only, no weights: white balloons + glyph projection + line merge. Misses text outside balloons. |
| detector | **ctd** | weights | comic-text-detector, all heads: blocks + DBNet text lines + glyph mask (~0.6s CPU) |
| detector | ctd_blk | `prune-ctd` ONNX | comic-text-detector pruned to the box head (training/prune_ctd.py). Boxes only. |
| detector | yolo | trained ONNX | YOLO11n trained in this repo on synthetic pages + ctd pseudo-labels of page01 views (training/README.md): page01 is in its training data. Boxes only; misses small text on real B/W scans (Arisa_010 19/30), use ctd. |
| ocr | 48px | torch, weights | manga-image-translator 48px autoregressive OCR, per text line, beam search; predicts glyph/outline colours (torch, ~1-2s/page CPU) |
| ocr | **manga_ocr** | weights | manga-ocr ViT+BERT on onnxruntime (split encoder/decoder, batched greedy). Japanese, whole-bubble crops. |
| ocr | manga_ocr_hf | torch, HF | manga-ocr via transformers/torch (kha-white/manga-ocr-base); supports beam search. |
| ocr | none | – | No OCR; a vision-capable translator reads the crops itself. |
| translator | anthropic | `ANTHROPIC_API_KEY`, network | Claude (Messages API): text mode with page image for context, or full vision (reads the crops) when OCR is off; structured JSON output. |
| translator | chain | torch, HF | Pivot through several translators, e.g. steps='m2m100:en,nllb:id'. |
| translator | deepseek | `DEEPSEEK_API_KEY`, network | DeepSeek chat API (text only, JSON mode). |
| translator | gemini | `GEMINI_API_KEY`, network | Gemini (REST): reads + translates every crop with page and series context when OCR is off, or translates OCR lines with the page image attached. |
| translator | local_llm | torch, HF | Local chat LLM via transformers (offline; Qwen/Qwen2.5-3B-Instruct bf16, ~9.5GB RAM, 30-60 s/page on 4 CPU cores): whole page in one generation into English, NLLB to the target; glossary + previous pages, per-item fallback to m2m100. Opts: model, dtype, max_new_tokens, threads, pivot (en\|none), pivot_translator, fallback. |
| translator | **m2m100** | torch, HF | M2M100 418M (offline, fastest; the default): best offline seq2seq ja->id (page01 10/13, held-out lines 23/36 adequate). One bubble at a time, no page context. |
| translator | m2m100_big | torch, HF | M2M100 1.2B (offline, downloads ~5GB on first use). |
| translator | nllb | torch, HF | NLLB-200 600M distilled (offline, CPU, ~2.5GB RAM): weaker than m2m100 on ja->id (page01 8/13, held-out lines 18/36 adequate); strong en->id (local_llm's pivot). |
| translator | nllb_big | torch, HF | NLLB-200 1.3B distilled (offline, downloads ~5GB on first use, slower). |
| translator | none | – | Erase text only: inpaint without typesetting. |
| translator | ollama | Ollama server | Local LLM through Ollama's OpenAI-compatible endpoint (no key; vision=true for VLMs such as qwen2.5vl / gemma3; 900s timeout for CPU inference). |
| translator | openai | `OPENAI_API_KEY`, network | Any OpenAI-compatible chat API (OpenAI, OpenRouter, Groq, LM Studio, vLLM...): numbered-line JSON protocol, vision with page + crops when OCR is off. |
| translator | original | – | Keep the OCR text (debug OCR/typesetting). |
| mask_refiner | bbox | – | Erase the whole padded text box. |
| mask_refiner | **default** | – | Glyph pixels only: detector mask components assigned to text lines (or local binarisation), grown into anti-aliased edges, dilated by font size, balloon outlines kept. |
| mask_refiner | dilate | – | Detector mask (or binarised box) inside the text zone, dilated. |
| inpainter | fill | – | Flat fill with the surrounding median colour (fast). |
| inpainter | lama | weights | big-lama Places2 (Carve ONNX 512), same tiling as lama_manga; better on photographic backgrounds, worse on line art and screentone. |
| inpainter | **lama_manga** | weights | LaMa fine-tuned on manga (ONNX 512), per text cluster at native resolution; flat-background holes filled exactly without the model (~3s per 512 tile CPU). |
| inpainter | ns | – | OpenCV Navier-Stokes inpaint (no weights, fast). |
| inpainter | telea | – | OpenCV Telea fast-marching inpaint (no weights, fast). |
| renderer | basic | – | Horizontal centred text in the detector box (v1). |
| renderer | **default** | – | Balloon-aware typesetting: finds each balloon's interior, fits the text to its shape (font-size search, Indonesian hyphenation, no orphans), page-consistent sizes, outline over art; vertical columns for CJK targets. |
| renderer | vertical | – | Vertical right-to-left columns for CJK targets (ja/zh/ko), balloon-aware, vertical punctuation forms, kinsoku; needs a CJK font (font_path or fonts/*CJK*). |

`terjemahin-worker methods` prints the same list with whether each method can run on this
machine (and why not); `--json` adds every method's named options and their defaults.

## Quick start (worker)

```sh
cd apps/worker
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -e ".[torch,web,dev]" \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
cp ../../.env.example .env          # optional: defaults, API keys

.venv/bin/terjemahin-worker fetch-weights      # optional: ~1 GB of ONNX/ckpt + M2M100 (~1.9 GB)
.venv/bin/terjemahin-worker translate page.png -o out/page.png --debug-dir out/debug
#   -> out/page.png, out/page.json (bubbles), out/debug/{detect,mask,inpainted,final}.png
.venv/bin/python -m pytest -q
```

Without `fetch-weights`, files download on first use. The CPU index installs the CPU torch
wheel (~0.75 GB) instead of the CUDA build (several GB); drop those two flags for a GPU
(`DEVICE=cuda`). Without the `torch` extra the default translator `m2m100` is unavailable: use
an API translator, or `-t none` to only erase text.

## CLI by stage

Every stage has a method flag and takes options as `--opt stage.key=value` (values are JSON;
`true/false/yes/no/on/off/null` are understood; stage aliases `detect`, `mask`, `inpaint`,
`render`, `translate`). Unknown methods and bad option values fail before any stage runs; an
unknown option key is ignored with a warning that names it (check the log for typos).

```sh
W=.venv/bin/terjemahin-worker
$W detect page.png -d ctd -o out/detect.png      # boxes + reading order; JSON on stdout,
                                                 # out/detect-mask.png = ctd's glyph mask
$W translate page.png -d yolo --opt detector.conf=0.3
$W translate page.png --ocr 48px                 # any script; manga_ocr is Japanese only
$W translate page.png -t nllb                    # or local_llm, chain, gemini, ...
$W translate page.png -t chain --opt translator.steps=m2m100:en,nllb:id
$W translate page.png --mask bbox -i lama --opt inpainter.flat_tol=0
$W translate page.png -t none -i telea           # erase only, no typesetting
$W translate page.png -t original -r basic       # typeset the OCR text (debugging)
$W translate page.png --opt renderer.font_path=/path/Font.ttf --opt renderer.uppercase=true
$W translate en.png --from en --to id            # non-Japanese source: OCR switches to 48px
```

Series context and hand edits:

```sh
cat > ctx.json <<'EOF'
{"series_title": "My Series",
 "glossary": [{"source": "メシア", "target": "Mesia"}],
 "previous_pages": [["Hei, Mei!"]], "style_notes": "casual Jakarta slang"}
EOF
$W translate page.png -o out/page.png --context ctx.json
# edit target_text (or boxes, font_size) in out/page.json, then re-typeset without
# running detect/OCR/translate again:
$W compose page.png out/page.json -o out/edited.png --opt renderer.uppercase=true
```

The glossary is used by the seq2seq translators (nllb, m2m100, chain: terms are substituted
around the model) and by the LLM translators; `series_title`, `previous_pages` and
`style_notes` only reach the LLM translators (including `local_llm`).

## Choosing methods

Measured on page01 with the evaluation harness (one stage changed at a time, the rest at the
defaults; full table under [Performance](#performance)):

- **Detector**: `ctd` (default) is the only one with text lines and a glyph mask, which the
  default mask refiner uses to erase glyphs only. Every detector's boxes are refined from the
  pixels (`DETECT_REFINE`, detect/postprocess.py): a box over two balloons is split, a box that
  stops short of its text grows over the cut-off glyphs, two-page spreads run as two halves and are
  read right page first, 4-koma strips one at a time. On five GT pages (page01, two Manga109
  spreads, two PSImera pages) `ctd` finds 64/64 bubbles with 1 false positive; `ctd_blk` 60/64 and
  `yolo` 50/64 (yolo was trained on page01 views, so its page01 score proves nothing). `classical`
  needs no weights but misses text outside white balloons.
- **OCR**: `manga_ocr` (default, ONNX, 1.2 s) for Japanese. `48px` reads any script (used
  automatically for non-Japanese sources) but read one non-text region on page01.
  `manga_ocr_hf` gives the same reads through torch (GPU, beam search).
- **Translator**: the weakest stage. Offline, the default `m2m100` keeps the meaning of 10/13
  page01 balloons (chrF 32.4) and 23/36 held-out lines (`eval/fixtures/lines_ja_id.tsv`); `nllb`
  8/13 and 18/36; `chain` (m2m100 -> en -> nllb) 9/13 and 24/36; `local_llm` 10/13 (chrF 38.5)
  and 24/36 at ~47 s and ~9.5 GB RAM per page. The seq2seq translators work bubble by bubble
  without page context; only the LLM translators see the page. With an API key, `gemini`,
  `anthropic`, `openai` and `deepseek` also use the series context (not measured here: no
  keys). `none` erases without typesetting, `original` checks OCR and typesetting.
- **Mask refiner**: keep `default`. `bbox` erases whole boxes (changes art around text);
  `dilate` is a looser glyph mask.
- **Inpainter**: `lama_manga` (default, 3.7 s) for screentone and art behind text. On plain
  white balloons `telea`, `ns` and `fill` (0.3 s) score the same on page01 and are 10x
  faster. `lama` (Places2) is slower and meant for photographic backgrounds.
- **Renderer**: `default` for Latin targets; `vertical` only for ja/zh/ko targets (with a CJK
  font); `basic` is the old fixed-box renderer (fonts down to 6 px on page01).

## Web UI

```sh
.venv/bin/terjemahin-worker web               # http://127.0.0.1:8000 (--host 0.0.0.0 to share)
```

Drop a page, pick a method and an options JSON per stage, set languages, series title,
glossary and style notes, then translate. The viewer shows the original next to the detection,
erase mask, inpainted or translated page. Every bubble's text can be edited or switched off;
"Re-typeset" re-runs only mask, inpaint and render (only render when just the text changed).
Jobs live in `$STORAGE_DIR/web/<id>/`, the newest `WEB_KEEP_JOBS` (100) are kept.

## API

### Worker JSON API (same server as the web UI)

```sh
curl -F file=@page.png \
     -F 'pipeline={"translator":"m2m100","options":{"renderer":{"uppercase":true}}}' \
     -F 'context={"glossary":[{"source":"メシア","target":"Mesia"}]}' \
     localhost:8000/api/translate                 # waits; -F wait=false returns at once
curl localhost:8000/api/jobs/<id>                 # status, stage, result (bubbles), image URLs
curl localhost:8000/api/jobs/<id>/download -o page.png
curl -H 'content-type: application/json' localhost:8000/api/compose \
     -d '{"job_id":"<id>","bubbles":[...edited result.bubbles...],
          "pipeline":{"options":{"renderer":{"uppercase":true}}}}'
curl localhost:8000/api/methods                   # stages, methods, availability, defaults
```

`pipeline` is a `PipelineConfig`: `detector`, `ocr`, `translator`, `mask_refiner`,
`inpainter`, `renderer` (method names; omitted = `.env` default), `source_lang`,
`target_lang`, `min_ocr_confidence` and `options` keyed by stage. An unknown or unavailable
method or a bad language is a 400; option values are checked when the job starts (before any
stage runs), so a bad one comes back as a failed job whose error names the option.

### Go API + queue (production path)

```sh
docker compose up lavinmq -d
(cd apps/api && go run .)                         # :8080
(cd apps/worker && .venv/bin/terjemahin-worker serve)
curl -F file=@page.jpg -F 'pipeline={"translator":"nllb","inpainter":"lama_manga"}' \
     -F 'context={"glossary":[{"source":"ナルト","target":"Naruto"}]}' -F debug=true \
     localhost:8080/v1/pages                      # 202 + job
curl localhost:8080/v1/jobs/<id>                  # queued | done | failed
curl localhost:8080/v1/jobs/<id>/image -o out.png
curl localhost:8080/v1/jobs/<id>/result           # PageResult JSON
curl localhost:8080/v1/jobs/<id>/debug/mask -o mask.png   # detect|mask|inpainted|final, debug=true jobs
```

The API validates `pipeline` structurally (known fields and stages) and the worker validates
method names and options. The job store is in-memory: jobs are lost when the API restarts
(Postgres is in compose but not wired yet).

## Docker

```sh
docker compose up --build                         # lavinmq, postgres, api, worker (queue)
docker compose --profile web up --build web       # standalone web UI on :8000
docker build --build-arg FETCH="--all" -t terjemahin-worker apps/worker
```

The worker image (`apps/worker/Dockerfile`) installs CPU torch and the `torch,web` extras,
copies `fonts/`, and runs `terjemahin-worker fetch-weights` at build time so the default
methods start without downloads. Build args: `EXTRAS` (e.g. `web` for a 0.8 GB image without
torch; 2.2 GB with it, before weights), `FETCH` (arguments of `fetch-weights`, e.g.
`"ctd manga_ocr lama_manga 48px"` or `--all`), `CJK_FONTS=1` (Noto CJK for vertical
Japanese/Chinese/Korean targets). Locally trained files (`-d yolo`, `-d ctd_blk`) are not in
the image: mount them and point `YOLO_WEIGHTS` / `CTD_WEIGHTS` at them.

## Configuration

All settings are environment variables (or `.env` in the working directory); see
[.env.example](.env.example) for every one with its default: the default method per stage,
`OCR_NON_JAPANESE`, limits (`MAX_PIXELS`, `MIN_OCR_CONFIDENCE`, `STAGE_CACHE_SIZE`,
`WEB_KEEP_JOBS`), `DEVICE`, weight and font paths, and the API keys and models of the LLM
translators. `TERJEMAHIN_OFFLINE=1` forbids weight downloads, `HF_HUB_OFFLINE=1` hub access.

## Training

Every stage can be fine-tuned and plugged back in with an `--opt`, no code change:
synthetic pages (`synth`), pseudo-labels from a teacher detector, YOLO11n detector, CTD box-head
pruning, manga-ocr fine-tune + ONNX export, NLLB/M2M100 fine-tune on a parallel corpus, AOT
inpainter fine-tune + ONNX export. All commands go through `python training/cli.py`; see
[apps/worker/training/README.md](apps/worker/training/README.md).

## Evaluation harness

```sh
cd apps/worker
.venv/bin/python -m terjemahin_worker.eval bench --page eval/fixtures/page01.png \
    --gt eval/fixtures/page01.gt.json --out out/bench            # baseline only (~45 s)
.venv/bin/python -m terjemahin_worker.eval bench ... --matrix    # vary every stage
.venv/bin/python -m terjemahin_worker.eval bench ... --matrix --stages inpainter -t m2m100
```

`page01.gt.json` holds the 13 balloons' Japanese text and a reference Indonesian translation;
`eval/fixtures/real/` adds two held-out real B/W pages with GT (PSImera, CC BY-NC) and
`eval/heldout/` the GT of two Manga109 spreads (images not redistributable). The harness scores
detection (GT balloons matched by OCR text; a box that reads under 75% of its bubble is
*partial* and counts as not found; false positives, duplicates), OCR CER, translation (chrF
against the reference, and "meaning": embedding cosine >= 0.8 with
paraphrase-multilingual-MiniLM **and** no flipped negation, swapped participants or
active-for-passive, `metrics.meaning_conflicts`), inpainting (text a judge detector+OCR still reads
in erased areas, residue, pixels changed outside the mask), rendering (ink inside the balloon,
the rendered text read back by an independent OCR, smallest font) and latency. It writes
`report.md`, `report.json` and a `contact_sheet.jpg`; each stage gets PASS/FAIL against fixed
thresholds. Methods that download GBs or need a network/server are skipped unless
`--allow-downloads` / `--network` is given.

```sh
.venv/bin/python -m terjemahin_worker.eval lines -t m2m100 -t nllb --out out/lines   # held-out lines
```

## Performance

Fresh harness run on 2026-10-24 (final fix round): page01 (1487x2048, 13 balloons), 4 shared CPU
cores, no GPU, `python -m terjemahin_worker.eval bench ... --matrix` (4 min 18 s for the whole
matrix). Other pages: docs/PROOF.md section 0.6.

Default pipeline (`ctd`, `manga_ocr`, `m2m100`, `default`, `lama_manga`, `default`):

| check | result | threshold |
|---|---|---|
| detection | 13/13 balloons, 0 partial, 0 false positives, 0 duplicates | PASS (>= 92%) |
| OCR | CER 0.008, 11/13 exact | PASS (<= 0.15) |
| translation | chrF 32.4; meaning kept in 10/13 balloons (mean cosine 0.86) | PASS (>= 70%) |
| inpainting | 0 residual text regions, ghost 0.0001, no change > 8 px outside the mask | PASS |
| rendering | read-back CER 0.005, but the 16 px aside (あとね) spills over its tiny balloon (worst inside share 0.94) | **FAIL** (>= 0.99) |
| latency, warm | detect 0.6, OCR 1.3, translate 2.9, mask 0.3, inpaint 3.8, render 0.6: **9.5 s** | |
| latency, first page | 22.1 s (models load lazily) | |

One stage changed at a time (meaning = adequate GT balloons; "erase" = the inpaint check):

| run | detection | OCR CER | chrF | meaning | erase | render | total s |
|---|---|---|---|---|---|---|---|
| default | 13/13 | 0.008 | 32.4 | 10/13 | PASS | FAIL (aside) | 21.3 |
| `-d classical` | 13/13 | 0.035 | 33.1 | 10/13 | PASS | PASS | 5.7 |
| `-d ctd_blk` | 13/13 | 0.005 | 32.4 | 10/13 | PASS | PASS | 9.5 |
| `-d yolo` | 13/13 | 0.005 | 32.4 | 10/13 | 1 residual, FAIL | PASS | 7.8 |
| `--ocr 48px` | 13/13, 1 fp | 0.005 | 32.4 | 10/13 | PASS | PASS | 12.4 |
| `--ocr manga_ocr_hf` | 13/13 | 0.008 | 32.4 | 10/13 | PASS | FAIL (aside) | 10.5 |
| `-t nllb` | 13/13 | 0.008 | 30.7 | 8/13 FAIL | PASS | PASS | 11.6 |
| `-t chain` | 13/13 | 0.008 | 31.6 | 9/13 FAIL | PASS | FAIL | 16.0 |
| `-t local_llm` | 13/13 | 0.008 | **38.5** | 10/13 | PASS | PASS | 54.9 |
| `--mask bbox` | 13/13 | 0.008 | 32.4 | 10/13 | PASS | PASS | 11.3 |
| `--mask dilate` | 13/13 | 0.008 | 32.4 | 10/13 | PASS | PASS | 5.7 |
| `-i telea` / `ns` / `fill` | 13/13 | 0.008 | 32.4 | 10/13 | PASS | PASS | 6.4 / 6.3 / 6.3 |
| `-i lama` | 13/13 | 0.008 | 32.4 | 10/13 | PASS | FAIL | 18.5 |
| `-r basic` | 13/13 | 0.008 | 32.4 | 10/13 | PASS | 6 px fonts, FAIL | 9.8 |

"total s" counts stages reused from the cache at the default's warm time, so first-load
costs only show in the default row. Not run: API translators (no keys), `ollama` (no server),
`nllb_big` / `m2m100_big` (multi-GB downloads), `--ocr none` (needs a vision translator);
`-r vertical` is for CJK targets and loses Indonesian text by design. Each run's images are in
`out/<dir>/runs/` and `contact_sheet.jpg`.

## Limitations

- Translation quality is the weak stage. The default M2M100-418M passes page01 (10/13) but
  fails the meaning check on both held-out real pages and keeps the meaning of 23 of 36
  held-out lines; it guesses subjects (no page context), and mistranslates casual lines and
  asides (あとね -> "Selanjutnya ya"). `local_llm` is no better on the held-out lines (24/36) at
  ~47 s and ~9.5 GB RAM per page. The API translators were not measured (no keys available).
- Real-page evidence is five GT pages (page01, two Manga109 spreads, two 640 px PSImera pages);
  other art styles and scan qualities will differ. Furigana in open balloons and ruby over art
  can still leave faint residue (Arisa_010 ghost share 0.029 > 0.01).
- Text that does not fit its balloon at a legible size (0.9 x the page minimum) is set at that
  size spilling over the balloon with an outline; the harness counts this as a render failure
  (page01's small aside does it).
- `manga_ocr` reads Japanese only; other sources need `--ocr 48px` (chosen automatically when
  `--from` is not Japanese).
- SFX outside balloons are detected but translated like dialogue; there is no SFX-specific
  lettering.
- CJK targets need a CJK font (not bundled, `CJK_FONTS=1` in Docker).
- CPU only was tested. `DEVICE=cuda` applies to the torch methods; the ONNX methods use the
  CPU execution provider.
- The Go API keeps jobs in memory. The queue path ran end to end once (Go API, RabbitMQ standing
  in for LavinMQ, `terjemahin-worker serve`; docs/PROOF.md section 6) on an earlier revision;
  Docker compose and LavinMQ themselves were not exercised.

## Layout

```
apps/api/                 Go HTTP API: upload, queue, job status, results
apps/worker/terjemahin_worker/
  detect/ ocr/ translate/ mask/ inpaint/ render/   one module per method
  pipeline.py registry.py models.py config.py weights.py cli.py consumer.py
  web/                    FastAPI web UI + JSON API
  eval/                   evaluation harness
apps/worker/training/     training and export scripts (training/README.md)
apps/worker/eval/         fixtures/ (page01, real/ PSImera pages, lines_ja_id.tsv) + heldout/ GT
apps/worker/weights/      model files (weights/README.md)
apps/worker/fonts/        lettering fonts (fonts/README.md)
```
