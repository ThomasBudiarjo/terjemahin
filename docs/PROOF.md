# terjemahin: evidence that the product works

Everything on this page was produced fresh on 2026-10-24 (13:30 to 14:00 UTC) by running the product.
Nothing was copied from older outputs. Each claim names the command that produced it and the
artifact it left in [`docs/proof/`](proof/).

- **Machine:** 4 shared CPU cores (Intel Xeon), 15 GB RAM shared with one other agent, no GPU.
- **Python:** 3.12 venv, CPU torch.
- **Tree:** `18b962b` ("WIP: robustness sweep fixes in progress") plus the uncommitted working-tree
  changes present at the time (13 files: `translate/textproc.py`, `translate/hf_seq2seq.py`,
  `render/default.py`, `cli.py`, ...). The numbers describe that working tree. For example, NLLB's
  chrF on page01 is 30.7 here, against 28.4 in the earlier baseline.
- **Where to run:** every command runs from `apps/worker` with `W=.venv/bin/terjemahin-worker` and
  `B=".venv/bin/python -m terjemahin_worker.eval bench"`.
- **Scratch output:** full-resolution scratch output went to `apps/worker/out/proof/` (not for commit).
  `docs/proof/` holds downscaled JPEGs (each <= 400 KB, 5 MB in total), reports, logs, the GT files and
  the helper scripts (`docs/proof/scripts/`, run from `apps/worker/out/proof`).

## Verdict at a glance

| stage | works? | strongest evidence | weak spots found |
|---|---|---|---|
| detect | yes | page01 13/13 GT bubbles, 0 false positives (ctd, ctd_blk, yolo, classical); tall strip 25/25; synthetic pages 19/20 | two free-floating text columns over art were merged into one duplicate region (synth 000003) |
| OCR | yes | page01 mean CER 0.008 (manga_ocr), 0.005 (48px, yolo); synthetic pages 0.000 to 0.053; Latin text via 48px 0.080 | 48px adds one false positive on page01; one Latin bubble read as `啊珈`, conf 0.24, left untouched |
| translate | **partly** | runs with every offline method. On page01 the default `nllb` **fails** the harness's meaning check (8/13 adequate, need >= 70%). `m2m100` (10/13) and `local_llm` (10/13, chrF 38.5) pass | the default translator is the weakest stage. API translators only tested against mocks (no keys) |
| mask + inpaint | yes on clean scans | page01 0 residual text regions, 0 dark/ghost residue, all changes within 8 px of the mask | **fails on a half-resolution JPEG q40 page** (5 residual regions, 38/141 chars remain); ghost screentone 0.019 > 0.01 on synth 000002 |
| render | yes | page01 read back from pixels by an independent OCR: CER 0.012 mean, 0 lost, all ink inside balloons | **2 bubbles lost / overlapping** where the detector produced a duplicate (synth 000003) |
| selectable methods | yes | 32 methods (28 runnable here); the `--matrix` ran 18 non-baseline variants with 0 errors (8 skipped with reasons); methods picked from CLI flags, the web UI dropdowns and the queue API's `pipeline` JSON | 4 API translators need keys; `ollama` needs a server; `*_big` need 5 GB downloads |
| training | yes (smoke) | fresh detector / OCR / translator / inpainter runs, each exported and loaded back into the product via `--opt`; the shipped `manga-text-yolo11n.onnx` scores 13/13 on page01 | the smoke runs are too short to improve quality (shown below). They prove the loop works, not that it helps |

## 1. Test suite

`.venv/bin/python -m pytest -q -rA --durations=15 -p no:cacheprovider` gave **487 passed, 1 skipped
in 156.8 s** (wall clock 2 min 39 s).

- The per-file counts and the 15 slowest tests are in [proof/pytest.txt](proof/pytest.txt).
- The skipped test is the real Qwen2.5-3B local-LLM test (`TERJEMAHIN_SLOW_LLM=1`). The same
  translator did run for real in the matrix (section 4).
- The suite was run again at the end of this session against the then-current tree: see the
  final-run line at the bottom of `pytest.txt`.

## 2. Methods

`$W methods` lists 6 stages and 32 methods, 28 of them `ok` on this machine. The 4 unavailable
methods are API translators without a key (`anthropic`, `deepseek`, `gemini`, `openai`). The full
table is in [proof/methods.txt](proof/methods.txt).

| stage | methods (* default) |
|---|---|
| detector | classical, **ctd***, ctd_blk, yolo |
| ocr | 48px, **manga_ocr***, manga_ocr_hf, none |
| translator | anthropic, chain, deepseek, gemini, local_llm, m2m100, m2m100_big, **nllb***, nllb_big, none, ollama, openai, original |
| mask_refiner | bbox, **default***, dilate |
| inpainter | fill, lama, **lama_manga***, ns, telea |
| renderer | basic, **default***, vertical |

## 3. End to end on page01 with defaults

`$W translate eval/fixtures/page01.png -o out/proof/e2e/page01.png --debug-dir out/proof/e2e/debug`
exited 0 in 26.7 s wall time for a cold process. The log is [proof/logs/page01_e2e_cli.log](proof/logs/page01_e2e_cli.log):
`page done: 14 regions, 14 bubbles | detect 0.68s, ocr 1.31s, translate 8.27s, mask 0.08s, inpaint 11.09s, render 0.71s`.

![page01 before / after](proof/page01_before_after.jpg)

| detect (boxes, text lines, reading order) | erase mask | inpainted | final |
|---|---|---|---|
| ![](proof/page01_1_detect.jpg) | ![](proof/page01_2_mask.jpg) | ![](proof/page01_3_inpainted.jpg) | ![](proof/page01_4_final.jpg) |

**Per bubble:** OCR against the hand-made GT, and translation against the reference. Matching uses
the harness's `metrics.match_detections`. The script is inline in this session; the table is
[proof/page01_bubbles.md](proof/page01_bubbles.md).

| GT # | GT Japanese | OCR (manga_ocr) | CER | translation (nllb) | reference | chrF |
|---|---|---|---|---|---|---|
| 0 | ねえねえ、めいは坂本のことどう思う？ | ねえねえ、めいは坂本のことどう思う？ | 0.00 | Hei, hei, apa yang kau pikirkan tentang Sakamoto? | Hei, hei, Mei, menurutmu Sakamoto itu gimana? | 35.6 |
| 1 | 坂本くん？ | 坂本くん？ | 0.00 | Sakamoto-kun? | Sakamoto-kun? | 100.0 |
| 2 | うーんそうだなぁ・・優しいよね | うーんそうだなあ．．優しいよね | 0.07 | Hmm, ya, baik-baik saja | Hmm, gimana ya... dia baik, kan? | 19.5 |
| 3 | えー、それだけー？ | えー、それだけー？ | 0.00 | Eh, itu saja? | Eh, cuma itu? | 22.5 |
| 4 | じゃあメシアはどうなの？ | じゃあメシアはどうなの？ | 0.00 | Kalau gitu, bagaimana dengan Mesias? | Kalau gitu, Mesia sendiri gimana? | 52.2 |
| 5 | え？ | え？ | 0.00 | Eh? | Eh? | 100.0 |
| 6 | わ、私はー・・・別にそんな・・・ | わ、私はー．．．別にそんな．． | 0.00 | A-aku tidak ingin... | A-aku sih... nggak gimana-gimana... | 18.8 |
| 7 | 反応がいちいち面白いなーとか | 反応がいちいち面白いなーとか | 0.00 | Reaksi itu sangat menarik, gitu | Reaksinya selalu lucu, gitu deh | 27.3 |
| 8 | うんうん | うんうん | 0.00 | Iya, iya | Iya, iya | 100.0 |
| 9 | 店の店員さんとかにも敬語使ってて偉いなーとか | 店の店員さんとかにも敬語使ってて偉いなーとか | 0.00 | Dia juga menggunakan kata-kata pujian untuk pemilik toko, gitu | Dia sopan pakai bahasa halus bahkan ke pelayan toko, keren gitu | 22.0 |
| 10 | わあ、いいね！ | わあ、いいね！ | 0.00 | Wah, baiklah! | Wah, bagus dong! | 31.1 |
| 11 | あっ、あとこないだね、髪型変えたらすぐに気付いてくれてっ・・ | あっあとこないだね、髪型変えたらすぐに気付いてくれてっ．．． | 0.03 | Ah, saat ini, aku baru tahu kalau rambutku berubah... | Ah, terus kemarin itu, waktu aku ganti gaya rambut, dia langsung sadar... | 21.6 |
| 12 | ふふっ、他には？ | ふふっ、他には？ | 0.00 | Fufu, ada yang lain? | Fufu, terus apa lagi? | 22.8 |
| (sfx) | あとね (handwritten, GT lists it as SFX) | あとね | – | Aku akan pergi | – | – |

- **Detection and OCR:** 13/13 GT bubbles found, mean CER 0.008, page chrF 30.7.
- **Translation errors:**
  - #9 changes the meaning: "praise words for the shop owner" instead of "polite to the shop clerk".
  - #11 loses "he noticed right away".
  - The handwritten あとね ("and also...") becomes "Aku akan pergi" ("I'm going to go") in a tiny font.
- The page renders cleanly, and the web UI lets a user fix all of these by hand (section 6).

## 4. Eval harness: baseline and full matrix

**Baseline:** `$B --page eval/fixtures/page01.png --gt eval/fixtures/page01.gt.json --out out/proof/bench_base`
(43 s). Report: [proof/reports/bench_baseline_report.md](proof/reports/bench_baseline_report.md).

- **detect** PASS: 13/13, 0 false positives, 0 duplicates.
- **ocr** PASS: mean CER 0.008, 11/13 exact.
- **translate FAIL:** chrF 30.7 passes (need >= 25), but the meaning check fails: 8/13 adequate,
  need >= 70%. Inadequate: GT #2, #6, #7, #9, #11.
- **inpaint** PASS: 0 residual regions, dark/ghost residue 0.0000, outside-mask MAD 0.017, every change
  within 8 px of the mask.
- **render** PASS: read back from pixels 283/283 letters, CER mean 0.012, worst 0.167, 0 lost, font 18 to 31 px.

**Matrix:** `$B ... --out out/proof/bench_matrix --matrix` (286 s; peak RSS 13.8 GB, logged by a
watchdog in `out/proof/bench_matrix_mem.log`). It varies one stage at a time over every method
available here. Full report:
[proof/reports/bench_matrix_report.md](proof/reports/bench_matrix_report.md).

| run | detection | OCR CER | chrF | meaning | inpaint / outside | render (inside, read-back CER) | stage s (first call) |
|---|---|---|---|---|---|---|---|
| **baseline** | 13/13 PASS | 0.008 | 30.7 | 8/13 FAIL | 0 residual, PASS | 1.000, 0.012 | – (total 23.1) |
| detector-classical | 13/13 PASS | 0.035 | 30.3 | 9/13 FAIL | 1 residual, FAIL | 1.000, 0.000 | 0.06 |
| detector-ctd_blk | 13/13 PASS | 0.005 | 30.7 | 8/13 FAIL | 0, PASS | 1.000, 0.018 | 0.09 |
| detector-yolo | 13/13 PASS | 0.005 | 30.7 | 8/13 FAIL | 0, PASS | 1.000, 0.000 | 0.03 |
| ocr-48px | 13/13, **fp 1** FAIL | 0.005 | 30.7 | 8/13 FAIL | 0, PASS | 1.000, 0.000 | 2.72 |
| ocr-manga_ocr_hf | 13/13 PASS | 0.008 | 30.7 | 8/13 FAIL | 0, PASS | 1.000, 0.012 | 1.16 |
| translator-chain | 13/13 | 0.008 | 31.6 | 9/13 FAIL | 0, PASS | 1.000, 0.013 | 7.50 |
| **translator-local_llm** | 13/13 | 0.008 | **38.5** | **10/13 PASS** | 0, PASS | 1.000, 0.004 | 47.38 |
| **translator-m2m100** | 13/13 | 0.008 | 32.4 | **10/13 PASS** | 0, PASS | 0.996, 0.016 | 4.85 |
| translator-none (erase only) | 13/13 | 0.008 | 0.0 | 0/13 (expected) | 0, PASS | – | 0.00 |
| translator-original (debug) | 13/13 | 0.008 | 0.0 | 0/13 (expected) | 0, PASS | – | 0.00 |
| mask_refiner-bbox | 13/13 | 0.008 | 30.7 | 8/13 | outside FAIL (0.0107 changed) | 1.000, 0.007 | 0.00 |
| mask_refiner-dilate | 13/13 | 0.008 | 30.7 | 8/13 | 0, PASS | 1.000, 0.006 | 0.01 |
| inpainter-fill | 13/13 | 0.008 | 30.7 | 8/13 | 0, PASS | 1.000, 0.006 | 0.39 |
| inpainter-lama | 13/13 | 0.008 | 30.7 | 8/13 | 0, PASS | 0.997, 0.012 | 12.04 |
| inpainter-ns | 13/13 | 0.008 | 30.7 | 8/13 | 0, PASS | 1.000, 0.006 | 0.28 |
| inpainter-telea | 13/13 | 0.008 | 30.7 | 8/13 | 0, PASS | 1.000, 0.006 | 0.30 |
| renderer-basic | 13/13 | 0.008 | 30.7 | 8/13 | 0, PASS | 0.956, 0.021, **min font 6 FAIL** | 0.03 |
| renderer-vertical | 13/13 | 0.008 | 30.7 | 8/13 | 0, PASS | **lost 14 FAIL**: vertical is for CJK targets; Latin id target unreadable, as designed | 0.56 |

**Skipped, with the reason recorded by the harness:**

- `ocr-none`: needs a vision translator.
- `anthropic`, `deepseek`, `gemini`, `openai`: no API key.
- `ollama`: no server.
- `m2m100_big`, `nllb_big`: 5 GB downloads; there is not enough disk.

**Errors:** none. Every runnable method produced a page.

![matrix contact sheet](proof/bench_matrix_contact_sheet.jpg)

## 5. Pages beyond page01

### Synthetic pages with exact GT

**Generator:** `training/synth.py`, run as
`.venv/bin/python training/synth.py --out out/proof/synth/gen --n 8 --seed 7 --corpus corpus_ja.txt --backgrounds out/proof/synth/bg --fonts <mit fonts> /usr/share/fonts/opentype/ipafont-gothic --val-ratio 0 --val-corpus-ratio 0 --emit det,inpaint`.

- **Corpus:** 20 Japanese lines, each with a hand-written Indonesian reference
  ([proof/gt/synth_corpus.tsv](proof/gt/synth_corpus.tsv)).
- **Background:** the page01 page inpainted by section 3, or procedural paper with screentone.
- **Same generator, horizontal pages:** `--vertical 0 --seed 11 --n 3` gives horizontal-only pages.
- **GT:** [proof/scripts/mkgt.py](proof/scripts/mkgt.py) turns `pages.jsonl` (the exact rendered text
  of every block) into harness GT ([proof/gt/](proof/gt/)). The harness reads the degraded JPEG
  (blur, noise, JPEG) the generator writes for detector training.

**Hostile inputs**, built by an inline script in this session (GT in `proof/gt/`):

- **Tall strip:** page01 + synth 000002 + synth 000007 stacked to 1487x6307 (webtoon-like).
- **Low resolution:** page01 at half resolution, JPEG q40.
- **Horizontal Latin text:** the Indonesian page01 output fed back in as a source page.

**Commands:** [proof/scripts/run_extra.sh](proof/scripts/run_extra.sh) runs
`$B --page <page> --gt <gt> --out out/proof/extra/<name>` for each page. Reports are in [proof/reports/](proof/reports/).

| page | what is hard | detect | OCR CER | chrF | meaning | inpaint | render | total s |
|---|---|---|---|---|---|---|---|---|
| synth 000002 (912x1292, 7 blocks) | screentone, cloud and rect balloons | 7/7 PASS | 0.053 | 45.7 | 4/7 FAIL | **FAIL**: ghost residue 0.019 > 0.01 | PASS, read-back 0.008 | 29.3 |
| synth 000003 (4 blocks) | white-stroked free text over art, no balloon | **3/4, 1 duplicate FAIL** | 0.000 | 44.3 | 2/4 FAIL | PASS | **FAIL: 2 bubbles lost**, read-back 0.48 | 22.0 |
| synth 000007 (826x1129, 5 blocks) | real-art background, spiky balloon, 17 px text | 5/5 PASS | 0.000 | 49.4 | 4/5 **PASS** | PASS | PASS, read-back 0.000 | 21.0 |
| synth horizontal 000000 (4 blocks) | horizontal Japanese, free text on screentone | 4/4 PASS | 0.000 | 46.1 | 2/4 FAIL | PASS | PASS, read-back 0.076 | 22.9 |
| tall strip 1487x6307 (25 bubbles) | 3 pages in one image | **25/25 PASS** | 0.018 | 38.1 | 16/25 FAIL | PASS | PASS, read-back 0.003 | 50.8 |
| page01 at 743x1024, JPEG q40 | low resolution, compression | 13/13 PASS | 0.005 | 30.7 | 8/13 FAIL | **FAIL: 5 residual regions** (38/141 chars still readable) | PASS, min font 10 px | 26.2 |

**Horizontal Latin source.** `$W translate out/proof/hostile/page01_id.png --from id --to en -o
out/proof/hostile/id_en/out.png --debug-dir ...` ran in 26 s. OCR switched to `48px` automatically.
Scored with `metrics.match_detections` against the exact rendered strings and hand-written English
references ([proof/reports/horizontal_latin_id_to_en.md](proof/reports/horizontal_latin_id_to_en.md)):

- 13/14 bubbles matched, 0 false positives, mean CER 0.080, page chrF 53.4.
- "Iya, iya" was read as `啊珈` with confidence 0.24, so the low-confidence guard left it untouched.
- Two partial reads truncated their translations: hyphenated "si Reaks itu sangat mena-" (CER 0.41) and the tiny-font "Aku akan" (CER 0.42).

| synth 000002 | synth 000007 | synth horizontal |
|---|---|---|
| ![](proof/synth_000002.jpg) | ![](proof/synth_000007.jpg) | ![](proof/synth_horizontal_000000.jpg) |

| tall strip | Indonesian page to English |
|---|---|
| ![](proof/tall_strip.jpg) | ![](proof/horizontal_latin_id_to_en.jpg) |

**Failures, shown rather than hidden:**

- **Free-floating white text over art (synth 000003):** two adjacent columns are detected as one
  region plus a duplicate. The renderer then draws two overlapping translations in a narrow column.
  ![](proof/FAIL_synth_000003_free_text.jpg)
- **Half-resolution JPEG q40 page01:** the glyph mask misses the JPEG-ringed edges of the glyphs, so
  grey specks of the original text stay in the balloons (inpainted panel).
  ![](proof/FAIL_page01_lowres_q40.jpg)

## 6. Web UI and queue consumer

### Web UI

**Command:** `STORAGE_DIR=out/proof/web/data $W web --port 8765`.

**Playwright run:**
`NODE_PATH=/opt/node-tools/node_modules node docs/proof/scripts/web_flow.js http://127.0.0.1:8765 eval/fixtures/page01.png out/proof/web/shots`
drives Chromium 1243 through the following steps (log:
[proof/logs/web_flow.log](proof/logs/web_flow.log); server requests:
[proof/logs/web_server.log](proof/logs/web_server.log)):

1. Upload page01.
2. Translate with the defaults: done in 26.1 s with 14 bubbles.
3. Switch the right pane to detect, mask and inpainted.
4. Edit bubble 13's translation and untick the SFX bubble 14.
5. Re-typeset: 2.9 s, with no detector, OCR or translator re-run.
6. Download the PNG (3.2 MB).
7. Run a second job with `yolo` / `m2m100` / `telea` picked from the dropdowns: 7.3 s, 13 bubbles.
8. Reload: the job comes back from the `#job=` URL.
9. Show the page at phone width.

| empty | running | done |
|---|---|---|
| ![](proof/web_01-empty.jpg) | ![](proof/web_02-running.jpg) | ![](proof/web_03-done.jpg) |
| **detect view** | **mask view** | **hand edit + re-typeset** (bubble 14 kept original) |
| ![](proof/web_05-view-detect.jpg) | ![](proof/web_05-view-mask.jpg) | ![](proof/web_06-retypeset.jpg) |
| **other methods chosen in the UI** | **phone width** | |
| ![](proof/web_07-other-methods.jpg) | ![](proof/web_08-mobile.jpg) | |

### Queue consumer, full stack

The full stack ran as: Go API, then AMQP broker, then `terjemahin-worker serve`, then the results
queue, then back to the API.

**Setup:**

- **Broker:** RabbitMQ 3.12.1, installed with apt for this test (`rabbitmq-server -detached`). It
  speaks the same AMQP 0-9-1 as the LavinMQ in `docker-compose.yml`; there is no Docker daemon here.
- **API:** `apps/api` built with `go build` and run with `STORAGE_DIR=out/proof/queue/data API_ADDR=127.0.0.1:8088`.
- **Worker:** `STORAGE_DIR=out/proof/queue/data $W serve`.
- **Client:** [proof/scripts/queue_demo.py](proof/scripts/queue_demo.py).

**Logs:** [proof/logs/queue_demo.log](proof/logs/queue_demo.log) and
[proof/logs/queue_worker.log](proof/logs/queue_worker.log).

**Results:**

- **Job 1:** `POST /v1/pages` returned 202 `queued`. The worker logged `page done: 14 regions,
  14 bubbles`, and `GET /v1/jobs/{id}` returned `done` after 26.1 s. `/image`, `/result` (14 bubbles)
  and `/debug/{detect,mask,inpainted,final}` all returned 200. The PNG from the queue is
  **byte-identical** to the CLI output of section 3 (md5 `6a4856d3...` for both).
- **Job 2:** `pipeline={"detector":"yolo","translator":"m2m100","inpainter":"telea"}` returned
  `done` in 7.2 s with 13 bubbles. The result JSON shows those methods were used.
- **Malformed message:** published straight onto `terjemahin.jobs`. The worker logged `bad message,
  dropping` and kept consuming.

## 7. Training

All runs are logged with their commands in [proof/logs/training.txt](proof/logs/training.txt).

**Training data:** `python training/cli.py synth --out out/proof/train/synth --n 40 --seed 100 ...`
wrote 40 pages / 199 blocks of detector, OCR and inpaint data in 9 s.
![](proof/train_synth_preview.jpg)

**Detector (YOLO11n).**

- **Shipped weights:** `weights/manga-text-yolo11n.onnx` was trained in this repo with `training/train.py`.
  `$W translate page01.png -d yolo` finds 13/13 GT bubbles, 0 false positives, 0 duplicates, OCR CER 0.005,
  with detect in 0.08 s. In the matrix (section 4) it passes every check `ctd` passes.
- **Fresh run:** `training/cli.py detector --data out/proof/train/synth/det/dataset.yaml --epochs 2
  --imgsz 640 --batch 8 --device cpu` took 16 s. Val mAP50 went 0.001 to 0.123 and it exported
  `det-smoke.onnx` (10.6 MB).
- **Fresh weights in the product:** `-d yolo --opt detector.weights=out/proof/train/det-smoke.onnx
  --opt detector.conf=0.1` runs end to end but gives 73 bubbles, 17 false positives and 33 fragments.
  That is expected from 2 epochs on 36 synthetic pages, and it shows the option really swaps the model.

![](proof/train_yolo.jpg)

**OCR (manga-ocr).**

- **Fine-tune:** `training/cli.py ocr --data out/proof/train/synth/ocr --steps 20 --batch 8 --freeze-encoder`
  took 23 s. Synthetic val CER went 0.0528 to 0.0503 and exact matches 0.667 to 0.708 (24 crops: not significant).
- **Export:** `export-ocr --model .../best --out manga-ocr-ft.onnx` wrote 461 MB. Parity check:
  max |logit diff| 1.1e-5, argmax agreement 1.0.
- **In the product:** `$W translate page01.png --opt ocr.weights=out/proof/train/manga-ocr-ft.onnx`
  logs `loading ocr=manga_ocr {'weights': ...}`. The OCR text is the same on 14/14 bubbles, but the
  confidences differ (0.9979 vs 0.9991, ...), so a different graph ran. Harness: CER 0.008, all
  checks as baseline ([report](proof/reports/train_bench_ocr_ft.md)).
- **Negative control:** `--opt ocr.weights=/nonexistent.onnx` fails with
  `error: ocr.weights: no such ONNX model`.

**Translator (NLLB-600M).**

- **Fine-tune:** `training/cli.py translator --data corpus.tsv --val-ratio 0.25 --steps 10 --batch 4
  --decoder-layers 2 --save-dtype float16` took 16 s, training 33.6M of 615M parameters. Held-out chrF
  stayed at 49.55: the script itself warns that no eval beat the base model.
- **In the product:** `-t nllb --opt translator.model=out/proof/train/nllb-ft` logs
  `loading translator=nllb {'model': ...}`. 4/13 page01 translations change, and page chrF goes
  30.7 to 30.2. This is a smoke test, not an improvement.

**Inpainter (AOT, LaMa ONNX interface).**

- **Fine-tune:** `training/cli.py inpaint --data out/proof/train/synth --steps 30 --size 256 --batch 4
  --export aot-ft.onnx` took 83 s. Hole PSNR went 32.95 to 32.88 dB (Telea 25.34). It exported
  23 MB with ONNX/torch max diff 2.6e-5.
- **In the product:** `-i lama_manga --opt inpainter.weights=out/proof/train/aot-ft.onnx --opt
  inpainter.flat_tol=0` sends every hole through the new model. Inpaint took 6.9 s cold. Stock lama_manga took 11.1 s cold, but
  with its default flat-fill shortcut, so the two times are not directly comparable. Harness inpaint check PASS: 0 residual regions, 0 ghost, outside MAD 0.037
  ([report](proof/reports/train_bench_aot_ft.md)).

![](proof/train_inpaint_aot.jpg)

**Cleanup:** the >200 MB checkpoints (OCR checkpoint and ONNX, NLLB checkpoint) were deleted after use.
Their metrics are kept in `out/proof/train/*.json`.

## 8. Latency on this CPU (page01, 1487x2048, defaults)

| stage | method | cold: first call in a fresh process (s) | warm: second uncached run (s) | CLI cold process (s) |
|---|---|---|---|---|
| detect | ctd | 0.67 | 0.52 | 0.68 |
| OCR | manga_ocr | 1.34 (+2.72 model build) | 1.13 | 1.31 |
| translate | nllb | 8.58 | 3.65 | 8.27 |
| mask | default | 0.09 | 0.08 | 0.08 |
| inpaint | lama_manga | 11.17 | 3.49 | 11.09 |
| render | default | 0.73 | 0.58 | 0.71 |
| **total** | | **22.6** | **9.5** | 22.1 in stages, **26.7 wall** |

- **Cold and warm columns:** from `bench_base` (`latency` and `warm latency` lines of
  [the baseline report](proof/reports/bench_baseline_report.md)).
- **CLI column:** from section 3.
- **Long-lived servers:** the second job is fast. Web: 26.1 s cold, then 7.3 s with
  yolo/m2m100/telea. Queue worker: 26.1 s, then 7.2 s. Web re-typeset: 2.9 s.
- **Alternative methods:** first-call stage times are in the matrix table above. For example,
  local_llm takes 47.4 s and the classical detector 0.06 s.
- **Shared CPU:** the cores were shared with another agent, so expect a few seconds of jitter.

## Limitations (read these)

- **Translation quality is the weak stage.** The default `nllb` fails the harness's meaning check on
  page01 (8/13 adequate, need >= 70%, i.e. 10). On five of the six extra pages, too few bubbles reach the
  harness's adequacy threshold. Only `m2m100` and `local_llm` (Qwen2.5-3B, 47 s/page, ~10 GB RAM) pass on page01.
- **API translators were not run for real.** `anthropic`, `gemini`, `openai` and `deepseek` were
  tested only against `httpx.MockTransport` fakes of each provider's wire format (44 tests in
  `tests/test_translate_llm.py`), because no keys exist here. `ollama` was not run because there is no server.
- **Large models not run:** `m2m100_big` and `nllb_big` were not run (5 GB downloads; ~5 GB of free disk).
- **Low-resolution, compressed scans** leave text residue after inpainting (section 5 failure 2).
- **Free-floating text over art** can be merged or duplicated by the detector, and the renderer then
  overlaps the translations (section 5 failure 1).
- **SFX:** handwritten SFX such as あとね on page01 are detected and translated like dialogue, sometimes wrongly.
- **Synthetic pages:** they use 6 fonts and a 20-line corpus with my own reference translations.
  They test the pipeline mechanics with exact GT, not real-world translation difficulty. Several
  corpus lines also appear in `synth.py`'s built-in corpus used for the training smoke runs.
- **Training runs are smoke tests:** 2 epochs, 10 to 30 steps, on CPU. They prove that each trainer
  runs, exports, and that the export loads into the product through `--opt`. They do not improve
  quality, and none of the metrics moved meaningfully.
- **Broker:** the queue test used RabbitMQ, not LavinMQ, and an in-memory job store in the Go API (as
  shipped: `jobs.NewMemoryStore`, no Postgres). Docker compose itself was not exercised.
- **Not covered:** GPU and `DEVICE=cuda` were not tested.
- **Uncommitted tree:** the evidence reflects the working tree, including uncommitted changes by
  other agents at the time of the run (see the tree note at the top).
