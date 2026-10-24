# terjemahin benchmark

Page 1654x1170; baseline detector=`ctd`, ocr=`manga_ocr`, translator=`m2m100`, mask_refiner=`default`, inpainter=`lama_manga`, renderer=`default`. Each matrix row changes one stage, the rest stay at the baseline.

Columns: detection = GT bubbles matched by OCR text (Hungarian on CER <= 0.5), false positives and duplicate detections; OCR = mean CER over matched bubbles; chrF vs `id_ref` (missing bubbles count as empty), e/j = empty / Japanese outputs; meaning = GT bubbles whose translation has embedding cosine >= 0.8 to its reference (paraphrase-multilingual-MiniLM) and no meaning conflict with it (flipped negation, swapped or added participants, active for passive), and the mean cosine; detection counts a bubble whose box covers less than 75% of its text as partial, not found; residual = text regions the judge detector+OCR still reads inside erased areas / dark and off-tone (ghost) pixel shares left in erased balloon interiors; outside = mean abs diff / changed share of pixels OUTSIDE the mask / changed pixels more than 8 px from it (must be 0: only edge feathering may touch the ring); inside = worst share of a bubble's rendered ink inside its balloon (or padded box) and bubbles whose text was lost; read-back = mean / worst CER (letters only) of the rendered text read back from pixels by an independent OCR (`48px`) against the translation; font = smallest font px; total = pipeline seconds for that config, stages reused from the cache counted at the baseline's warm time.

| run | status | detection | OCR CER | chrF | meaning | residual | outside mask | inside | read-back CER | min font | stage s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | ok | 12/12 fp 1 dup 0 FAIL | 0.014 PASS | 0.0 e0 j0 | – – n/a | 0 / 0.0000 / 0.0000 | 0.015 / 0.0026 / 0 PASS | 1.000 lost 0 | 0.000 / 0.00 | 13 PASS | – | 20.5 |

## Baseline checks

- **detect** FAIL: 12/12 GT bubbles matched, 0 of them partial (box covers < 75% of the text: none), 0 not fully found (need >= 92% fully found), 1 false positives, 0 SFX regions, 0 fragments, 0 duplicates, 13 raw regions.
- **ocr** PASS: mean CER 0.014 (need <= 0.15), 0.006 ignoring punctuation, 10/12 exact.
- **translate** n/a: chrF 0.0 (need >= 25.0), chrF++ 0.0, 0 empty, 0 still Japanese of 13; meaning NOT checked (the GT has no id_ref).
- **inpaint** PASS: 19793 px erased, 0 residual text regions (0 chars of 69 erased), dark residue 0.0000 (was 0.134 before erasing), ghost 0.0000 (17321 px), outside-mask MAD 0.0152, max 255, changed 0.002578 (all within 8 px of the mask: True).
- **render** PASS: 13 bubbles, worst inside share 1.0000, 0 ink px outside, 0 lost, layout chars 129/129 (13 layouts), read back from pixels: 129/129 letters, CER mean 0.000 (need <= 0.1), worst 0.000 (need <= 0.34), 7 overflowing, font 13..13 px (min..median, layout).
- **latency** (s, first call, includes lazy model loading): detect 1.87, ocr 1.10, translate 5.80, mask 0.16, inpaint 11.47, render 0.14; total 20.5.

## Baseline per bubble

| # | GT | OCR | CER | translation | reference | chrF | cosine | read back from pixels | read CER | font | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | なに…？ | なに．．．？ | 0.00 | Apa? |  | 0.0 | – | Apa? | 0.00 | 14 | 1.000 |
| 1 | ドクン | ドクン | 0.00 | Dokun |  | 0.0 | – | D-okun | 0.00 | 13 | 1.000 |
| 2 | ドクン | ドクン | 0.00 | Dokun |  | 0.0 | – | D-okun | 0.00 | 13 | 1.000 |
| 3 | ドクン | ドクン | 0.00 | Dokun |  | 0.0 | – | D-okun | 0.00 | 13 | 1.000 |
| 4 | 何かいる…のか？ | 何かいる．．．のか？ | 0.00 | Apakah ada sesuatu? |  | 0.0 | – | Apakah ada sesu-atu? | 0.00 | 14 | 1.000 |
| 5 | ドクン | ドクン | 0.00 | Dokun |  | 0.0 | – | D-okun | 0.00 | 13 | 1.000 |
| 6 | ドクン | ドクン | 0.00 | Dokun |  | 0.0 | – | D-okun | 0.00 | 13 | 1.000 |
| 7 | ドクン | ドクン | 0.00 | Dokun |  | 0.0 | – | D-okun | 0.00 | 13 | 1.000 |
| 8 | グリフォン、スクロップ・テイクオフ | グリフォン。スクロップ・ティクオフ | 0.12 | Dan Grifon. Teka-teki dari scrap |  | 0.0 | – | Dan Gri-fon. Teka-teki dari scrap | 0.00 | 13 | 1.000 |
| 9 | *false_positive* | ナオォ | – | Nagoya |  | – | – | Nagoya | 0.00 | 16 | 1.000 |
| 10 | エリー | エリー | 0.00 | Ali yang |  | 0.0 | – | Ali yang | 0.00 | 13 | 1.000 |
| 11 | こちら“パック”先に行って待ってるからね | こちら”パック”先に行って待ってるからね | 0.05 | “Pak, aku akan pergi dan menunggu dulu |  | 0.0 | – | “Pak, aku akan pergi dan menunggu dulu | 0.00 | 15 | 1.000 |
| 12 | エリー？ | エリー？ | 0.00 | Dengan Ellie? |  | 0.0 | – | Dengan Ellie? | 0.00 | 13 | 1.000 |

## Model load (s)

detector:ctd 0.2, ocr:manga_ocr 2.72, translator:m2m100 0.0, mask_refiner:default 0.0, inpainter:lama_manga 0.0, renderer:default 0.0, judge:48px 0.39


