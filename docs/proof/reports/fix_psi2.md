# terjemahin benchmark

Page 640x640; baseline detector=`ctd`, ocr=`manga_ocr`, translator=`m2m100`, mask_refiner=`default`, inpainter=`lama_manga`, renderer=`default`. Each matrix row changes one stage, the rest stay at the baseline.

Columns: detection = GT bubbles matched by OCR text (Hungarian on CER <= 0.5), false positives and duplicate detections; OCR = mean CER over matched bubbles; chrF vs `id_ref` (missing bubbles count as empty), e/j = empty / Japanese outputs; meaning = GT bubbles whose translation has embedding cosine >= 0.8 to its reference (paraphrase-multilingual-MiniLM) and no meaning conflict with it (flipped negation, swapped or added participants, active for passive), and the mean cosine; detection counts a bubble whose box covers less than 75% of its text as partial, not found; residual = text regions the judge detector+OCR still reads inside erased areas / dark and off-tone (ghost) pixel shares left in erased balloon interiors; outside = mean abs diff / changed share of pixels OUTSIDE the mask / changed pixels more than 8 px from it (must be 0: only edge feathering may touch the ring); inside = worst share of a bubble's rendered ink inside its balloon (or padded box) and bubbles whose text was lost; read-back = mean / worst CER (letters only) of the rendered text read back from pixels by an independent OCR (`48px`) against the translation; font = smallest font px; total = pipeline seconds for that config, stages reused from the cache counted at the baseline's warm time.

| run | status | detection | OCR CER | chrF | meaning | residual | outside mask | inside | read-back CER | min font | stage s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | ok | 3/3 fp 0 dup 0 PASS | 0.000 PASS | 44.1 e0 j0 | 2/3 0.84 FAIL | 0 / 0.0000 / 0.0000 | 0.051 / 0.0050 / 0 PASS | 1.000 lost 0 | 0.000 / 0.00 | 10 PASS | – | 15.9 |

## Baseline checks

- **detect** PASS: 3/3 GT bubbles matched, 0 of them partial (box covers < 75% of the text: none), 0 not fully found (need >= 92% fully found), 0 false positives, 2 SFX regions, 0 fragments, 0 duplicates, 5 raw regions.
- **ocr** PASS: mean CER 0.000 (need <= 0.15), 0.000 ignoring punctuation, 3/3 exact.
- **translate** FAIL: chrF 44.1 (need >= 25.0), chrF++ 42.0, 0 empty, 0 still Japanese of 5; meaning: 2/3 GT bubbles adequate (cosine >= 0.8 and no meaning conflict, need >= 70%), mean cosine 0.843, 3 closest to their own reference; inadequate GT #1; conflicts: none.
- **inpaint** PASS: 10451 px erased, 0 residual text regions (0 chars of 24 erased), dark residue 0.0000 (was 0.206 before erasing), ghost 0.0000 (10077 px), outside-mask MAD 0.0507, max 240, changed 0.005026 (all within 8 px of the mask: True).
- **render** PASS: 5 bubbles, worst inside share 1.0000, 0 ink px outside, 0 lost, layout chars 87/87 (5 layouts), read back from pixels: 87/87 letters, CER mean 0.000 (need <= 0.1), worst 0.000 (need <= 0.34), 1 overflowing, font 10..13 px (min..median, layout).
- **latency** (s, first call, includes lazy model loading): detect 0.71, ocr 0.41, translate 5.09, mask 0.09, inpaint 9.62, render 0.04; total 16.0.

## Baseline per bubble

| # | GT | OCR | CER | translation | reference | chrF | cosine | read back from pixels | read CER | font | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 5日の間に随分と増えたな | ５日の間に随分と増えたな | 0.00 | Dalam lima hari ini sudah banyak yang meningkat | Dalam lima hari, jumlahnya bertambah banyak. | 47.1 | 0.92 | Dalam lima hari ini sudah banyak yang meningkat | 0.00 | 13 | 1.000 |
| 1 | *sfx* | ．．． | – | ... |  | – | – | … | 0.00 | 10 | 1.000 |
| 2 | 脱走者は見つかったか？ | 脱走者は見つかったか？ | 0.00 | Apakah orang yang melarikan diri ditemukan? | Apakah pelarian itu sudah ditemukan? | 46.2 | 0.73 | Apakah orang yang melarikan diri ditemukan? | 0.00 | 19 | 1.000 |
| 3 | *sfx* | ！？ | – | !? |  | – | – | !? | 0.00 | 19 | 1.000 |
| 4 | お前は…！ | お前は．．．！ | 0.00 | Kamu adalah! | Kau...! | 11.0 | 0.88 | Kamu ada-lah! | 0.00 | 13 | 1.000 |

## Model load (s)

detector:ctd 0.21, ocr:manga_ocr 2.95, translator:m2m100 0.0, mask_refiner:default 0.0, inpainter:lama_manga 0.0, renderer:default 0.0, judge:48px 0.45


