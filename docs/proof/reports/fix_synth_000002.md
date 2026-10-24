# terjemahin benchmark

Page 912x1292; baseline detector=`ctd`, ocr=`manga_ocr`, translator=`m2m100`, mask_refiner=`default`, inpainter=`lama_manga`, renderer=`default`. Each matrix row changes one stage, the rest stay at the baseline.

Columns: detection = GT bubbles matched by OCR text (Hungarian on CER <= 0.5), false positives and duplicate detections; OCR = mean CER over matched bubbles; chrF vs `id_ref` (missing bubbles count as empty), e/j = empty / Japanese outputs; meaning = GT bubbles whose translation has embedding cosine >= 0.8 to its reference (paraphrase-multilingual-MiniLM) and no meaning conflict with it (flipped negation, swapped or added participants, active for passive), and the mean cosine; detection counts a bubble whose box covers less than 75% of its text as partial, not found; residual = text regions the judge detector+OCR still reads inside erased areas / dark and off-tone (ghost) pixel shares left in erased balloon interiors; outside = mean abs diff / changed share of pixels OUTSIDE the mask / changed pixels more than 8 px from it (must be 0: only edge feathering may touch the ring); inside = worst share of a bubble's rendered ink inside its balloon (or padded box) and bubbles whose text was lost; read-back = mean / worst CER (letters only) of the rendered text read back from pixels by an independent OCR (`48px`) against the translation; font = smallest font px; total = pipeline seconds for that config, stages reused from the cache counted at the baseline's warm time.

| run | status | detection | OCR CER | chrF | meaning | residual | outside mask | inside | read-back CER | min font | stage s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | ok | 7/7 fp 0 dup 0 FAIL | 0.053 PASS | 44.8 e0 j0 | 4/7 0.86 FAIL | 0 / 0.0005 / 0.0047 | 0.041 / 0.0045 / 0 PASS | 1.000 lost 0 | 0.000 / 0.00 | 12 PASS | – | 18.4 |

## Baseline checks

- **detect** FAIL: 7/7 GT bubbles matched, 1 of them partial (box covers < 75% of the text: 0), 1 not fully found (need >= 92% fully found), 0 false positives, 0 SFX regions, 0 fragments, 0 duplicates, 7 raw regions.
- **ocr** PASS: mean CER 0.053 (need <= 0.15), 0.041 ignoring punctuation, 5/7 exact.
- **translate** FAIL: chrF 44.8 (need >= 25.0), chrF++ 40.8, 0 empty, 0 still Japanese of 7; meaning: 4/7 GT bubbles adequate (cosine >= 0.8 and no meaning conflict, need >= 70%), mean cosine 0.860, 6 closest to their own reference; inadequate GT #3, #5, #6; conflicts: #3 participants, #6 voice.
- **inpaint** PASS: 58241 px erased, 0 residual text regions (0 chars of 65 erased), dark residue 0.0005 (was 0.167 before erasing), ghost 0.0047 (57835 px), outside-mask MAD 0.0407, max 253, changed 0.004483 (all within 8 px of the mask: True).
- **render** PASS: 7 bubbles, worst inside share 1.0000, 0 ink px outside, 0 lost, layout chars 150/150 (7 layouts), read back from pixels: 150/150 letters, CER mean 0.000 (need <= 0.1), worst 0.000 (need <= 0.34), 0 overflowing, font 12..20 px (min..median, layout).
- **latency** (s, first call, includes lazy model loading): detect 0.68, ocr 0.62, translate 5.46, mask 0.18, inpaint 11.27, render 0.19; total 18.4.

## Baseline per bubble

| # | GT | OCR | CER | translation | reference | chrF | cosine | read back from pixels | read CER | font | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | あの時の約束、覚えてる？ | あの時の約束、覚えて．．．る？ | 0.08 | Aku ingat janji-janji itu, kau ingat? | Kamu ingat janji waktu itu? | 46.6 | 0.93 | Aku ingat janji-janji itu, kau ingat? | 0.00 | 17 | 1.000 |
| 1 | わかった、行くよ誰かいるの？ | わかった、行くよ誰かいるの？ | 0.00 | Aku tahu, aku akan pergi, ada orang lain? | Baiklah, aku pergi Ada orang di sana? | 36.9 | 0.85 | Aku tahu, aku akan pergi, ada orang lain? | 0.00 | 23 | 1.000 |
| 2 | 一緒に帰ろうよ | 緒に帰ろう | 0.29 | Mari pulang bersama | Ayo pulang bareng | 38.1 | 0.89 | Mari pulang bersama | 0.00 | 20 | 1.000 |
| 3 | 今日は本当にありがとう | 今日は本当にありがとう | 0.00 | Sungguh terima kasih hari ini | Terima kasih banyak untuk hari ini | 49.0 | 0.93 | Sungguh terima kasih hari ini | 0.00 | 19 | 1.000 |
| 4 | 私の名前はさくらです | 私の名前はさくらです | 0.00 | Namaku Sakura | Namaku Sakura | 100.0 | 0.99 | Namaku Sakura | 0.00 | 21 | 1.000 |
| 5 | 一緒に帰ろうよ | 一緒に帰ろうよ | 0.00 | Aku akan pulang bersama-sama | Ayo pulang bareng | 35.0 | 0.77 | Aku akan pulang bersama-sama | 0.00 | 12 | 1.000 |
| 6 | 先生に怒られちゃった | 先生に怒られちゃった | 0.00 | Guru itu marah | Aku dimarahi guru | 28.9 | 0.65 | Guru itu marah | 0.00 | 23 | 1.000 |

## Model load (s)

detector:ctd 0.22, ocr:manga_ocr 2.88, translator:m2m100 0.0, mask_refiner:default 0.0, inpainter:lama_manga 0.0, renderer:default 0.0, judge:48px 0.44


