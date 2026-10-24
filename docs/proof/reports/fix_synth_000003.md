# terjemahin benchmark

Page 1091x1617; baseline detector=`ctd`, ocr=`manga_ocr`, translator=`m2m100`, mask_refiner=`default`, inpainter=`lama_manga`, renderer=`default`. Each matrix row changes one stage, the rest stay at the baseline.

Columns: detection = GT bubbles matched by OCR text (Hungarian on CER <= 0.5), false positives and duplicate detections; OCR = mean CER over matched bubbles; chrF vs `id_ref` (missing bubbles count as empty), e/j = empty / Japanese outputs; meaning = GT bubbles whose translation has embedding cosine >= 0.8 to its reference (paraphrase-multilingual-MiniLM) and no meaning conflict with it (flipped negation, swapped or added participants, active for passive), and the mean cosine; detection counts a bubble whose box covers less than 75% of its text as partial, not found; residual = text regions the judge detector+OCR still reads inside erased areas / dark and off-tone (ghost) pixel shares left in erased balloon interiors; outside = mean abs diff / changed share of pixels OUTSIDE the mask / changed pixels more than 8 px from it (must be 0: only edge feathering may touch the ring); inside = worst share of a bubble's rendered ink inside its balloon (or padded box) and bubbles whose text was lost; read-back = mean / worst CER (letters only) of the rendered text read back from pixels by an independent OCR (`48px`) against the translation; font = smallest font px; total = pipeline seconds for that config, stages reused from the cache counted at the baseline's warm time.

| run | status | detection | OCR CER | chrF | meaning | residual | outside mask | inside | read-back CER | min font | stage s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | ok | 4/4 fp 0 dup 1 FAIL | 0.000 PASS | 35.7 e0 j0 | 4/4 0.88 PASS | 0 / 0.0000 / 0.0000 | 0.033 / 0.0041 / 0 PASS | 0.989 lost 3 | 0.552 / 1.00 | 13 FAIL | – | 16.7 |

## Baseline checks

- **detect** FAIL: 4/4 GT bubbles matched, 0 of them partial (box covers < 75% of the text: none), 0 not fully found (need >= 92% fully found), 0 false positives, 0 SFX regions, 0 fragments, 1 duplicates, 4 raw regions.
- **ocr** PASS: mean CER 0.000 (need <= 0.15), 0.000 ignoring punctuation, 4/4 exact.
- **translate** PASS: chrF 35.7 (need >= 25.0), chrF++ 34.0, 0 empty, 0 still Japanese of 5; meaning: 4/4 GT bubbles adequate (cosine >= 0.8 and no meaning conflict, need >= 70%), mean cosine 0.879, 4 closest to their own reference; inadequate GT #; conflicts: none.
- **inpaint** PASS: 39626 px erased, 0 residual text regions (0 chars of 52 erased), dark residue 0.0000 (was 0.116 before erasing), ghost 0.0000 (33858 px), outside-mask MAD 0.0327, max 217, changed 0.004089 (all within 8 px of the mask: True).
- **render** FAIL: 5 bubbles, worst inside share 0.9893, 6 ink px outside, 3 lost, layout chars 125/125 (5 layouts), read back from pixels: 104/125 letters, CER mean 0.552 (need <= 0.1), worst 1.000 (need <= 0.34), 2 overflowing, font 13..19 px (min..median, layout).
- **latency** (s, first call, includes lazy model loading): detect 0.65, ocr 0.45, translate 5.45, mask 0.16, inpaint 9.79, render 0.15; total 16.6.

## Baseline per bubble

| # | GT | OCR | CER | translation | reference | chrF | cosine | read back from pixels | read CER | font | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 誰かいるの？ | 誰かいるの？ | 0.00 | Ada yang ada? | Ada orang di sana? | 20.8 | 0.86 | ri-IーI I | 1.00 | 13 | 0.989 |
| 1 | *duplicate* | 誰かいるの？ | – | Ada yang ada? |  | – | – | 一ーーー | 1.00 | 22 | 1.000 |
| 2 | 早く逃げろ！！ | 早く逃げろ！！ | 0.00 | Cepatlah melarikan diri! | Cepat lari!! | 54.6 | 0.85 | gep086 iken | 0.76 | 13 | 1.000 |
| 3 | わかった、行くよ静かにして！え？今なんて言った？ | わかった、行くよ静かにして！え？今なんて言った？ | 0.00 | Aku tahu, aku akan pergi, tenanglah! ya ya? Apa yang kau katakan sekarang? | Baiklah, aku pergi Diam! Hah? Barusan kamu bilang apa? | 21.7 | 0.88 | Aku tahu, aku akan pergi, tenanglah! ya ya? Apa yang kau katakan sekarang? | 0.00 | 25 | 1.000 |
| 4 | わかった、行くよ私の名前はさくらです | わかった、行くよ私の名前はさくらです | 0.00 | Aku tahu, aku pergi, namaku Sakura | Baiklah, aku pergi Namaku Sakura | 60.1 | 0.93 | Aku tahu, aku pergi, namaku Sakura | 0.00 | 19 | 1.000 |

## Model load (s)

detector:ctd 0.19, ocr:manga_ocr 2.7, translator:m2m100 0.0, mask_refiner:default 0.0, inpainter:lama_manga 0.0, renderer:default 0.0, judge:48px 0.45


