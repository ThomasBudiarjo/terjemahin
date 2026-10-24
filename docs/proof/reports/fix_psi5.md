# terjemahin benchmark

Page 640x640; baseline detector=`ctd`, ocr=`manga_ocr`, translator=`m2m100`, mask_refiner=`default`, inpainter=`lama_manga`, renderer=`default`. Each matrix row changes one stage, the rest stay at the baseline.

Columns: detection = GT bubbles matched by OCR text (Hungarian on CER <= 0.5), false positives and duplicate detections; OCR = mean CER over matched bubbles; chrF vs `id_ref` (missing bubbles count as empty), e/j = empty / Japanese outputs; meaning = GT bubbles whose translation has embedding cosine >= 0.8 to its reference (paraphrase-multilingual-MiniLM) and no meaning conflict with it (flipped negation, swapped or added participants, active for passive), and the mean cosine; detection counts a bubble whose box covers less than 75% of its text as partial, not found; residual = text regions the judge detector+OCR still reads inside erased areas / dark and off-tone (ghost) pixel shares left in erased balloon interiors; outside = mean abs diff / changed share of pixels OUTSIDE the mask / changed pixels more than 8 px from it (must be 0: only edge feathering may touch the ring); inside = worst share of a bubble's rendered ink inside its balloon (or padded box) and bubbles whose text was lost; read-back = mean / worst CER (letters only) of the rendered text read back from pixels by an independent OCR (`48px`) against the translation; font = smallest font px; total = pipeline seconds for that config, stages reused from the cache counted at the baseline's warm time.

| run | status | detection | OCR CER | chrF | meaning | residual | outside mask | inside | read-back CER | min font | stage s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | ok | 6/6 fp 0 dup 0 PASS | 0.000 PASS | 36.9 e0 j0 | 2/6 0.67 FAIL | 0 / 0.0000 / 0.0000 | 0.038 / 0.0075 / 0 PASS | 0.971 lost 0 | 0.005 / 0.03 | 10 FAIL | – | 9.8 |

## Baseline checks

- **detect** PASS: 6/6 GT bubbles matched, 0 of them partial (box covers < 75% of the text: none), 0 not fully found (need >= 92% fully found), 0 false positives, 1 SFX regions, 0 fragments, 0 duplicates, 7 raw regions.
- **ocr** PASS: mean CER 0.000 (need <= 0.15), 0.000 ignoring punctuation, 6/6 exact.
- **translate** FAIL: chrF 36.9 (need >= 25.0), chrF++ 33.3, 0 empty, 0 still Japanese of 7; meaning: 2/6 GT bubbles adequate (cosine >= 0.8 and no meaning conflict, need >= 70%), mean cosine 0.672, 6 closest to their own reference; inadequate GT #0, #1, #2, #5; conflicts: #5 negation/participants.
- **inpaint** PASS: 19334 px erased, 0 residual text regions (0 chars of 103 erased), dark residue 0.0000 (was 0.111 before erasing), ghost 0.0000 (19259 px), outside-mask MAD 0.0376, max 234, changed 0.007538 (all within 8 px of the mask: True).
- **render** FAIL: 7 bubbles, worst inside share 0.9710, 47 ink px outside, 0 lost, layout chars 259/259 (7 layouts), read back from pixels: 259/259 letters, CER mean 0.005 (need <= 0.1), worst 0.033 (need <= 0.34), 4 overflowing, font 10..10 px (min..median, layout).
- **latency** (s, first call, includes lazy model loading): detect 0.71, ocr 1.00, translate 7.72, mask 0.17, inpaint 0.01, render 0.15; total 9.8.

## Baseline per bubble

| # | GT | OCR | CER | translation | reference | chrF | cosine | read back from pixels | read CER | font | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | …ずっと | ．．．ずっと | 0.00 | Selamanya | ...Terus-menerus. | 3.2 | 0.45 | Sela-manya | 0.00 | 10 | 1.000 |
| 1 | 同じ夢を見るんだ | 同じ夢を見るんだ | 0.00 | Mimpi yang sama | Aku selalu memimpikan mimpi yang sama. | 35.6 | 0.79 | Mimpi yang sama | 0.00 | 12 | 1.000 |
| 2 | *sfx* | ★この物語はフィクションであり、実在の人物・国体・出来事などとは、一切剛体ありません。 | – | Kisah ini adalah fiksi, dan tidak ada apa-apa yang berkaitan dengan orang-orang, negara, peristiwa, dan lain-lain. |  | – | – | Kisah ini adalah fiksi, dan tidak ada apa-apa yang berkaitan denaan orana-orana, negara, perist iwa, dan lain-lain, | 0.03 | 10 | 0.971 |
| 3 | 開拓地で首を吊ったおじさんの夢だ | 開拓地で首を吊ったおじさんの夢だ | 0.00 | Ini adalah mimpi Ttaoji-san yang menggantung leher di tempat terbuka | Mimpi tentang paman yang gantung diri di daerah pemukiman. | 35.3 | 0.61 | Ini adalah mimpi Ttaoji-san yang me ngg ant ung leher di tempat terbuka | 0.00 | 10 | 1.000 |
| 4 | 何で首をくくる前に僕達にあんな話したんだろうって… | 何で首をくくる前に僕達にあんな話したんだろうって．．． | 0.00 | Kenapa dia mengatakan hal ini kepada kami sebelum dia menggelengkan leher... | Kenapa dia menceritakan hal itu kepada kami sebelum gantung diri... | 59.1 | 0.88 | Kenapa dia mengat akan hal ini kepada kami sebel um dia menggelengkan leher... | 0.00 | 10 | 1.000 |
| 5 | そんなの… | そんなの．．． | 0.00 | Begitulah... | Hal seperti itu... | 17.1 | 0.90 | Begi-tu-lah.. | 0.00 | 10 | 1.000 |
| 6 | わかるわけないだろ | わかるわけないだろ | 0.00 | Kau tidak mengerti | Mana mungkin kami tahu. | 9.6 | 0.41 | Kau tidak mengerti | 0.00 | 10 | 1.000 |

## Model load (s)

detector:ctd 0.23, ocr:manga_ocr 2.68, translator:m2m100 0.0, mask_refiner:default 0.0, inpainter:lama_manga 0.0, renderer:default 0.0, judge:48px 0.43


