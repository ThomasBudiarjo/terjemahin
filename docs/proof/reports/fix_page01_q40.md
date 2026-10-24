# terjemahin benchmark

Page 743x1024; baseline detector=`ctd`, ocr=`manga_ocr`, translator=`m2m100`, mask_refiner=`default`, inpainter=`lama_manga`, renderer=`default`. Each matrix row changes one stage, the rest stay at the baseline.

Columns: detection = GT bubbles matched by OCR text (Hungarian on CER <= 0.5), false positives and duplicate detections; OCR = mean CER over matched bubbles; chrF vs `id_ref` (missing bubbles count as empty), e/j = empty / Japanese outputs; meaning = GT bubbles whose translation has embedding cosine >= 0.8 to its reference (paraphrase-multilingual-MiniLM) and no meaning conflict with it (flipped negation, swapped or added participants, active for passive), and the mean cosine; detection counts a bubble whose box covers less than 75% of its text as partial, not found; residual = text regions the judge detector+OCR still reads inside erased areas / dark and off-tone (ghost) pixel shares left in erased balloon interiors; outside = mean abs diff / changed share of pixels OUTSIDE the mask / changed pixels more than 8 px from it (must be 0: only edge feathering may touch the ring); inside = worst share of a bubble's rendered ink inside its balloon (or padded box) and bubbles whose text was lost; read-back = mean / worst CER (letters only) of the rendered text read back from pixels by an independent OCR (`48px`) against the translation; font = smallest font px; total = pipeline seconds for that config, stages reused from the cache counted at the baseline's warm time.

| run | status | detection | OCR CER | chrF | meaning | residual | outside mask | inside | read-back CER | min font | stage s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | ok | 13/13 fp 0 dup 0 PASS | 0.005 PASS | 32.4 e0 j0 | 10/13 0.86 PASS | 0 / 0.0000 / 0.0087 | 0.070 / 0.0094 / 0 PASS | 0.722 lost 1 | 0.047 / 0.54 | 10 FAIL | – | 26.4 |

## Baseline checks

- **detect** PASS: 13/13 GT bubbles matched, 0 of them partial (box covers < 75% of the text: none), 0 not fully found (need >= 92% fully found), 0 false positives, 1 SFX regions, 0 fragments, 0 duplicates, 14 raw regions.
- **ocr** PASS: mean CER 0.005 (need <= 0.15), 0.006 ignoring punctuation, 12/13 exact.
- **translate** PASS: chrF 32.4 (need >= 25.0), chrF++ 32.9, 0 empty, 0 still Japanese of 14; meaning: 10/13 GT bubbles adequate (cosine >= 0.8 and no meaning conflict, need >= 70%), mean cosine 0.859, 12 closest to their own reference; inadequate GT #2, #9, #11; conflicts: #11 participants.
- **inpaint** PASS: 55418 px erased, 0 residual text regions (0 chars of 141 erased), dark residue 0.0000 (was 0.186 before erasing), ghost 0.0087 (55084 px), outside-mask MAD 0.0696, max 253, changed 0.009368 (all within 8 px of the mask: True).
- **render** FAIL: 14 bubbles, worst inside share 0.7216, 120 ink px outside, 1 lost, layout chars 302/302 (14 layouts), read back from pixels: 299/302 letters, CER mean 0.047 (need <= 0.1), worst 0.538 (need <= 0.34), 1 overflowing, font 10..14 px (min..median, layout).
- **latency** (s, first call, includes lazy model loading): detect 0.72, ocr 1.23, translate 6.85, mask 0.16, inpaint 17.18, render 0.22; total 26.4.

## Baseline per bubble

| # | GT | OCR | CER | translation | reference | chrF | cosine | read back from pixels | read CER | font | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ねえねえ、めいは坂本のことどう思う？ | ねえねえ、めいは坂本のことどう思う？ | 0.00 | Hei, hei, apa pendapat kamu tentang Sakamoto? | Hei, hei, Mei, menurutmu Sakamoto itu gimana? | 36.6 | 0.90 | Hei, hei, apa pendapat kamu tentang Sakamoto? | 0.00 | 15 | 1.000 |
| 1 | 坂本くん？ | 坂本くん？ | 0.00 | Sakamoto-kun? | Sakamoto-kun? | 100.0 | 0.99 | Sakamoto-kun? | 0.00 | 12 | 1.000 |
| 2 | うーんそうだなぁ・・優しいよね | うーんそうだなあ．．優しいよね | 0.07 | Hmm, ya, ya, itu baik-baik saja | Hmm, gimana ya... dia baik, kan? | 19.8 | 0.61 | Hmm, ya, ya, itu baik-baik saja | 0.00 | 14 | 1.000 |
| 3 | えー、それだけー？ | えー、それだけー？ | 0.00 | Eh, itu saja? | Eh, cuma itu? | 22.5 | 0.96 | Eh, itu saja? | 0.00 | 17 | 1.000 |
| 4 | じゃあメシアはどうなの？ | じゃあメシアはどうなの？ | 0.00 | Kalau gitu, bagaimana dengan Messias? | Kalau gitu, Mesia sendiri gimana? | 48.0 | 0.96 | Kalau gitu, bagaimana de ngan Messias? | 0.00 | 15 | 1.000 |
| 5 | え？ | え？ | 0.00 | Eh? | Eh? | 100.0 | 0.99 | Eh? | 0.00 | 17 | 1.000 |
| 6 | わ、私はー・・・別にそんな・・・ | わ、私はー．．．別にそんな．． | 0.00 | A-aku... tidak seperti itu... | A-aku sih... nggak gimana-gimana... | 18.9 | 0.85 | A-aku... tidak seperti itu... | 0.00 | 17 | 1.000 |
| 7 | 反応がいちいち面白いなーとか | 反応がいちいち面白いなーとか | 0.00 | Reaksi itu semua lucu, gitu | Reaksinya selalu lucu, gitu deh | 45.9 | 0.92 | Reaksi itu semua lucu, gitu | 0.00 | 13 | 1.000 |
| 8 | うんうん | うんうん | 0.00 | Iya, iya | Iya, iya | 100.0 | 1.00 | Iya, iya | 0.00 | 13 | 1.000 |
| 9 | 店の店員さんとかにも敬語使ってて偉いなーとか | 店の店員さんとかにも敬語使ってて偉いなーとか | 0.00 | Kedai-kedai ini juga menggunakan bahasa yang baik, gitu | Dia sopan pakai bahasa halus bahkan ke pelayan toko, keren gitu | 23.2 | 0.62 | Kedai-kedai ini juga meng-gunakan bahasa yang baik, gitu | 0.00 | 15 | 1.000 |
| 10 | わあ、いいね！ | わあ、いいね！ | 0.00 | Wah, baiklah ya! | Wah, bagus dong! | 30.0 | 0.88 | Wah, baik-lah ya! | 0.00 | 16 | 1.000 |
| 11 | あっ、あとこないだね、髪型変えたらすぐに気付いてくれてっ・・ | あっ、あとこないだね、髪型変えたらすぐに気付いてくれてっ．．． | 0.00 | Ah, jika kamu ingin mengubah rambut kamu, kamu akan segera menyadari... | Ah, terus kemarin itu, waktu aku ganti gaya rambut, dia langsung sadar... | 24.3 | 0.62 | Ah，jika kamu ingin mengubah rambut kamu, kamu akan segera menya-dari | 0.00 | 13 | 1.000 |
| 12 | ふふっ、他には？ | ふふっ、他には？ | 0.00 | Fufu, untuk yang lain? | Fufu, terus apa lagi? | 23.2 | 0.87 | Fufu, untuk yang 10in? | 0.12 | 13 | 1.000 |
| 13 | *sfx* | あとね | – | Selanjutnya ya |  | – | – | se1on-nynt. a | 0.54 | 10 | 0.722 |

## Model load (s)

detector:ctd 0.21, ocr:manga_ocr 2.74, translator:m2m100 0.0, mask_refiner:default 0.0, inpainter:lama_manga 0.0, renderer:default 0.0, judge:48px 0.44


