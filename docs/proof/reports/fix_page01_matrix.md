# terjemahin benchmark

Page 1487x2048; baseline detector=`ctd`, ocr=`manga_ocr`, translator=`m2m100`, mask_refiner=`default`, inpainter=`lama_manga`, renderer=`default`. Each matrix row changes one stage, the rest stay at the baseline.

Columns: detection = GT bubbles matched by OCR text (Hungarian on CER <= 0.5), false positives and duplicate detections; OCR = mean CER over matched bubbles; chrF vs `id_ref` (missing bubbles count as empty), e/j = empty / Japanese outputs; meaning = GT bubbles whose translation has embedding cosine >= 0.8 to its reference (paraphrase-multilingual-MiniLM) and no meaning conflict with it (flipped negation, swapped or added participants, active for passive), and the mean cosine; detection counts a bubble whose box covers less than 75% of its text as partial, not found; residual = text regions the judge detector+OCR still reads inside erased areas / dark and off-tone (ghost) pixel shares left in erased balloon interiors; outside = mean abs diff / changed share of pixels OUTSIDE the mask / changed pixels more than 8 px from it (must be 0: only edge feathering may touch the ring); inside = worst share of a bubble's rendered ink inside its balloon (or padded box) and bubbles whose text was lost; read-back = mean / worst CER (letters only) of the rendered text read back from pixels by an independent OCR (`48px`) against the translation; font = smallest font px; total = pipeline seconds for that config, stages reused from the cache counted at the baseline's warm time.

| run | status | detection | OCR CER | chrF | meaning | residual | outside mask | inside | read-back CER | min font | stage s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 32.4 e0 j0 | 10/13 0.86 PASS | 0 / 0.0000 / 0.0001 | 0.016 / 0.0015 / 0 PASS | 0.944 lost 0 | 0.005 / 0.08 | 16 FAIL | – | 21.3 |
| detector-classical | ok | 13/13 fp 0 dup 0 PASS | 0.035 PASS | 33.1 e0 j0 | 10/13 0.86 PASS | 0 / 0.0000 / 0.0000 | 0.004 / 0.0017 / 0 PASS | 1.000 lost 0 | 0.000 / 0.00 | 25 PASS | 0.06 | 5.7 |
| detector-ctd_blk | ok | 13/13 fp 0 dup 0 PASS | 0.005 PASS | 32.4 e0 j0 | 10/13 0.86 PASS | 0 / 0.0000 / 0.0001 | 0.017 / 0.0023 / 0 PASS | 1.000 lost 0 | 0.013 / 0.11 | 17 PASS | 0.11 | 9.5 |
| detector-yolo | ok | 13/13 fp 0 dup 0 PASS | 0.005 PASS | 32.4 e0 j0 | 10/13 0.86 PASS | 1 / 0.0000 / 0.0001 | 0.006 / 0.0018 / 0 FAIL | 1.000 lost 0 | 0.000 / 0.00 | 21 PASS | 0.03 | 7.8 |
| ocr-48px | ok | 13/13 fp 1 dup 0 FAIL | 0.005 PASS | 32.4 e0 j0 | 10/13 0.86 PASS | 0 / 0.0000 / 0.0001 | 0.016 / 0.0015 / 0 PASS | 1.000 lost 0 | 0.008 / 0.11 | 22 PASS | 2.90 | 12.4 |
| ocr-manga_ocr_hf | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 32.4 e0 j0 | 10/13 0.86 PASS | 0 / 0.0000 / 0.0001 | 0.016 / 0.0015 / 0 PASS | 0.944 lost 0 | 0.005 / 0.08 | 16 FAIL | 1.32 | 10.5 |
| ocr-none | skipped | needs a vision translator (default 'm2m100' is text-only) | | | | | | | | | | | |
| translator-anthropic | skipped | ANTHROPIC_API_KEY not set | | | | | | | | | | | |
| translator-chain | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 31.6 e0 j0 | 9/13 0.84 FAIL | 0 / 0.0000 / 0.0001 | 0.016 / 0.0015 / 0 PASS | 0.967 lost 0 | 0.006 / 0.09 | 16 FAIL | 8.53 | 16.0 |
| translator-deepseek | skipped | DEEPSEEK_API_KEY not set | | | | | | | | | | | |
| translator-gemini | skipped | GEMINI_API_KEY not set | | | | | | | | | | | |
| translator-local_llm | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 38.5 e0 j0 | 10/13 0.89 PASS | 0 / 0.0000 / 0.0001 | 0.016 / 0.0015 / 0 PASS | 1.000 lost 0 | 0.004 / 0.05 | 17 PASS | 47.37 | 54.9 |
| translator-m2m100_big | skipped | large download on first use (pass --allow-downloads) | | | | | | | | | | | |
| translator-nllb | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 30.7 e0 j0 | 8/13 0.84 FAIL | 0 / 0.0000 / 0.0001 | 0.016 / 0.0015 / 0 PASS | 0.998 lost 0 | 0.000 / 0.00 | 16 PASS | 4.15 | 11.6 |
| translator-nllb_big | skipped | large download on first use (pass --allow-downloads) | | | | | | | | | | | |
| translator-none | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 0.0 e14 j0 | 0/13 0.00 FAIL | 0 / 0.0000 / 0.0001 | 0.016 / 0.0015 / 0 PASS | 1.000 lost 0 | – / – | – n/a | 0.00 | 6.8 |
| translator-ollama | skipped | needs a running Ollama server (OLLAMA_BASE_URL) | | | | | | | | | | | |
| translator-openai | skipped | OPENAI_API_KEY not set | | | | | | | | | | | |
| translator-original | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 0.0 e0 j14 | 0/13 0.00 FAIL | 0 / 0.0000 / 0.0001 | 0.016 / 0.0015 / 0 PASS | 1.000 lost 0 | – / – | 19 PASS | 0.00 | 7.5 |
| mask_refiner-bbox | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 32.4 e0 j0 | 10/13 0.86 PASS | 0 / 0.0000 / 0.0000 | 0.048 / 0.0047 / 0 PASS | 1.000 lost 0 | 0.012 / 0.11 | 16 PASS | 0.00 | 11.3 |
| mask_refiner-dilate | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 32.4 e0 j0 | 10/13 0.86 PASS | 0 / 0.0000 / 0.0000 | 0.021 / 0.0011 / 0 PASS | 1.000 lost 0 | 0.000 / 0.00 | 16 PASS | 0.01 | 5.7 |
| inpainter-fill | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 32.4 e0 j0 | 10/13 0.86 PASS | 0 / 0.0000 / 0.0000 | 0.000 / 0.0000 / 0 PASS | 1.000 lost 0 | 0.005 / 0.08 | 16 PASS | 0.28 | 6.3 |
| inpainter-lama | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 32.4 e0 j0 | 10/13 0.86 PASS | 0 / 0.0000 / 0.0001 | 0.016 / 0.0012 / 0 PASS | 0.935 lost 0 | 0.005 / 0.08 | 16 FAIL | 12.48 | 18.5 |
| inpainter-ns | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 32.4 e0 j0 | 10/13 0.86 PASS | 0 / 0.0000 / 0.0019 | 0.000 / 0.0000 / 0 PASS | 1.000 lost 0 | 0.005 / 0.08 | 16 PASS | 0.27 | 6.3 |
| inpainter-telea | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 32.4 e0 j0 | 10/13 0.86 PASS | 0 / 0.0000 / 0.0019 | 0.000 / 0.0000 / 0 PASS | 1.000 lost 0 | 0.005 / 0.08 | 16 PASS | 0.33 | 6.4 |
| renderer-basic | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 32.4 e0 j0 | 10/13 0.86 PASS | 0 / 0.0000 / 0.0001 | 0.016 / 0.0015 / 0 PASS | 0.978 lost 0 | 0.023 / 0.27 | 6 FAIL | 0.03 | 9.8 |
| renderer-vertical | ok | 13/13 fp 0 dup 0 PASS | 0.008 PASS | 32.4 e0 j0 | 10/13 0.86 PASS | 0 / 0.0000 / 0.0001 | 0.016 / 0.0015 / 0 PASS | 0.845 lost 14 | 0.836 / 1.00 | 16 FAIL | 0.61 | 10.4 |

## Baseline checks

- **detect** PASS: 13/13 GT bubbles matched, 0 of them partial (box covers < 75% of the text: none), 0 not fully found (need >= 92% fully found), 0 false positives, 1 SFX regions, 0 fragments, 0 duplicates, 14 raw regions.
- **ocr** PASS: mean CER 0.008 (need <= 0.15), 0.006 ignoring punctuation, 11/13 exact.
- **translate** PASS: chrF 32.4 (need >= 25.0), chrF++ 32.9, 0 empty, 0 still Japanese of 14; meaning: 10/13 GT bubbles adequate (cosine >= 0.8 and no meaning conflict, need >= 70%), mean cosine 0.859, 12 closest to their own reference; inadequate GT #2, #9, #11; conflicts: #11 participants.
- **inpaint** PASS: 216527 px erased, 0 residual text regions (0 chars of 141 erased), dark residue 0.0000 (was 0.200 before erasing), ghost 0.0001 (215815 px), outside-mask MAD 0.0160, max 253, changed 0.001504 (all within 8 px of the mask: True).
- **render** FAIL: 14 bubbles, worst inside share 0.9439, 35 ink px outside, 0 lost, layout chars 302/302 (14 layouts), read back from pixels: 302/302 letters, CER mean 0.005 (need <= 0.1), worst 0.077 (need <= 0.34), 1 overflowing, font 16..28 px (min..median, layout).
- **latency** (s, first call, includes lazy model loading): detect 0.67, ocr 1.32, translate 6.73, mask 0.46, inpaint 11.40, render 0.72; total 21.3.
- **warm latency** (s, second uncached run): detect 0.82, ocr 1.25, translate 2.98, mask 0.31, inpaint 4.42, render 0.66; total 10.4.

## Baseline per bubble

| # | GT | OCR | CER | translation | reference | chrF | cosine | read back from pixels | read CER | font | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ねえねえ、めいは坂本のことどう思う？ | ねえねえ、めいは坂本のことどう思う？ | 0.00 | Hei, hei, apa pendapat kamu tentang Sakamoto? | Hei, hei, Mei, menurutmu Sakamoto itu gimana? | 36.6 | 0.90 | Hei, hei, apa pendapat kamu tentang Sakamoto? | 0.00 | 31 | 1.000 |
| 1 | 坂本くん？ | 坂本くん？ | 0.00 | Sakamoto-kun? | Sakamoto-kun? | 100.0 | 0.99 | Sakamoto-kun? | 0.00 | 25 | 1.000 |
| 2 | うーんそうだなぁ・・優しいよね | うーんそうだなあ．．優しいよね | 0.07 | Hmm, ya, ya, itu baik-baik saja | Hmm, gimana ya... dia baik, kan? | 19.8 | 0.61 | Hmm, ya, ya, itu baik-baik saja | 0.00 | 30 | 1.000 |
| 3 | えー、それだけー？ | えー、それだけー？ | 0.00 | Eh, itu saja? | Eh, cuma itu? | 22.5 | 0.96 | Eh, itu saja? | 0.00 | 34 | 1.000 |
| 4 | じゃあメシアはどうなの？ | じゃあメシアはどうなの？ | 0.00 | Kalau gitu, bagaimana dengan Messias? | Kalau gitu, Mesia sendiri gimana? | 48.0 | 0.96 | Kalau gitu, bagaimana dengan Messias? | 0.00 | 31 | 1.000 |
| 5 | え？ | え？ | 0.00 | Eh? | Eh? | 100.0 | 0.99 | Eh? | 0.00 | 34 | 1.000 |
| 6 | わ、私はー・・・別にそんな・・・ | わ、私はー．．．別にそんな．． | 0.00 | A-aku... tidak seperti itu... | A-aku sih... nggak gimana-gimana... | 18.9 | 0.85 | A-aku... tidak seperti itu... | 0.00 | 34 | 1.000 |
| 7 | 反応がいちいち面白いなーとか | 反応がいちいち面白いなーとか | 0.00 | Reaksi itu semua lucu, gitu | Reaksinya selalu lucu, gitu deh | 45.9 | 0.92 | Reaksi itu semua lucu, gitu | 0.00 | 26 | 1.000 |
| 8 | うんうん | うんうん | 0.00 | Iya, iya | Iya, iya | 100.0 | 1.00 | Iya, iya | 0.00 | 26 | 1.000 |
| 9 | 店の店員さんとかにも敬語使ってて偉いなーとか | 店の店員さんとかにも敬語使ってて偉いなーとか | 0.00 | Kedai-kedai ini juga menggunakan bahasa yang baik, gitu | Dia sopan pakai bahasa halus bahkan ke pelayan toko, keren gitu | 23.2 | 0.62 | Kedai-kedai ini juga menggunakan bahasa yang baik, gitu | 0.00 | 27 | 1.000 |
| 10 | わあ、いいね！ | わあ、いいね！ | 0.00 | Wah, baiklah ya! | Wah, bagus dong! | 30.0 | 0.88 | Wah, baik-lah ya! | 0.00 | 33 | 1.000 |
| 11 | あっ、あとこないだね、髪型変えたらすぐに気付いてくれてっ・・ | あっあとこないだね、髪型変えたらすぐに気付いてくれてっ．．． | 0.03 | Ah, jika kamu ingin mengubah rambut kamu, kamu akan segera menyadari... | Ah, terus kemarin itu, waktu aku ganti gaya rambut, dia langsung sadar... | 24.3 | 0.62 | Ah, jika kamu ingin mengubah rambut kamu, kamu akan segera menyadari... | 0.00 | 23 | 1.000 |
| 12 | ふふっ、他には？ | ふふっ、他には？ | 0.00 | Fufu, untuk yang lain? | Fufu, terus apa lagi? | 23.2 | 0.87 | Fufu, untuk yang lain? | 0.00 | 27 | 1.000 |
| 13 | *sfx* | あとね | – | Selanjutnya ya |  | – | – | Selan. jut-nya yc | 0.08 | 16 | 0.944 |

## Model load (s)

detector:ctd 0.2, ocr:manga_ocr 2.78, translator:m2m100 0.0, mask_refiner:default 0.0, inpainter:lama_manga 0.0, renderer:default 0.0, judge:48px 0.44, detector:classical 0.0, detector:ctd_blk 0.07, detector:yolo 0.04, ocr:48px 0.37, ocr:manga_ocr_hf 0.77, translator:chain 0.0, translator:local_llm 0.0, translator:nllb 0.0, translator:none 0.0, translator:original 0.0, mask_refiner:bbox 0.0, mask_refiner:dilate 0.0, inpainter:fill 0.0, inpainter:lama 0.0, inpainter:ns 0.0, inpainter:telea 0.0, renderer:basic 0.0, renderer:vertical 0.0


