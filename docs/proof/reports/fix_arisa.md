# terjemahin benchmark

Page 1654x1170; baseline detector=`ctd`, ocr=`manga_ocr`, translator=`m2m100`, mask_refiner=`default`, inpainter=`lama_manga`, renderer=`default`. Each matrix row changes one stage, the rest stay at the baseline.

Columns: detection = GT bubbles matched by OCR text (Hungarian on CER <= 0.5), false positives and duplicate detections; OCR = mean CER over matched bubbles; chrF vs `id_ref` (missing bubbles count as empty), e/j = empty / Japanese outputs; meaning = GT bubbles whose translation has embedding cosine >= 0.8 to its reference (paraphrase-multilingual-MiniLM) and no meaning conflict with it (flipped negation, swapped or added participants, active for passive), and the mean cosine; detection counts a bubble whose box covers less than 75% of its text as partial, not found; residual = text regions the judge detector+OCR still reads inside erased areas / dark and off-tone (ghost) pixel shares left in erased balloon interiors; outside = mean abs diff / changed share of pixels OUTSIDE the mask / changed pixels more than 8 px from it (must be 0: only edge feathering may touch the ring); inside = worst share of a bubble's rendered ink inside its balloon (or padded box) and bubbles whose text was lost; read-back = mean / worst CER (letters only) of the rendered text read back from pixels by an independent OCR (`48px`) against the translation; font = smallest font px; total = pipeline seconds for that config, stages reused from the cache counted at the baseline's warm time.

| run | status | detection | OCR CER | chrF | meaning | residual | outside mask | inside | read-back CER | min font | stage s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | ok | 30/30 fp 0 dup 0 PASS | 0.020 PASS | 0.0 e0 j0 | – – n/a | 0 / 0.0005 / 0.0286 | 0.232 / 0.0147 / 0 FAIL | 1.000 lost 2 | 0.260 / 4.50 | 13 FAIL | – | 43.4 |

## Baseline checks

- **detect** PASS: 30/30 GT bubbles matched, 0 of them partial (box covers < 75% of the text: none), 0 not fully found (need >= 92% fully found), 0 false positives, 0 SFX regions, 0 fragments, 0 duplicates, 31 raw regions.
- **ocr** PASS: mean CER 0.020 (need <= 0.15), 0.007 ignoring punctuation, 26/30 exact.
- **translate** n/a: chrF 0.0 (need >= 25.0), chrF++ 0.0, 0 empty, 0 still Japanese of 30; meaning NOT checked (the GT has no id_ref).
- **inpaint** FAIL: 133104 px erased, 0 residual text regions (0 chars of 256 erased), dark residue 0.0005 (was 0.189 before erasing), ghost 0.0286 (98119 px), outside-mask MAD 0.2321, max 255, changed 0.014682 (all within 8 px of the mask: True).
- **render** FAIL: 30 bubbles, worst inside share 1.0000, 0 ink px outside, 2 lost, layout chars 529/529 (30 layouts), read back from pixels: 541/529 letters, CER mean 0.260 (need <= 0.1), worst 4.500 (need <= 0.34), 4 overflowing, font 13..16 px (min..median, layout).
- **latency** (s, first call, includes lazy model loading): detect 1.95, ocr 2.62, translate 9.32, mask 0.41, inpaint 28.45, render 0.67; total 43.4.

## Baseline per bubble

| # | GT | OCR | CER | translation | reference | chrF | cosine | read back from pixels | read CER | font | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | わ…わたわたし… | わ．．．わたわたし．． | 0.00 | Aku... Aku... |  | 0.0 | – | Aku... Aku... | 0.00 | 20 | 1.000 |
| 1 | あの… | あの．．． | 0.00 | Anu... |  | 0.0 | – | Anu... | 0.00 | 17 | 1.000 |
| 2 | な名前沢渡有紗でです… | な名前沢渡有紗でです．．． | 0.00 | Ya, nama yang ada di sana... |  | 0.0 | – | Ya, nama yang ada di sana... | 0.00 | 18 | 1.000 |
| 3 | えと… | えと．．． | 0.00 | Anu... |  | 0.0 | – | Anu... | 0.00 | 18 | 1.000 |
| 4 | えと…その | えと．．．その | 0.00 | Anu... itu yang |  | 0.0 | – | Anu... itu yang | 0.00 | 13 | 1.000 |
| 5 | うっ | うっ | 0.00 | Dan |  | 0.0 | – | Dan | 0.00 | 17 | 1.000 |
| 6 | ぐすん | ぐすん | 0.00 | Berjalanlah |  | 0.0 | – | Ber-jalan-lah | 0.00 | 13 | 1.000 |
| 7 | … | ．．． | 0.00 | ... |  | 0.0 | – | 이 | 1.00 | 13 | 1.000 |
| 8 | たしか前にもこれと同じようなことが… | たしか前にもこれと同じようなことが．．． | 0.00 | Mungkin sebelum ini juga terjadi hal yang sama... |  | 0.0 | – | Mungkin sebelum ini juga teriadi hal yang | 0.13 | 20 | 1.000 |
| 9 | 小学校の頃クラス替えの自己紹介のときあいつ泣き出して | 小学校の頃クラス替えの自己紹介のときあいつ泣き出して | 0.00 | Ketika dia mengganti kelas, dia menangis |  | 0.0 | – | Vetika dia mengganti kelas, dia menangis | 0.03 | 20 | 1.000 |
| 10 | やっぱりあいつありさだ…！ | やっぱりあいつありさだ．．．！ | 0.00 | Dia pasti ada di sana! |  | 0.0 | – | 7iaっ sii a 'a.i a1a! | 0.47 | 20 | 1.000 |
| 11 | 変わってねぇなまったく… | 変わってねぇなまったく．．． | 0.00 | Tidak berubah sama sekali... |  | 0.0 | – | Tidak berubah sama sekali... | 0.00 | 16 | 1.000 |
| 12 | じゃあいつは……？ | じゃあいつは | 0.25 | Kalau gitu, kapan yang |  | 0.0 | – | Kalau git'' ke yu | 0.44 | 16 | 1.000 |
| 13 | しっかりしろアリサ2っ！ | しっかりしろアリサ２っ！ | 0.00 | Selamat berpuasa, Alicia 2! |  | 0.0 | – | Selamat ber-puasa, Alicia 2! | 0.00 | 13 | 1.000 |
| 14 | ううっ… | ううっ．．． | 0.00 | Ya... |  | 0.0 | – | Ya... | 0.00 | 13 | 1.000 |
| 15 | ではその書類は明日までにな | ではその書類は明日までにな | 0.00 | Dokumen-dokumen tersebut akan disimpan sampai besok |  | 0.0 | – | Dokumen-dokumen tersebut akan disimpan sampai besok | 0.00 | 13 | 1.000 |
| 16 | はーい | はーい | 0.00 | Ya |  | 0.0 | – | Ya | 0.00 | 20 | 1.000 |
| 17 | し失礼します | し失礼します | 0.00 | Aku minta maaf |  | 0.0 | – | Aku minta maaf | 0.00 | 13 | 1.000 |
| 18 | あっ | あっ | 0.00 | Ah |  | 0.0 | – | Ah | 0.00 | 20 | 1.000 |
| 19 | しんちゃん！待っててくれたの？ | しんちゃん！待っててくれたの？ | 0.00 | Dan Shin-chan Apakah kau menunggu aku? |  | 0.0 | – | Dan Shin-chan Apakah kau menunggu aku? | 0.00 | 18 | 1.000 |
| 20 | よっ | よっ | 0.00 | Dan |  | 0.0 | – | Dan | 0.00 | 20 | 1.000 |
| 21 | ちょっといいか？ | ちょっといいか？ | 0.00 | Apakah sedikit lebih baik? |  | 0.0 | – | Apakah sedikit lebih baik? | 0.00 | 15 | 1.000 |
| 22 | え… | え．．． | 0.00 | Eh... |  | 0.0 | – | L Eh... | 0.50 | 17 | 1.000 |
| 23 | キャー♡ | キャー♡ | 0.00 | Karir hati |  | 0.0 | – | Karir hati | 0.00 | 20 | 1.000 |
| 24 | 夕日に染まった廊下で5年ぶりに再会した幼なじみに愛の告白っ！ | 夕日に染まった廊下で５年ぶりに再会した幼なじみに愛の告白っ！ | 0.00 | Suatu hari aku bertemu kembali di koridor yang bercahaya lima tahun yang lalu! |  | 0.0 | – | Suatu hari aku bertemu kembali di koridor yang bercahaya lima tahun yang lalu! | 0.00 | 15 | 1.000 |
| 25 | 最っ高にィどきわくモンのシチュエーションだよっ！ | 最っ高にイどきわくモンのシチュエーションだよっ！ | 0.04 | Ini adalah posisi tertinggi yang paling menakutkan! |  | 0.0 | – | Ini adalah posisi tertinggi yang paling menakutkan! | 0.00 | 16 | 1.000 |
| 26 | でもねでもね | でもねでもわ | 0.17 | Tapi ya ya ya |  | 0.0 | – | Tapi ya ya ya I 00000000000000000000000000000000000 700181146 | 4.50 | 13 | 1.000 |
| 27 | 私の答えはずっと前から決まってたよっ | 私の答えはずっと前から決まってたよっ | 0.00 | Jawabannya sudah lama ditentukan |  | 0.0 | – | b s d tukan | 0.72 | 16 | 1.000 |
| 28 | 私はっしんちゃんのことが… | 私はっしんちゃんのことが．．． | 0.00 | Hal ini dilakukan oleh Hasshin-chan... |  | 0.0 | – | Hal ini dilakukan oleh Hasshin-chan... | 0.00 | 15 | 1.000 |
| 29 | だ大好きだーー | だ大好きだー | 0.14 | Aku mencintaimu |  | 0.0 | – | Aku mencintaimu | 0.00 | 20 | 1.000 |

## Model load (s)

detector:ctd 0.2, ocr:manga_ocr 2.75, translator:m2m100 0.0, mask_refiner:default 0.0, inpainter:lama_manga 0.0, renderer:default 0.0, judge:48px 0.4


