# terjemahin benchmark

Page 1487x6307; baseline detector=`ctd`, ocr=`manga_ocr`, translator=`nllb`, mask_refiner=`default`, inpainter=`lama_manga`, renderer=`default`. Each matrix row changes one stage, the rest stay at the baseline.

Columns: detection = GT bubbles matched by OCR text (Hungarian on CER <= 0.5), false positives and duplicate detections; OCR = mean CER over matched bubbles; chrF vs `id_ref` (missing bubbles count as empty), e/j = empty / Japanese outputs; meaning = GT bubbles whose translation has embedding cosine >= 0.8 to its reference (paraphrase-multilingual-MiniLM) and the mean cosine; residual = text regions the judge detector+OCR still reads inside erased areas / dark and off-tone (ghost) pixel shares left in erased balloon interiors; outside = mean abs diff / changed share of pixels OUTSIDE the mask / changed pixels more than 8 px from it (must be 0: only edge feathering may touch the ring); inside = worst share of a bubble's rendered ink inside its balloon (or padded box) and bubbles whose text was lost; read-back = mean / worst CER (letters only) of the rendered text read back from pixels by an independent OCR (`48px`) against the translation; font = smallest font px; total = pipeline seconds for that config, stages reused from the cache counted at the baseline's warm time.

| run | status | detection | OCR CER | chrF | meaning | residual | outside mask | inside | read-back CER | min font | stage s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | ok | 25/25 fp 0 dup 0 PASS | 0.018 PASS | 38.1 e0 j0 | 16/25 0.84 FAIL | 0 / 0.0000 / 0.0071 | 0.029 / 0.0090 / 0 PASS | 1.000 lost 0 | 0.003 / 0.08 | 18 PASS | – | 50.8 |

## Baseline checks

- **detect** PASS: 25/25 GT bubbles matched (need >= 92%), 0 false positives, 1 SFX regions, 0 fragments, 0 duplicates, 26 raw regions.
- **ocr** PASS: mean CER 0.018 (need <= 0.15), 0.015 ignoring punctuation, 22/25 exact.
- **translate** FAIL: chrF 38.1 (need >= 25.0), chrF++ 37.6, 0 empty, 0 still Japanese of 26; meaning: 16/25 GT bubbles adequate (cosine >= 0.8, need >= 70%), mean cosine 0.843, 21 closest to their own reference; inadequate GT #2, #6, #7, #9, #11, #15, #17, #18, #24.
- **inpaint** PASS: 471042 px erased, 0 residual text regions (0 chars of 263 erased), dark residue 0.0000 (was 0.199 before erasing), ghost 0.0071 (463757 px), outside-mask MAD 0.0288, max 254, changed 0.009020 (all within 8 px of the mask: True).
- **render** PASS: 26 bubbles, worst inside share 1.0000, 0 ink px outside, 0 lost, layout chars 548/548 (26 layouts), read back from pixels: 547/548 letters, CER mean 0.003 (need <= 0.1), worst 0.083 (need <= 0.34), 0 overflowing, font 18..31 px (min..median, layout).
- **latency** (s, first call, includes lazy model loading): detect 2.44, ocr 2.36, translate 11.56, mask 0.22, inpaint 31.63, render 2.62; total 50.8.
- **warm latency** (s, second uncached run): detect 2.26, ocr 2.32, translate 6.78, mask 0.22, inpaint 23.53, render 2.30; total 37.4.

## Baseline per bubble

| # | GT | OCR | CER | translation | reference | chrF | cosine | read back from pixels | read CER | font | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ねえねえ、めいは坂本のことどう思う？ | ねえねえ、めいは坂本のことどう思う？ | 0.00 | Hei, hei, apa yang kau pikirkan tentang Sakamoto? | Hei, hei, Mei, menurutmu Sakamoto itu gimana? | 35.6 | 0.89 | Hei, hei, apa yang kau pikirkan tentang Sakamoto? | 0.00 | 31 | 1.000 |
| 1 | 坂本くん？ | 坂本くん？ | 0.00 | Sakamoto-kun? | Sakamoto-kun? | 100.0 | 0.99 | Sakamoto-kun? | 0.00 | 25 | 1.000 |
| 2 | うーんそうだなぁ・・優しいよね | うーんそうだなあ．．優しいよね | 0.07 | Hmm, ya, baik-baik saja | Hmm, gimana ya... dia baik, kan? | 19.5 | 0.64 | Hmm, ya, baik-baik saja | 0.00 | 31 | 1.000 |
| 3 | えー、それだけー？ | えー、それだけー？ | 0.00 | Eh, itu saja? | Eh, cuma itu? | 22.5 | 0.95 | Eh, itu saja? | 0.00 | 35 | 1.000 |
| 4 | じゃあメシアはどうなの？ | じゃあメシアはどうなの？ | 0.00 | Kalau gitu, bagaimana dengan Mesias? | Kalau gitu, Mesia sendiri gimana? | 52.2 | 0.96 | Kalau gitu, bagaimana dengan Mesias? | 0.00 | 32 | 1.000 |
| 5 | え？ | え？ | 0.00 | Eh? | Eh? | 100.0 | 1.00 | Eh? | 0.00 | 36 | 1.000 |
| 6 | わ、私はー・・・別にそんな・・・ | わ、私はー．．．別にそんな．． | 0.00 | A-aku tidak ingin... | A-aku sih... nggak gimana-gimana... | 18.8 | 0.69 | A-aku tidak ingin... | 0.00 | 36 | 1.000 |
| 7 | 反応がいちいち面白いなーとか | 反応がいちいち面白いなーとか | 0.00 | Reaksi itu sangat menarik, gitu | Reaksinya selalu lucu, gitu deh | 27.3 | 0.67 | Reaksi itu sangat mena-rik, gitu | 0.00 | 26 | 1.000 |
| 8 | うんうん | うんうん | 0.00 | Iya, iya | Iya, iya | 100.0 | 1.00 | Iya, iya | 0.00 | 26 | 1.000 |
| 9 | 店の店員さんとかにも敬語使ってて偉いなーとか | 店の店員さんとかにも敬語使ってて偉いなーとか | 0.00 | Dia juga menggunakan kata-kata pujian untuk pemilik toko, gitu | Dia sopan pakai bahasa halus bahkan ke pelayan toko, keren gitu | 22.0 | 0.63 | Dia juga meng-gunakan kata-kata pujian untuk pemilik toko, gitu | 0.00 | 31 | 1.000 |
| 10 | わあ、いいね！ | わあ、いいね！ | 0.00 | Wah, baiklah! | Wah, bagus dong! | 31.1 | 0.89 | Wah, baik-lah! | 0.00 | 33 | 1.000 |
| 11 | あっ、あとこないだね、髪型変えたらすぐに気付いてくれてっ・・ | あっ、あとこないだね、髪型変えたらすぐに気付いてくれてっ．．． | 0.00 | Ah, saat ini, aku baru tahu kalau rambutku berubah... | Ah, terus kemarin itu, waktu aku ganti gaya rambut, dia langsung sadar... | 21.6 | 0.77 | Ah, saat ini, aku baru tahu kalau rambutku berubah... | 0.00 | 29 | 1.000 |
| 12 | ふふっ、他には？ | ふふっ、他には？ | 0.00 | Fufu, ada yang lain? | Fufu, terus apa lagi? | 22.8 | 0.90 | Fufu, ada yang lain? | 0.00 | 27 | 1.000 |
| 13 | *sfx* | あとね | – | Aku akan pergi |  | – | – | Aku akan perg: | 0.08 | 19 | 1.000 |
| 14 | あの時の約束、覚えてる？ | あの時の約束、覚えて．．．る？ | 0.08 | Apa kau ingat janji itu? | Kamu ingat janji waktu itu? | 53.1 | 0.97 | Apa kau ingat janji itu? | 0.00 | 34 | 1.000 |
| 15 | わかった、行くよ誰かいるの？ | わかった、行くよ誰かいるの？ | 0.00 | Baiklah, kita pergi. Ada siapa? | Baiklah, aku pergi Ada orang di sana? | 36.0 | 0.71 | Baiklah, kita pergi. Ada siapa? | 0.00 | 36 | 1.000 |
| 16 | 一緒に帰ろうよ | 緒に帰ろう | 0.29 | Aku akan kembali ke rumahku | Ayo pulang bareng | 11.2 | 0.72 | Aku akan kembali ke rumahku | 0.00 | 31 | 1.000 |
| 17 | 今日は本当にありがとう | 今日は本当にありがとう | 0.00 | Terima kasih banyak hari ini | Terima kasih banyak untuk hari ini | 77.3 | 0.98 | Terima kasih banyak hari ini | 0.00 | 31 | 1.000 |
| 18 | 私の名前はさくらです | 私の名前はさくらです | 0.00 | Nama aku Saura | Namaku Sakura | 46.6 | 0.65 | Nama aku Saura | 0.00 | 31 | 1.000 |
| 19 | 一緒に帰ろうよ | 一緒に帰ろうよ | 0.00 | Kita akan pulang bersama | Ayo pulang bareng | 35.9 | 0.81 | Kita akan pulang bersama | 0.00 | 22 | 1.000 |
| 20 | 先生に怒られちゃった | 先生に怒られちゃった | 0.00 | Aku marah pada guru | Aku dimarahi guru | 38.0 | 0.83 | Aku marah pada guru | 0.00 | 36 | 1.000 |
| 21 | この本、すごく面白いね | この本、すごく面白いね | 0.00 | Buku ini sangat menarik | Buku ini seru banget ya | 41.1 | 0.87 | Buku ini sangat mena-rik | 0.00 | 27 | 1.000 |
| 22 | 大丈夫？ケガはない？私の名前はさくらです先生に怒られちゃった | 大丈夫？ケガはない？私の名前はさくらです先生に怒られちゃった | 0.00 | Kau baik-baik saja? Kau tidak terluka? Nama aku Sakura. Aku marah pada guru | Kamu nggak apa-apa? Nggak terluka? Namaku Sakura Aku dimarahi guru | 46.1 | 0.94 | Kau baik-baik saja? Kau tidak terluka? Nama aku Sakura. Aku marah pada guru | 0.00 | 36 | 1.000 |
| 23 | 静かにして！ | 静かにして！ | 0.00 | Diamlah! | Diam! | 50.0 | 0.99 | Diam-lah! | 0.00 | 18 | 1.000 |
| 24 | もう行かなきゃ…… | もう行かなきゃ．．． | 0.00 | Jika kau harus pergi lagi... | Aku harus pergi sekarang... | 47.6 | 0.64 | Jika kau harus pergi lagi... | 0.00 | 28 | 1.000 |
| 25 | わかった、行くよ | わかった、行くよ | 0.00 | Baiklah, aku akan pergi | Baiklah, aku pergi | 75.1 | 0.98 | Baiklah, aku akan pergi | 0.00 | 30 | 1.000 |

## Model load (s)

detector:ctd 0.21, ocr:manga_ocr 2.78, translator:nllb 0.0, mask_refiner:default 0.0, inpainter:lama_manga 0.0, renderer:default 0.0, judge:48px 0.46

(contact sheet not copied: single-run bench)
