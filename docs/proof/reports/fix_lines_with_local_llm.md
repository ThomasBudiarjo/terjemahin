# Line translation check (eval/fixtures/lines_ja_id.tsv)

adequate = cosine >= 0.8 to the reference and no meaning conflict (negation / participants / roles / voice).

| translator | adequate | conflicts | chrF | seconds |
|---|---|---|---|---|
| nllb | 19/36 | 5 | 39.4 | 13.6 |
| m2m100 | 23/36 | 4 | 39.1 | 10.1 |
| local_llm | 24/36 | 1 | 39.6 | 118.8 |

| # | ja | reference | nllb | m2m100 | local_llm |
|---|---|---|---|---|---|
| 0 | お前、何してんだよ！ | Kamu lagi ngapain sih! | Apa yang kau lakukan! (ok 0.948) | Apa yang kau lakukan! (ok 0.947) | Hei, apa yang kau lakukan! (ok 0.9) |
| 1 | 別に、あんたのためじゃないんだからね！ | Ini bukan buat kamu, tahu! | Itu bukan untukmu! (ok 0.955) | Karena itu bukan untukmu! (ok 0.957) | Ini tidak untuk kepentinganmu pula! (ok 0.932) |
| 2 | 遅刻だ遅刻！ | Telat, telat! | Terlambat. Terlambat! (NO 0.561) | Terlambat dan terlambat! (NO 0.622) | Terlambat lagi! (NO 0.489) |
| 3 | このままじゃ負けちまう… | Kalau begini terus, kita bakal kalah... | Kita akan kalah jika kita terus seperti ini... (ok 0.971) | Dengan demikian kita akan kalah... (ok 0.952) | Kita akan kalah... (ok 0.933) |
| 4 | 絶対に許さない！ | Nggak akan pernah kumaafkan! | Aku tak akan pernah memaafkanmu! (ok 0.92) | Aku tidak akan pernah memaafkan! (ok 0.986) | Aku tidak akan memaafkanmu! (ok 0.864) |
| 5 | 腹減ったー | Aduh, lapar... | Aku lapar (ok 0.881) | Aku kelaparan (ok 0.812) | Aku kelaparan (NO 0.798) |
| 6 | 待たせたな | Maaf ya, aku bikin kamu nunggu lama. | Kau harus menunggu (NO 0.393) | Kau membiarkan aku menunggu (NO 0.508 [roles]) | Aku membuatmu menunggu (NO 0.539) |
| 7 | 嘘でしょ？ | Bercanda, kan? | Kau berbohong, kan? (NO 0.646) | Kau berbohong kan? (NO 0.646) | Kau berbohong, kan? (NO 0.647) |
| 8 | 先輩、好きです！ | Senpai, aku suka kamu! | Aku menyukainya! (NO 0.692 [participants]) | Tuanku, aku suka denganmu! (ok 0.895) | Senior, aku menyukaimu! (NO 0.692) |
| 9 | 何度言ったらわかるんだ | Harus berapa kali kubilang baru kamu ngerti? | Aku tahu berapa kali aku bilang (NO 0.634) | Berapa kali aku berkata, aku tahu (NO 0.795) | Aku sudah bilang ini berkali-kali! (NO 0.634) |
| 10 | 今日の放課後、屋上に来て | Nanti pulang sekolah, datang ke atap ya. | Setelah sekolah hari ini, aku datang ke atap (NO 0.756) | Setelah kuliah, datanglah ke langit-langit (NO 0.648) | Datang ke atap setelah sekolah hari ini, oke? (ok 0.844) |
| 11 | 俺はもう逃げない | Aku nggak akan lari lagi. | Aku tak bisa melarikan diri lagi (ok 0.805) | Aku tidak akan melarikan diri lagi (ok 0.87) | Aku tidak akan lari lagi (ok 0.987) |
| 12 | あいつ、いつも一人で弁当食べてるよな | Dia selalu makan bekal sendirian, ya. | Dia selalu makan siang sendirian (ok 0.809) | Dia selalu makan sendirian (ok 0.931) | Dia selalu makan siang sendirian, kau tahu (ok 0.804) |
| 13 | まさか、お前が犯人だったとは… | Jadi kamu pelakunya... Sulit dipercaya. | Tidak, kau adalah pelakunya... (NO 0.714 [negation]) | Tidak ada yang mengatakan bahwa kamu adalah seorang pembunuh... (NO 0.35 [negation]) | Kau pasti salah, kan... (NO 0.456) |
| 14 | 手伝ってくれてありがとう、助かったよ | Makasih sudah membantu, sangat menolong. | Terima kasih atas bantuanmu (NO 0.749) | Terima kasih telah membantu aku, aku selamat (NO 0.657) | Terima kasih telah membantuku, itu sangat membantu (ok 0.872) |
| 15 | ちょっと、どこ見てんのよ！ | Hei, kamu lihat ke mana sih! | Tunggu, aku tak melihatmu! (NO 0.679 [negation/participants]) | Tunggu, di manakah kamu melihatnya! (ok 0.874) | Tunggu, apa yang kau lihat! (ok 0.844) |
| 16 | それって、デートってこと？ | Maksudnya, itu kencan? | Apa maksudmu kencan? (ok 0.857) | Apakah itu sebuah kencan? (ok 0.864) | Itu berarti kau akan berkencan, kan? (NO 0.78) |
| 17 | 心配すんな、俺に任せとけ | Jangan khawatir, serahkan padaku. | Jangan khawatir, aku yang akan mengurusnya (NO 0.789) | Jangan khawatir, biarkan aku pergi (ok 0.895) | Jangan khawatir, aku akan mengurusnya (NO 0.773) |
| 18 | 宿題やった？ | Kamu sudah mengerjakan PR? | Kau sudah menyelesaikan tugasmu? (NO 0.551) | Apakah kamu melakukan tugas rumah? (NO 0.644) | Apa kau sudah mengerjakan tugasmu? (ok 0.817) |
| 19 | 田中くんって、意外と優しいんだね | Tanaka-kun ternyata baik juga, ya. | Tanaka-kun, yang sangat lembut (NO 0.738) | Tanaka-kun memang sangat baik (ok 0.908) | Tanaka-kun sangat baik (ok 0.926) |
| 20 | うるさいなぁ、もう寝かせてよ | Berisik ah, biarin aku tidur dong. | Aku ingin kau tidur (NO 0.855 [participants]) | Suara, biarlah aku tidur (ok 0.93) | Diamlah, aku mencoba tidur (ok 0.809) |
| 21 | 行ってきます！ | Aku berangkat dulu! | Aku akan pergi! (ok 0.943) | Aku akan pergi! (ok 0.951) | Aku akan pergi sekarang! (ok 0.916) |
| 22 | 君の笑顔が見たいんだ | Aku ingin melihat senyummu. | Aku ingin melihat senyummu (ok 0.995) | Aku ingin melihat senyummu (ok 0.994) | Aku ingin melihat senyummu (ok 0.994) |
| 23 | 一体どういうつもりだ？ | Sebenarnya apa maksudmu? | Apa maksudmu? (ok 0.961) | Apa yang akan kamu maksudkan? (ok 0.936) | Apa yang kau rencanakan? (NO 0.398) |
| 24 | 先生に怒られた | Aku dimarahi guru. | Dia marah pada guru (NO 0.681 [participants/voice]) | Aku marah dengan guru (NO 0.862 [voice]) | Aku punya masalah dengan guru (ok 0.886) |
| 25 | あいつに嫌われてるかも | Mungkin dia membenciku. | Mungkin dia membencinya (NO 0.775) | Mungkin dia membencinya (NO 0.763) | Aku mungkin tidak disukai olehnya (NO 0.731 [negation]) |
| 26 | 一緒に帰ろうよ | Ayo pulang bareng. | Kita akan pulang bersama (ok 0.846) | Aku akan pulang bersama-sama (NO 0.792) | Mari pulang bersama (ok 0.906) |
| 27 | わかった、行くよ | Oke, aku pergi. | Baiklah, aku akan pergi (ok 0.969) | Aku tahu, aku akan pergi (ok 0.812) | Baiklah, aku akan pergi (ok 0.967) |
| 28 | もう二度と会えないの？ | Kita nggak bisa ketemu lagi? | Kau tak akan pernah bertemu lagi? (NO 0.748) | Tidak akan pernah bertemu lagi? (NO 0.786) | Kau tidak akan bertemu lagi? (ok 0.819) |
| 29 | そんなこと言ってないよ | Aku nggak bilang begitu. | Aku tidak mengatakan itu (ok 0.965) | Aku tidak mengatakan hal itu (ok 0.974) | Aku tidak mengatakan itu (ok 0.954) |
| 30 | 彼に助けられた | Aku diselamatkan olehnya. | Dia yang membantuku (NO 0.568) | Dia telah diselamatkan (NO 0.873 [participants]) | Aku diselamatkan olehnya (ok 0.994) |
| 31 | 泣かないで | Jangan menangis. | Jangan menangis (ok 0.98) | Jangan menangis (ok 0.972) | Jangan menangis (ok 0.975) |
| 32 | 私の名前はさくらです | Namaku Sakura. | Nama aku Sakura (ok 0.991) | Nama aku Sakura (ok 0.991) | Namaku Sakura (ok 0.99) |
| 33 | ゆいちゃん、おはよう！ | Pagi, Yui-chan! | Yui-chan Selamat pagi! (ok 0.945) | Yui-chan Selamat pagi! (ok 0.948) | Selamat pagi, Yui-chan! Selamat pagi! (ok 0.942) |
| 34 | けんたくんはどこ？ | Kenta-kun di mana? | Dimana Kenta-kun? (ok 0.838) | Di manakah Kenta-kun? (ok 0.874) | Dimana Kenta-kun? (ok 0.883) |
| 35 | めいは坂本のことどう思う？ | Mei, menurutmu Sakamoto itu gimana? | Apa yang kau pikirkan tentang Sakamoto? (ok 0.86) | Apa pendapat kamu tentang Sakamoto? (ok 0.865) | Bu. Sakamoto, apa pendapatmu tentang dia? (NO 0.791) |
