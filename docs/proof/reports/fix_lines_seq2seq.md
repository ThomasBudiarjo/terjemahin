# Line translation check (eval/fixtures/lines_ja_id.tsv)

adequate = cosine >= 0.8 to the reference and no meaning conflict (negation / participants / roles / voice).

| translator | adequate | conflicts | chrF | seconds |
|---|---|---|---|---|
| nllb | 18/36 | 4 | 40.9 | 14.3 |
| m2m100 | 23/36 | 4 | 40.0 | 8.2 |
| chain | 24/36 | 6 | 41.4 | 15.4 |

| # | ja | reference | nllb | m2m100 | chain |
|---|---|---|---|---|---|
| 0 | お前、何してんだよ！ | Kamu lagi ngapain sih! | Apa yang kau lakukan! (ok 0.95) | Apa yang kau lakukan! (ok 0.948) | Apa yang kau lakukan! (ok 0.95) |
| 1 | 別に、あんたのためじゃないんだからね！ | Ini bukan buat kamu, tahu! | Itu bukan untukmu! (ok 0.96) | Karena itu bukan untukmu! (ok 0.958) | Bukan karena itu bukan untukmu! (ok 0.935) |
| 2 | 遅刻だ遅刻！ | Telat, telat! | Terlambat. Terlambat! (NO 0.559) | Terlambat dan terlambat! (NO 0.606) | Sudah terlambat, sudah terlambat! (NO 0.565) |
| 3 | このままじゃ負けちまう… | Kalau begini terus, kita bakal kalah... | Kita akan kalah jika kita terus seperti ini... (ok 0.969) | Dengan demikian kita akan kalah... (ok 0.951) | Jadi itu akan hilang... (NO 0.652) |
| 4 | 絶対に許さない！ | Nggak akan pernah kumaafkan! | Aku tak akan pernah memaafkanmu! (ok 0.923) | Aku tidak akan pernah memaafkan! (ok 0.987) | Aku tidak akan pernah memaafkan! (ok 0.987) |
| 5 | 腹減ったー | Aduh, lapar... | Aku lapar (ok 0.877) | Aku kelaparan (NO 0.796) | Aku lapar (ok 0.884) |
| 6 | 待たせたな | Maaf ya, aku bikin kamu nunggu lama. | Kau harus menunggu (NO 0.377) | Kau membiarkan aku menunggu (NO 0.506 [roles]) | Biarkan aku menunggu (NO 0.375) |
| 7 | 嘘でしょ？ | Bercanda, kan? | Kau berbohong, kan? (NO 0.651) | Kau berbohong kan? (NO 0.649) | Itu bohong, kan? (NO 0.682) |
| 8 | 先輩、好きです！ | Senpai, aku suka kamu! | Senpai, aku suka! (ok 0.849) | Senpai, aku suka! (ok 0.854) | Senpai, aku menyukainya! (NO 0.836 [participants]) |
| 9 | 何度言ったらわかるんだ | Harus berapa kali kubilang baru kamu ngerti? | Aku tahu berapa kali aku bilang (NO 0.643) | Berapa kali aku berkata, aku tahu (NO 0.79) | Berapa kali kamu mengatakan, kamu tahu (NO 0.781) |
| 10 | 今日の放課後、屋上に来て | Nanti pulang sekolah, datang ke atap ya. | Setelah sekolah hari ini, aku datang ke atap (NO 0.76) | Setelah kuliah, datanglah ke langit-langit (NO 0.651) | Setelah sekolah, datang ke atap (ok 0.831) |
| 11 | 俺はもう逃げない | Aku nggak akan lari lagi. | Aku tak bisa melarikan diri lagi (NO 0.797) | Aku tidak akan melarikan diri lagi (ok 0.87) | Aku tidak akan melarikan diri lagi (ok 0.875) |
| 12 | あいつ、いつも一人で弁当食べてるよな | Dia selalu makan bekal sendirian, ya. | Dia selalu makan siang sendirian (NO 0.793) | Dia selalu makan sendirian (ok 0.925) | Dia selalu makan sendirian (ok 0.933) |
| 13 | まさか、お前が犯人だったとは… | Jadi kamu pelakunya... Sulit dipercaya. | Tidak, kau adalah pelakunya... (NO 0.712 [negation]) | Tidak ada yang mengatakan bahwa kamu adalah seorang pembunuh... (NO 0.334 [negation]) | Bukan kau pembunuhnya... (NO 0.414 [negation]) |
| 14 | 手伝ってくれてありがとう、助かったよ | Makasih sudah membantu, sangat menolong. | Terima kasih atas bantuanmu (NO 0.752) | Terima kasih telah membantu aku, aku selamat (NO 0.656) | Terima kasih telah membantuku, aku diselamatkan (NO 0.618) |
| 15 | ちょっと、どこ見てんのよ！ | Hei, kamu lihat ke mana sih! | Tunggu, aku tak melihatmu! (NO 0.679 [negation/participants]) | Tunggu, di manakah kamu melihatnya! (ok 0.866) | Tunggu, dimana kau melihatnya! (ok 0.871) |
| 16 | それって、デートってこと？ | Maksudnya, itu kencan? | Apa maksudmu kencan? (ok 0.855) | Apakah itu sebuah kencan? (ok 0.872) | Apa itu kencan? (ok 0.91) |
| 17 | 心配すんな、俺に任せとけ | Jangan khawatir, serahkan padaku. | Jangan khawatir, aku yang akan mengurusnya (NO 0.786) | Jangan khawatir, biarkan aku pergi (ok 0.884) | Jangan khawatir, biarkan aku (ok 0.946) |
| 18 | 宿題やった？ | Kamu sudah mengerjakan PR? | Kau sudah menyelesaikan tugasmu? (NO 0.559) | Apakah kamu melakukan tugas rumah? (NO 0.658) | Apa kau sudah mengerjakan pekerjaan rumah? (ok 0.986) |
| 19 | 田中くんって、意外と優しいんだね | Tanaka-kun ternyata baik juga, ya. | Tanaka-kun, yang sangat lembut (NO 0.75) | Tanaka-kun memang sangat baik (ok 0.905) | Tanaka-kun sangat baik (ok 0.924) |
| 20 | うるさいなぁ、もう寝かせてよ | Berisik ah, biarin aku tidur dong. | Aku ingin kau tidur (NO 0.855 [participants]) | Suara, biarlah aku tidur (ok 0.918) | Suara, biarkan aku tidur (ok 0.873) |
| 21 | 行ってきます！ | Aku berangkat dulu! | Aku akan pergi! (ok 0.948) | Aku akan pergi! (ok 0.948) | Aku akan pergi! (ok 0.944) |
| 22 | 君の笑顔が見たいんだ | Aku ingin melihat senyummu. | Aku ingin melihat senyummu (ok 0.995) | Aku ingin melihat senyummu (ok 0.995) | Aku ingin melihat senyummu (ok 0.994) |
| 23 | 一体どういうつもりだ？ | Sebenarnya apa maksudmu? | Apa maksudmu? (ok 0.959) | Apa yang akan kamu maksudkan? (ok 0.934) | Apa maksudmu? (ok 0.96) |
| 24 | 先生に怒られた | Aku dimarahi guru. | Dia marah pada guru (NO 0.681 [participants/voice]) | Aku marah dengan guru (NO 0.859 [voice]) | Aku marah pada guru (NO 0.855 [voice]) |
| 25 | あいつに嫌われてるかも | Mungkin dia membenciku. | Mungkin dia membencinya (NO 0.765) | Mungkin dia membencinya (NO 0.767) | Dia mungkin membencimu (NO 0.768 [participants]) |
| 26 | 一緒に帰ろうよ | Ayo pulang bareng. | Kita akan pulang bersama (ok 0.833) | Aku akan pulang bersama-sama (ok 0.808) | Pergilah pulang bersama kami (ok 0.902) |
| 27 | わかった、行くよ | Oke, aku pergi. | Baiklah, aku akan pergi (ok 0.972) | Aku tahu, aku akan pergi (ok 0.812) | Aku mengerti, aku pergi (ok 0.926) |
| 28 | もう二度と会えないの？ | Kita nggak bisa ketemu lagi? | Kau tak akan pernah bertemu lagi? (NO 0.756) | Tidak akan pernah bertemu lagi? (NO 0.781) | Kau tak bisa melihatku lagi? (NO 0.764 [roles]) |
| 29 | そんなこと言ってないよ | Aku nggak bilang begitu. | Aku tidak mengatakan itu (ok 0.971) | Aku tidak mengatakan hal itu (ok 0.975) | Aku tidak mengatakan itu (ok 0.963) |
| 30 | 彼に助けられた | Aku diselamatkan olehnya. | Dia yang membantuku (NO 0.568) | Dia telah diselamatkan (NO 0.868 [participants]) | Dia dibantu (NO 0.518 [participants]) |
| 31 | 泣かないで | Jangan menangis. | Jangan menangis (ok 0.977) | Jangan menangis (ok 0.973) | Jangan menangis (ok 0.975) |
| 32 | 私の名前はさくらです | Namaku Sakura. | Namaku Sakura (ok 0.988) | Namaku Sakura (ok 0.991) | Namaku Sakura (ok 0.99) |
| 33 | ゆいちゃん、おはよう！ | Pagi, Yui-chan! | Yui-chan, Selamat pagi! (ok 0.965) | Yui-chan, Selamat pagi! (ok 0.966) | Yui-chan, Selamat pagi! (ok 0.966) |
| 34 | けんたくんはどこ？ | Kenta-kun di mana? | Dimana Kenta-kun? (ok 0.888) | Di manakah Kenta-kun? (ok 0.872) | Dimana Kenta-kun? (ok 0.842) |
| 35 | めいは坂本のことどう思う？ | Mei, menurutmu Sakamoto itu gimana? | Apa yang kau pikirkan tentang Sakamoto? (ok 0.867) | Apa pendapat kamu tentang Sakamoto? (ok 0.87) | Apa pendapatmu tentang Sakamoto? (ok 0.875) |
