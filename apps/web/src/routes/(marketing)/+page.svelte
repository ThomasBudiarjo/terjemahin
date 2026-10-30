<script lang="ts">
import Compare from "#lib/components/Compare.svelte";

const github = "https://github.com/ThomasBudiarjo/terjemahin";

const steps = [
  {
    n: "01",
    title: "Unggah halaman",
    body: "Seret satu halaman atau satu chapter sekaligus. PNG, JPG, dan WebP.",
    icon: "upload",
  },
  {
    n: "02",
    title: "Pilih model terjemahan",
    body: "Pakai model bawaan kami, atau sambungkan API key milikmu sendiri.",
    icon: "model",
  },
  {
    n: "03",
    title: "Baca dan unduh",
    body: "Rapikan kalimat yang kurang pas langsung di balonnya, lalu unduh hasilnya.",
    icon: "download",
  },
];

const features = [
  {
    jp: "文脈",
    title: "Paham konteks series",
    body: "Simpan glosarium nama dan istilah serta gaya bahasa. Halaman berikutnya ikut konsisten.",
  },
  {
    jp: "写植",
    title: "Typesetting ikut bentuk balon",
    body: "Ukuran huruf, pemenggalan kata, dan perataan disesuaikan dengan bentuk tiap balon.",
  },
  {
    jp: "修復",
    title: "Bersih tanpa bekas",
    body: "Teks asli dihapus dan latar di baliknya diperbaiki, termasuk screentone.",
  },
  {
    jp: "鍵",
    title: "Model pilihan atau API key sendiri",
    body: "Pilih model AI yang kamu suka. Punya API key? Pakai saja, tanpa biaya tambahan dari kami.",
  },
];
</script>
<svelte:head>
  <title>terjemahin — Terjemahan manga otomatis ke Bahasa Indonesia</title>
  <meta name="description" content="Unggah halaman manga, terjemahannya diketik ulang ke dalam balon. Open source, atau pakai versi cloud tanpa setup." />
</svelte:head>

  <!-- Nav -->
  <header class="sticky top-4 z-50 px-4">
    <nav class="mx-auto flex max-w-5xl items-center justify-between rounded-2xl bg-ink py-2.5 pr-2.5 pl-5 text-white shadow-lg shadow-ink/10">
      <a href="/" class="flex items-center gap-2.5">
        <span class="flex size-7 items-center justify-center rounded-md bg-shu font-jp text-sm font-extrabold">訳</span>
        <span class="font-display text-lg font-bold tracking-tight">terjemahin</span>
      </a>
      <div class="hidden items-center gap-8 text-sm text-white/75 md:flex">
        <a href="#cara-kerja" class="hover:text-white">Cara kerja</a>
        <a href="#fitur" class="hover:text-white">Fitur</a>
        <a href="#harga" class="hover:text-white">Harga</a>
        <a href={github} class="hover:text-white">GitHub</a>
      </div>
      <div class="flex items-center gap-2">
        <a href="/console" class="hidden px-3 py-2 text-sm text-white/75 hover:text-white sm:block">Masuk</a>
        <a href="/console" class="rounded-xl bg-shu px-4 py-2 text-sm font-semibold hover:bg-shu-dark">Coba gratis</a>
      </div>
    </nav>
  </header>

  <main>
    <!-- Hero -->
    <section class="relative overflow-hidden px-4 pt-16 pb-24 sm:pt-24">
      <div class="mx-auto max-w-6xl text-center">
        <p class="mb-6 inline-flex items-center gap-2 rounded-full border border-ink/15 bg-white/60 px-3 py-1 text-sm text-ink-soft">
          <span class="font-jp whitespace-nowrap text-shu">マンガ翻訳</span>
          <span class="h-3 w-px bg-ink/20"></span>
          Jepang ke Indonesia, langsung di balonnya
        </p>
        <h1 class="font-display text-[clamp(3rem,10vw,8.5rem)] leading-[0.9] font-extrabold tracking-[-0.04em]">
          Baca manga<br />
          dalam <span class="text-shu">bahasamu.</span>
        </h1>
      </div>

      <div class="relative mx-auto mt-14 max-w-6xl">
        <div class="bg-grid absolute inset-x-0 -inset-y-10 [mask-image:radial-gradient(ellipse_at_center,black_40%,transparent_75%)]"></div>

        <div class="relative mx-auto grid max-w-5xl items-center gap-10 md:grid-cols-[1fr_minmax(0,440px)_1fr]">
          <div class="hidden text-right md:block">
            <p class="font-jp text-sm text-ink-soft">原文</p>
            <p class="mt-1 text-sm text-ink-soft">Halaman asli,<br />teks Jepang vertikal</p>
            <div class="mt-3 ml-auto h-px w-24 bg-ink/30"></div>
          </div>

          <Compare />

          <div class="hidden md:block">
            <p class="font-jp text-sm text-shu">翻訳</p>
            <p class="mt-1 text-sm text-ink-soft">Hasil terjemahan,<br />diketik ulang di balon yang sama</p>
            <div class="mt-3 h-px w-24 bg-shu/60"></div>
          </div>
        </div>
        <p class="relative mt-6 text-center text-xs text-ink-soft">Geser untuk membandingkan</p>
      </div>

      <div class="relative mx-auto mt-12 max-w-xl text-center">
        <p class="text-lg text-ink-soft">
          Unggah halaman, dan terjemahannya langsung diketik ulang ke dalam balon. Hasilnya rapi dan siap dibaca. Tanpa install, tanpa GPU.
        </p>
        <div class="mt-8 flex flex-wrap justify-center gap-3">
          <a href="/console" class="rounded-xl bg-shu px-6 py-3 font-semibold text-white hover:bg-shu-dark">Coba gratis</a>
          <a href={github} class="rounded-xl border-2 border-ink px-6 py-3 font-semibold hover:bg-ink hover:text-white">Self-host di GitHub</a>
        </div>
      </div>
    </section>

    <!-- How it works -->
    <section id="cara-kerja" class="border-t border-ink/10 px-4 py-24">
      <div class="mx-auto max-w-5xl">
        <p class="font-jp text-sm text-shu">使い方</p>
        <h2 class="mt-2 font-display text-4xl font-bold tracking-tight sm:text-5xl">Tiga langkah, satu chapter.</h2>

        <div class="mt-12 grid gap-5 md:grid-cols-3">
          {#each steps as s}
              <div class="rounded-2xl border border-ink/10 bg-white p-6">
                <div class="flex items-center justify-between">
                  <span class="font-display text-sm font-bold text-shu">{s.n}</span>
                  <svg viewBox="0 0 48 48" class="size-12" fill="none" stroke="#151515" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    {#if s.icon === "upload"}
                        <path d="M12 6h18l8 8v28H12z" />
                        <path d="M30 6v8h8" />
                        <path d="M25 34V20M19 26l6-6 6 6" />
                    {:else if s.icon === "model"}
                        <ellipse cx="22" cy="22" rx="16" ry="13" />
                        <path d="M14 33l-4 8 10-5" />
                        <path d="M38 6v8M34 10h8" stroke="#e5402a" />
                        <path d="M15 20h14M15 25h9" />
                    {:else if s.icon === "download"}
                        <path d="M10 8h28v32H10z" />
                        <path d="M10 24h28M24 8v16" />
                        <ellipse cx="31" cy="15" rx="4.5" ry="3.5" stroke="#e5402a" />
                        <path d="M18 30v6M15 33l3 3 3-3" />
                    {/if}
                  </svg>
                </div>
                <h3 class="mt-6 text-lg font-semibold">{s.title}</h3>
                <p class="mt-2 text-ink-soft">{s.body}</p>
              </div>
          {/each}
        </div>
      </div>
    </section>

    <!-- Features -->
    <section id="fitur" class="border-t border-ink/10 px-4 py-24">
      <div class="mx-auto max-w-5xl">
        <p class="font-jp text-sm text-shu">特徴</p>
        <h2 class="mt-2 max-w-2xl font-display text-4xl font-bold tracking-tight sm:text-5xl">Bukan sekadar tempel teks di atas gambar.</h2>

        <!-- One balloon, three stages -->
        <div class="mt-12 grid items-center gap-4 rounded-2xl border border-ink/10 bg-white p-6 sm:grid-cols-[1fr_auto_1fr_auto_1fr] sm:p-10">
          <figure class="text-center">
            <div class="bg-tone mx-auto flex aspect-[4/5] w-36 items-center justify-center rounded-lg">
              <div class="flex h-36 w-24 items-center justify-center rounded-[50%] border-2 border-ink bg-white">
                <p class="font-jp text-base font-medium" style="writing-mode: vertical-rl;">また明日ね！</p>
              </div>
            </div>
            <figcaption class="mt-4 text-sm"><span class="font-semibold">Baca</span><span class="block text-ink-soft">deteksi balon dan OCR</span></figcaption>
          </figure>
          <span class="text-center text-2xl text-ink/30 max-sm:rotate-90">→</span>
          <figure class="text-center">
            <div class="bg-tone mx-auto flex aspect-[4/5] w-36 items-center justify-center rounded-lg">
              <div class="flex h-36 w-24 items-center justify-center rounded-[50%] border-2 border-ink bg-white">
                <div class="h-16 w-8 rounded border border-dashed border-shu"></div>
              </div>
            </div>
            <figcaption class="mt-4 text-sm"><span class="font-semibold">Bersihkan</span><span class="block text-ink-soft">teks dihapus, latar dirapikan</span></figcaption>
          </figure>
          <span class="text-center text-2xl text-ink/30 max-sm:rotate-90">→</span>
          <figure class="text-center">
            <div class="bg-tone mx-auto flex aspect-[4/5] w-36 items-center justify-center rounded-lg">
              <div class="flex h-36 w-24 items-center justify-center rounded-[50%] border-2 border-ink bg-white px-2">
                <p class="text-xs leading-tight font-semibold uppercase">Sampai besok, ya!</p>
              </div>
            </div>
            <figcaption class="mt-4 text-sm"><span class="font-semibold">Ketik ulang</span><span class="block text-ink-soft">terjemahan pas di balon</span></figcaption>
          </figure>
        </div>

        <div class="mt-5 grid gap-5 sm:grid-cols-2">
          {#each features as f}
              <div class="rounded-2xl border border-ink/10 bg-white p-6">
                <span class="inline-flex size-9 items-center justify-center rounded-md border border-shu/30 font-jp text-sm font-extrabold text-shu">{f.jp}</span>
                <h3 class="mt-5 text-lg font-semibold">{f.title}</h3>
                <p class="mt-2 text-ink-soft">{f.body}</p>
              </div>
          {/each}
        </div>
      </div>
    </section>

    <!-- Pricing -->
    <section id="harga" class="border-t border-ink/10 px-4 py-24">
      <div class="mx-auto max-w-5xl">
        <p class="font-jp text-sm text-shu">料金</p>
        <h2 class="mt-2 font-display text-4xl font-bold tracking-tight sm:text-5xl">Gratis di mesinmu sendiri.<br />Praktis di cloud.</h2>

        <div class="mt-12 grid gap-5 md:grid-cols-2">
          <div class="flex flex-col rounded-2xl border border-ink/10 bg-white p-8">
            <h3 class="text-lg font-semibold">Self-host</h3>
            <p class="mt-4 font-display text-5xl font-bold tracking-tight">Gratis</p>
            <p class="mt-2 text-ink-soft">Open source. Jalan di CPU, semua model offline.</p>
            <ul class="mt-6 space-y-2 text-sm">
              <li>✓ Semua fitur, tanpa batas halaman</li>
              <li>✓ Kamu atur sendiri model dan servernya</li>
              <li>✓ Data tidak keluar dari mesinmu</li>
            </ul>
            <a href={github} class="mt-8 rounded-xl border-2 border-ink px-5 py-3 text-center font-semibold hover:bg-ink hover:text-white">Lihat di GitHub</a>
          </div>

          <div class="relative flex flex-col overflow-hidden rounded-2xl bg-ink p-8 text-white">
            <div class="bg-tone pointer-events-none absolute -top-10 -right-10 size-56 rounded-full opacity-40 invert"></div>
            <h3 class="relative text-lg font-semibold">Cloud</h3>
            <p class="relative mt-4 font-display text-5xl font-bold tracking-tight">Segera</p>
            <p class="relative mt-2 text-white/70">Bayar per halaman. Tanpa setup, tanpa server.</p>
            <ul class="relative mt-6 space-y-2 text-sm text-white/90">
              <li>✓ Unggah satu chapter sekaligus</li>
              <li>✓ Model AI pilihan, atau API key sendiri</li>
              <li>✓ Glosarium dan konteks per series</li>
            </ul>
            <a href="mailto:thomasbudiarjo@gmail.com?subject=Waitlist%20terjemahin" class="relative mt-8 rounded-xl bg-shu px-5 py-3 text-center font-semibold hover:bg-shu-dark">Gabung waitlist</a>
          </div>
        </div>
      </div>
    </section>
  </main>

  <footer class="border-t border-ink/10 px-4 py-10">
    <div class="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-4 text-sm text-ink-soft">
      <div class="flex items-center gap-2">
        <span class="flex size-6 items-center justify-center rounded bg-shu font-jp text-xs font-extrabold text-white">訳</span>
        <span class="font-display font-bold text-ink">terjemahin</span>
        <span>· © 2026</span>
      </div>
      <div class="flex gap-6">
        <a href="#cara-kerja" class="hover:text-ink">Cara kerja</a>
        <a href="#harga" class="hover:text-ink">Harga</a>
        <a href={github} class="hover:text-ink">GitHub</a>
      </div>
    </div>
  </footer>
