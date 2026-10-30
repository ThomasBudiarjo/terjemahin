<script lang="ts">
// A hand-drawn (SVG) manga page. The art is identical in both languages; only
// the balloon text changes, which is exactly what the product does.
let { lang, id }: { lang: "ja" | "id"; id: string } = $props();

// Deterministic jitter so the concentration lines look hand-drawn but stable.
const rnd = (i: number) => {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

// Concentration (focus) lines for panel 3: thin wedges from the border to the centre.
const p3 = { x: 304, y: 294, w: 276, h: 250 };
const c3 = { x: 442, y: 412 };
const wedges: string[] = [];
const n = 90;
for (let i = 0; i < n; i++) {
  const a = (i / n) * Math.PI * 2 + rnd(i) * 0.05;
  const far = 260;
  const ox = c3.x + Math.cos(a) * far;
  const oy = c3.y + Math.sin(a) * far;
  const inner = 0.85 + rnd(i + 99) * 0.55;
  const ix = c3.x + Math.cos(a) * 92 * inner;
  const iy = c3.y + Math.sin(a) * 76 * inner;
  const spread = 0.012 + rnd(i + 7) * 0.02;
  const ox2 = c3.x + Math.cos(a + spread) * far;
  const oy2 = c3.y + Math.sin(a + spread) * far;
  wedges.push(`M${ox.toFixed(1)},${oy.toFixed(1)} L${ix.toFixed(1)},${iy.toFixed(1)} L${ox2.toFixed(1)},${oy2.toFixed(1)}Z`);
}

// Shout balloon (burst) around the centre of panel 3.
const burst: string[] = [];
const spikes = 18;
for (let i = 0; i < spikes * 2; i++) {
  const a = (i / (spikes * 2)) * Math.PI * 2;
  const r = i % 2 === 0 ? 1 : 0.8 + rnd(i + 40) * 0.06;
  burst.push(`${(c3.x + Math.cos(a) * 74 * r).toFixed(1)},${(c3.y + Math.sin(a) * 60 * r).toFixed(1)}`);
}

const lines = {
  ja: ["ここで待ってたの？", "遅いよ、先輩！", "ごめん、電車が遅れてさ…", "もう…"],
  id: ["Kamu nunggu di sini dari tadi?", "Telat banget, Kak!", "Maaf, keretanya telat…", "Dasar…"],
};
const text = $derived(lines[lang]);

// Text boxes in % of the 600x840 page, matching the balloons below.
const boxes = [
  { left: 11.7, top: 6.4, width: 16.7, height: 19.6, size: 3.4 },
  { left: 65.3, top: 43.6, width: 16.7, height: 11, size: 3.6 },
  { left: 10.4, top: 70.6, width: 19.2, height: 14.2, size: 3.1 },
  { left: 81.2, top: 69.4, width: 11, height: 7.6, size: 3.1 },
];
const t = $derived(`tone-${id}`);
</script>
<div class="relative w-full" style="aspect-ratio: 600 / 840; container-type: inline-size;">
  <svg viewBox="0 0 600 840" class="absolute inset-0 h-full w-full" aria-hidden="true">
    <defs>
      <pattern id={`${t}-dots`} width="6" height="6" patternUnits="userSpaceOnUse">
        <circle cx="3" cy="3" r="1.25" fill="#151515" />
      </pattern>
      <pattern id={`${t}-dense`} width="5" height="5" patternUnits="userSpaceOnUse">
        <circle cx="2.5" cy="2.5" r="1.6" fill="#151515" />
      </pattern>
      <linearGradient id={`${t}-fade-down`} x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#fff" />
        <stop offset="1" stop-color="#fff" stop-opacity="0" />
      </linearGradient>
      <mask id={`${t}-sky1`}><rect x="20" y="20" width="560" height="200" fill={`url(#${t}-fade-down)`} /></mask>
      <mask id={`${t}-sky4`}>
        <rect x="20" y="558" width="560" height="220" fill={`url(#${t}-fade-down)`} />
        <circle cx="300" cy="745" r="72" fill="#000" />
      </mask>
      <clipPath id={`${t}-p1`}><rect x="20" y="20" width="560" height="260" /></clipPath>
      <clipPath id={`${t}-p2`}><rect x="20" y="294" width="270" height="250" /></clipPath>
      <clipPath id={`${t}-p3`}><rect x={p3.x} y={p3.y} width={p3.w} height={p3.h} /></clipPath>
      <clipPath id={`${t}-p4`}><rect x="20" y="558" width="560" height="262" /></clipPath>
    </defs>

    <rect width="600" height="840" fill="#fdfcf9" />

    <g fill="none" stroke="#151515" stroke-linecap="round" stroke-linejoin="round">
      <!-- Panel 1: rooftop, water tower, distant city -->
      <g clip-path={`url(#${t}-p1)`}>
        <rect x="20" y="20" width="560" height="200" fill={`url(#${t}-dots)`} mask={`url(#${t}-sky1)`} stroke="none" />
        <g stroke-width="1.6" fill="#fdfcf9">
          <rect x="190" y="168" width="44" height="64" />
          <rect x="232" y="150" width="36" height="82" />
          <rect x="266" y="178" width="58" height="54" />
          <rect x="322" y="140" width="30" height="92" />
          <rect x="350" y="172" width="64" height="60" />
        </g>
        <g stroke-width="1.2">
          <path d="M200 182h8M216 182h8M200 196h8M216 196h8M240 164h6M254 164h6M240 178h6M254 178h6M240 192h6M276 192h10M296 192h10M330 156v8M342 156v8M330 176v8M342 176v8M360 186h10M378 186h10M396 186h10" />
        </g>
        <!-- water tower -->
        <g stroke-width="2">
          <path d="M440 118 v62 a40 9 0 0 0 80 0 v-62" fill="#fdfcf9" />
          <ellipse cx="480" cy="118" rx="40" ry="9" fill="#fdfcf9" />
          <path d="M452 186 l-10 46 M508 186 l10 46 M470 189 v43 M490 189 v43 M446 210 h68" />
          <path d="M448 140 h64 M448 160 h64" stroke-width="1.2" />
        </g>
        <!-- birds -->
        <path d="M300 70 q6 -6 12 0 q6 -6 12 0 M340 52 q5 -5 10 0 q5 -5 10 0 M372 82 q4 -4 8 0 q4 -4 8 0" stroke-width="1.6" />
        <!-- railing in front -->
        <g stroke-width="3">
          <path d="M20 236 H580 M20 252 H580" />
        </g>
        <path d="M40 236v44M70 236v44M100 236v44M130 236v44M160 236v44M190 236v44M220 236v44M250 236v44M280 236v44M310 236v44M340 236v44M370 236v44M400 236v44M430 236v44M460 236v44M490 236v44M520 236v44M550 236v44" stroke-width="2" />
        <!-- balloon 1 -->
        <path d="M150 208 L196 248 L124 224" fill="#fff" stroke-width="2.4" />
        <ellipse cx="120" cy="136" rx="60" ry="94" fill="#fff" stroke-width="2.4" />
        <path d="M128 222 L150 208" stroke="#fff" stroke-width="4" />
      </g>

      <!-- Panel 2: two cans on the rail, SFX -->
      <g clip-path={`url(#${t}-p2)`}>
        <rect x="20" y="294" width="270" height="250" fill={`url(#${t}-dots)`} opacity="0.35" stroke="none" />
        <path d="M30 330 L280 318 M30 352 L280 344" stroke-width="1" opacity="0.5" />
        <g stroke-width="2.2" fill="#fdfcf9">
          <path d="M84 404 v66 a22 6 0 0 0 44 0 v-66" />
          <ellipse cx="106" cy="404" rx="22" ry="6" />
          <rect x="84" y="422" width="44" height="24" fill={`url(#${t}-dense)`} stroke="none" />
          <path d="M84 422 h44 M84 446 h44" stroke-width="1.4" />
          <g transform="rotate(8 170 440)">
            <path d="M150 412 v58 a20 5.5 0 0 0 40 0 v-58" fill="#fdfcf9" />
            <ellipse cx="170" cy="412" rx="20" ry="5.5" />
            <path d="M150 432 h40 M150 438 h40" stroke-width="1.4" />
          </g>
        </g>
        <path d="M20 476 H290 M20 496 H290" stroke-width="3" />
        <path d="M40 496v48M80 496v48M120 496v48M160 496v48M200 496v48M240 496v48M280 496v48" stroke-width="2" />
        <path d="M196 402 l14 -8 M200 418 l16 -2 M192 388 l8 -12" stroke-width="2" />
        <text x="214" y="370" transform="rotate(-12 214 370)" font-family="Noto Sans JP" font-weight="800" font-size="34" fill="#151515" stroke="#fdfcf9" stroke-width="5" paint-order="stroke">カラン</text>
      </g>

      <!-- Panel 3: concentration lines, shout balloon -->
      <g clip-path={`url(#${t}-p3)`}>
        <path d={wedges.join(" ")} fill="#151515" stroke="none" />
        <polygon points={burst.join(" ")} fill="#fff" stroke-width="2.4" />
      </g>

      <!-- Panel 4: sunset, two silhouettes -->
      <g clip-path={`url(#${t}-p4)`}>
        <rect x="20" y="558" width="560" height="220" fill={`url(#${t}-dense)`} mask={`url(#${t}-sky4)`} stroke="none" />
        <circle cx="300" cy="745" r="72" stroke-width="1.6" />
        <path d="M20 762 H580" stroke-width="1.6" />
        <g fill="#151515" stroke="none">
          <circle cx="382" cy="676" r="21" />
          <path d="M350 822 L346 738 Q348 712 372 704 L392 704 Q416 712 418 738 L414 822 Z" />
          <circle cx="462" cy="664" r="22" />
          <path d="M478 656 q24 6 18 40 q-10 -18 -22 -22 z" />
          <path d="M430 822 L426 730 Q428 702 452 694 L472 694 Q496 702 498 730 L494 822 Z" />
        </g>
        <path d="M20 772 H580 M20 788 H580" stroke-width="3" />
        <path d="M40 788v40M70 788v40M100 788v40M130 788v40M160 788v40M190 788v40M220 788v40M250 788v40M280 788v40M310 788v40M340 788v40M370 788v40M400 788v40M430 788v40M460 788v40M490 788v40M520 788v40M550 788v40" stroke-width="2" />
        <!-- balloons -->
        <path d="M168 694 L300 706 L176 676" fill="#fff" stroke-width="2.4" />
        <ellipse cx="124" cy="652" rx="74" ry="66" fill="#fff" stroke-width="2.4" />
        <path d="M168 694 L176 676" stroke="#fff" stroke-width="4" />
        <path d="M506 640 L486 676 L522 646" fill="#fff" stroke-width="2.4" />
        <ellipse cx="520" cy="614" rx="40" ry="38" fill="#fff" stroke-width="2.4" />
        <path d="M506 640 L522 646" stroke="#fff" stroke-width="4" />
      </g>

      <!-- Panel borders -->
      <g stroke-width="4">
        <rect x="20" y="20" width="560" height="260" />
        <rect x="20" y="294" width="270" height="250" />
        <rect x={p3.x} y={p3.y} width={p3.w} height={p3.h} />
        <rect x="20" y="558" width="560" height="262" />
      </g>
    </g>
  </svg>

  {#each boxes as b, i}
    <div
      class="absolute flex items-center justify-center text-center leading-tight text-ink"
      style={`left:${b.left}%;top:${b.top}%;width:${b.width}%;height:${b.height}%;`}
    >
      {#if lang === "ja"}
        <p class="font-jp font-medium" style={`writing-mode:vertical-rl;font-size:${b.size}cqw;line-height:1.35;`}>{text[i]}</p>
      {:else}
        <p class="font-sans font-semibold uppercase" style={`font-size:${b.size * 0.78}cqw;line-height:1.2;`}>{text[i]}</p>
      {/if}
    </div>
  {/each}
</div>
