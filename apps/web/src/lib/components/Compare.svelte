<script lang="ts">
	import MangaPage from './MangaPage.svelte';

	// Without JavaScript (the prerendered HTML) the slider rests in the middle.
	let pos = $state(50);
</script>

<div
	class="relative select-none overflow-hidden rounded-sm border-2 border-ink bg-white shadow-[8px_8px_0_0_var(--color-ink)]"
	style:--pos="{pos}%"
>
	<MangaPage lang="ja" id="ja" />
	<div class="absolute inset-0" style="clip-path: inset(0 0 0 var(--pos));">
		<MangaPage lang="id" id="id" />
	</div>

	<div class="pointer-events-none absolute inset-y-0 w-0.5 bg-shu" style="left: var(--pos);">
		<div
			class="absolute top-1/2 left-1/2 flex size-10 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 border-ink bg-white text-ink"
		>
			<svg viewBox="0 0 24 24" class="size-5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l-6 6 6 6M15 6l6 6-6 6" /></svg>
		</div>
	</div>

	<span class="pointer-events-none absolute top-3 left-3 rounded-full bg-ink px-2.5 py-1 font-jp text-xs font-medium text-white">日本語</span>
	<span class="pointer-events-none absolute top-3 right-3 rounded-full bg-shu px-2.5 py-1 text-xs font-semibold text-white">Indonesia</span>

	<input
		type="range"
		min="0"
		max="100"
		bind:value={pos}
		aria-label="Geser untuk membandingkan halaman asli dan terjemahan"
		class="absolute inset-0 h-full w-full cursor-ew-resize opacity-0"
	/>
</div>
