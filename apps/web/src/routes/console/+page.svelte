<script lang="ts">
	import { Clerk } from '@clerk/clerk-js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	let phase = $state<'loading' | 'signed-out' | 'signed-in' | 'error'>('loading');
	let message = $state('');
	let signInEl = $state<HTMLDivElement>();
	let userEl = $state<HTMLDivElement>();
	let clerk = $state.raw<Clerk>();

	// Clerk JS quickstart: https://clerk.com/docs/js-frontend/getting-started/quickstart
	async function start(key: string): Promise<Clerk> {
		const domain = atob(key.split('_')[2]).slice(0, -1); // Frontend API host, from the key
		await new Promise((resolve, reject) => {
			const s = document.createElement('script');
			s.src = `https://${domain}/npm/@clerk/ui@1/dist/ui.browser.js`;
			s.async = true;
			s.crossOrigin = 'anonymous';
			s.onload = resolve;
			s.onerror = () => reject(new Error('Gagal memuat Clerk UI'));
			document.head.appendChild(s);
		});
		const clerk = new Clerk(key);
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		await clerk.load({ ui: { ClerkUI: (window as any).__internal_ClerkUICtor } });
		return clerk;
	}

	$effect(() => {
		if (!data.clerkKey) {
			phase = 'error';
			message = 'PUBLIC_CLERK_PUBLISHABLE_KEY belum diset.';
			return;
		}
		start(data.clerkKey).then(
			(c) => {
				clerk = c;
				const sync = () => (phase = c.isSignedIn ? 'signed-in' : 'signed-out');
				sync();
				c.addListener(sync);
			},
			(e: Error) => {
				phase = 'error';
				message = e.message;
			}
		);
	});

	// mount Clerk's components once their containers exist
	$effect(() => {
		if (phase === 'signed-out' && signInEl) clerk?.mountSignIn(signInEl);
		if (phase === 'signed-in' && userEl) clerk?.mountUserButton(userEl);
	});
</script>

<svelte:head>
	<title>Console · terjemahin</title>
	<meta name="robots" content="noindex" />
</svelte:head>

<div class="min-h-dvh">
	<header class="flex items-center justify-between border-b border-ink/10 px-6 py-4">
		<a href="/" class="flex items-center gap-2.5">
			<span class="flex size-7 items-center justify-center rounded-md bg-shu font-jp text-sm font-extrabold text-white">訳</span>
			<span class="font-display text-lg font-bold tracking-tight">terjemahin</span>
		</a>
		<div bind:this={userEl}></div>
	</header>

	<main class="mx-auto max-w-3xl px-6 py-16">
		{#if phase === 'loading'}
			<p class="text-ink-soft">Memuat…</p>
		{:else if phase === 'error'}
			<p class="text-shu">{message}</p>
		{:else if phase === 'signed-out'}
			<div class="flex justify-center" bind:this={signInEl}></div>
		{:else}
			<p class="font-jp text-sm text-shu">コンソール</p>
			<h1 class="mt-2 font-display text-4xl font-bold tracking-tight">Console segera hadir.</h1>
			<p class="mt-3 text-ink-soft">Unggah halaman dan pilih model terjemahan akan ada di sini.</p>
		{/if}
	</main>
</div>
