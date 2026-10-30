import adapter from '@sveltejs/adapter-cloudflare';
import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		tailwindcss(),
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			adapter: adapter(),
			// The only POST endpoints are APIs for scripts and the worker (multipart, no Origin
			// header). They need a bearer token, which a cross-site form cannot send, so the
			// form-origin check is off; requireUser ignores the session cookie for that reason.
			csrf: { trustedOrigins: ['*'] }
		})
	]
});
