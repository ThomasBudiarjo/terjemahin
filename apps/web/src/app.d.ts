// See https://svelte.dev/docs/kit/types#app.d.ts
// Bindings, vars and secrets of wrangler.jsonc, read with `import { env } from 'cloudflare:workers'`.
declare global {
	namespace Cloudflare {
		interface Env {
			DB: D1Database;
			FILES: R2Bucket;
			JOBS: Queue;
			QUEUE_DRIVER: 'cloudflare' | 'd1';
			PUBLIC_CLERK_PUBLISHABLE_KEY?: string;
			CLERK_JWT_KEY?: string;
			WORKER_TOKEN?: string;
			CF_ACCOUNT_ID?: string;
			CF_QUEUE_ID?: string;
			CF_QUEUES_API_TOKEN?: string;
		}
	}
}

export {};
