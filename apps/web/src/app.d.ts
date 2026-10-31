// See https://svelte.dev/docs/kit/types#app.d.ts
import type { MongoClient } from 'mongodb';

// Bindings, vars and secrets of wrangler.jsonc, read with `import { env } from 'cloudflare:workers'`.
declare global {
	namespace App {
		interface Locals {
			mongo?: MongoClient; // opened by services(), closed by hooks.server.ts
		}
	}
	namespace Cloudflare {
		interface Env {
			FILES: R2Bucket;
			MONGODB_URI?: string;
			PUBLIC_CLERK_PUBLISHABLE_KEY?: string;
			CLERK_JWT_KEY?: string;
			WORKER_TOKEN?: string;
		}
	}
}

export {};
