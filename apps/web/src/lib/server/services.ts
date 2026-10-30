import { dev } from '$app/env';
import { verifyToken } from '@clerk/backend';
import { error, type RequestEvent } from '@sveltejs/kit';
import { env } from 'cloudflare:workers';
import { r2Store, type BlobStore } from './blobs.js';
import { d1JobStore, type JobStore } from './jobs.js';
import { cloudflareQueue, d1Queue, type JobQueue } from './queue.js';

export interface Services {
	jobs: JobStore;
	queue: JobQueue;
	blobs: BlobStore;
}

export function services(): Services {
	let queue: JobQueue;
	if (env.QUEUE_DRIVER === 'd1') {
		queue = d1Queue(env.DB);
	} else {
		if (!env.CF_ACCOUNT_ID || !env.CF_QUEUE_ID || !env.CF_QUEUES_API_TOKEN) {
			error(500, 'QUEUE_DRIVER=cloudflare needs CF_ACCOUNT_ID, CF_QUEUE_ID and CF_QUEUES_API_TOKEN');
		}
		queue = cloudflareQueue(env.JOBS, {
			accountId: env.CF_ACCOUNT_ID,
			queueId: env.CF_QUEUE_ID,
			token: env.CF_QUEUES_API_TOKEN
		});
	}
	return { jobs: d1JobStore(env.DB), queue, blobs: r2Store(env.FILES) };
}

/** The signed-in Clerk user, from a session token in `Authorization: Bearer` (the console
 * sends `await clerk.session.getToken()`). It is verified offline against CLERK_JWT_KEY, the
 * instance's public key, so the app holds no Clerk secret. The `__session` cookie alone is not
 * accepted: a cross-site form could send it (see `csrf` in vite.config.ts). Without
 * CLERK_JWT_KEY, `vite dev` (never a build) treats every request as user "dev". */
export async function requireUser(event: RequestEvent): Promise<string> {
	if (!env.CLERK_JWT_KEY) {
		if (dev) return 'dev';
		error(500, 'CLERK_JWT_KEY is not set');
	}
	const auth = event.request.headers.get('authorization') ?? '';
	if (!auth.startsWith('Bearer ')) {
		error(401, 'sign in required (Authorization: Bearer <Clerk session token>)');
	}
	let userId: string | undefined;
	try {
		const claims = await verifyToken(auth.slice(7), {
			jwtKey: env.CLERK_JWT_KEY,
			authorizedParties: [event.url.origin]
		});
		userId = (claims as { sub?: string }).sub;
	} catch {
		// throws when expired, badly signed, or minted for another site (azp)
	}
	if (!userId) error(401, 'sign in required');
	return userId;
}

/** The Python worker, by `Authorization: Bearer $WORKER_TOKEN`. */
export function requireWorker(event: RequestEvent): void {
	const token = env.WORKER_TOKEN;
	if (!token) error(500, 'WORKER_TOKEN is not set');
	const got = event.request.headers.get('authorization') ?? '';
	if (!timingSafeEqual(got, `Bearer ${token}`)) error(401, 'bad worker token');
}

function timingSafeEqual(a: string, b: string): boolean {
	const x = new TextEncoder().encode(a);
	const y = new TextEncoder().encode(b);
	let diff = x.length ^ y.length;
	for (let i = 0; i < Math.max(x.length, y.length); i++) diff |= (x[i] ?? 0) ^ (y[i] ?? 0);
	return diff === 0;
}
