import { dev } from '$app/env';
import { getRequestEvent } from '$app/server';
import { verifyToken } from '@clerk/backend';
import { error, type RequestEvent } from '@sveltejs/kit';
import { env } from 'cloudflare:workers';
import { MongoClient } from 'mongodb';
import { r2Store, type BlobStore } from './blobs.js';
import { mongoJobStore, type JobDoc, type JobStore } from './jobs.js';
import { mongoQueue, type JobQueue } from './queue.js';

export interface Services {
	jobs: JobStore;
	queue: JobQueue;
	blobs: BlobStore;
}

/** The current request's services. Workers cannot share sockets between requests, so each request
 * opens its own MongoDB connection (on first use); hooks.server.ts closes it. */
export function services(): Services {
	const { locals } = getRequestEvent();
	if (!locals.mongo) {
		if (!env.MONGODB_URI) error(500, 'MONGODB_URI is not set');
		locals.mongo = new MongoClient(env.MONGODB_URI, { maxPoolSize: 1, serverSelectionTimeoutMS: 5000 });
	}
	const col = locals.mongo.db('terjemahin').collection<JobDoc>('jobs');
	return { jobs: mongoJobStore(col), queue: mongoQueue(col), blobs: r2Store(env.FILES) };
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
