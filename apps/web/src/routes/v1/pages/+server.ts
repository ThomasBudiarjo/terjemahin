import { error, json } from '@sveltejs/kit';
import { waitUntil } from 'cloudflare:workers';
import { publicJob, RETENTION_MS } from '#lib/server/jobs.js';
import { requireUser, services } from '#lib/server/services.js';
import { checkContext, checkPipeline, parseBool } from '#lib/server/validate.js';
import type { RequestHandler } from './$types';

const MAX_UPLOAD = 25 << 20;
const TYPES: Record<string, string> = {
	png: 'image/png',
	jpg: 'image/jpeg',
	jpeg: 'image/jpeg',
	webp: 'image/webp'
};

// multipart: file=<image>, context=<PageContext JSON>, pipeline=<PipelineConfig JSON>,
// debug=<bool: keep per-stage images>; all but file are optional.
export const POST: RequestHandler = async (event) => {
	const userId = await requireUser(event);
	const size = Number(event.request.headers.get('content-length') ?? 0);
	if (size > MAX_UPLOAD) error(413, 'upload larger than 25 MB');

	let form: FormData;
	try {
		form = await event.request.formData();
	} catch {
		error(400, 'invalid multipart form');
	}
	const file = form.get('file');
	if (!(file instanceof File)) error(400, 'missing file');
	if (file.size > MAX_UPLOAD) error(413, 'upload larger than 25 MB');
	const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
	if (!TYPES[ext]) error(400, 'unsupported image type');

	let context: string | null = null;
	let pipeline: string | null = null;
	let debug: boolean;
	try {
		const c = String(form.get('context') ?? '').trim();
		const p = String(form.get('pipeline') ?? '').trim();
		context = c ? checkContext(c) : null;
		pipeline = p ? checkPipeline(p) : null;
		debug = parseBool(form.get('debug') as string | null, 'debug');
	} catch (e) {
		error(400, (e as Error).message);
	}

	const { jobs, queue, blobs } = services();
	const id = crypto.randomUUID().replaceAll('-', '');
	const inputKey = `jobs/${id}/input.${ext}`;
	await blobs.put(inputKey, await file.arrayBuffer(), TYPES[ext]);

	const now = Date.now();
	const job = {
		id,
		userId,
		status: 'queued' as const,
		error: null,
		inputKey,
		context,
		pipeline,
		debug,
		bubbles: null,
		createdAt: now,
		updatedAt: now
	};
	// stored before it is queued, so a worker that claims it at once finds the row
	await jobs.create(job);
	try {
		await queue.enqueue(id);
	} catch (e) {
		console.error('enqueue', e);
		await jobs.finish(id, { status: 'failed', error: 'queue unavailable' });
		error(503, 'queue unavailable');
	}
	waitUntil(jobs.purge(now - RETENTION_MS));
	return json(publicJob(job), { status: 202 });
};
