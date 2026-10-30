import { error, json } from '@sveltejs/kit';
import { publicJob } from '#lib/server/jobs.js';
import { requireUser, services } from '#lib/server/services.js';
import type { RequestHandler } from './$types';

const DEBUG_STAGES = ['detect', 'mask', 'inpainted', 'final'];

// GET /v1/jobs/:id                -> job status
// GET /v1/jobs/:id/image          -> translated page (PNG)
// GET /v1/jobs/:id/result         -> PageResult JSON
// GET /v1/jobs/:id/debug/:stage   -> detect | mask | inpainted | final (debug=true jobs)
export const GET: RequestHandler = async (event) => {
	const userId = await requireUser(event);
	const { jobs, blobs } = services();
	const job = await jobs.get(event.params.id);
	if (!job || job.userId !== userId) error(404, 'job not found');

	const file = event.params.file;
	if (file === '') return json(publicJob(job));

	let key: string;
	if (file === 'image') key = 'output.png';
	else if (file === 'result') key = 'result.json';
	else if (file.startsWith('debug/') && DEBUG_STAGES.includes(file.slice(6))) {
		if (!job.debug) error(404, 'job was not run with debug=true');
		key = `debug/${file.slice(6)}.png`;
	} else error(404, 'not found');

	if (job.status !== 'done') error(409, `job not finished: ${publicJob(job).status}`);
	const blob = await blobs.get(`jobs/${job.id}/${key}`);
	if (!blob) error(410, 'file expired');
	return new Response(blob.body, { headers: { 'content-type': blob.contentType } });
};
