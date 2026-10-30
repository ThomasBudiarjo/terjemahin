import { error } from '@sveltejs/kit';
import { requireWorker, services } from '#lib/server/services.js';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async (event) => {
	requireWorker(event);
	const { jobs, blobs } = services();
	const job = await jobs.get(event.params.id);
	if (!job) error(404, 'job not found');
	const blob = await blobs.get(job.inputKey);
	if (!blob) error(410, 'input expired');
	return new Response(blob.body, { headers: { 'content-type': blob.contentType } });
};
