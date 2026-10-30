import { error, json } from '@sveltejs/kit';
import { requireWorker, services } from '#lib/server/services.js';
import type { RequestHandler } from './$types';

const DEBUG_STAGES = ['detect', 'mask', 'inpainted', 'final'];

// multipart from the worker: lease, status=done|failed, and either error=<message>, or
// bubbles=<count>, output=<PNG>, result=<PageResult JSON>, debug_<stage>=<PNG> (debug jobs).
export const POST: RequestHandler = async (event) => {
	requireWorker(event);
	const { jobs, queue, blobs } = services();
	const job = await jobs.get(event.params.id);
	if (!job) error(404, 'job not found');
	const form = await event.request.formData();
	const lease = String(form.get('lease') ?? '');
	const status = form.get('status');
	if (!lease) error(400, 'missing lease');

	if (status === 'failed') {
		await jobs.finish(job.id, { status: 'failed', error: String(form.get('error') ?? 'failed') });
	} else if (status === 'done') {
		const output = form.get('output');
		const result = form.get('result');
		if (!(output instanceof File) || !(result instanceof File)) error(400, 'missing output or result');
		const dir = `jobs/${job.id}`;
		const puts = [
			blobs.put(`${dir}/output.png`, await output.arrayBuffer(), 'image/png'),
			blobs.put(`${dir}/result.json`, await result.arrayBuffer(), 'application/json')
		];
		for (const stage of DEBUG_STAGES) {
			const f = form.get(`debug_${stage}`);
			if (f instanceof File) puts.push(blobs.put(`${dir}/debug/${stage}.png`, await f.arrayBuffer(), 'image/png'));
		}
		await Promise.all(puts);
		await jobs.finish(job.id, { status: 'done', bubbles: Number(form.get('bubbles') ?? 0) });
	} else {
		error(400, 'status must be done or failed');
	}
	await queue.ack(lease);
	return json({ ok: true });
};
