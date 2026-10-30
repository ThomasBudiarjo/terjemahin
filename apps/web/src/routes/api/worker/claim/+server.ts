import { json } from '@sveltejs/kit';
import { requireWorker, services } from '#lib/server/services.js';
import type { RequestHandler } from './$types';

// A worker has this long to post the result before the job is handed to another worker.
const VISIBILITY_MS = 15 * 60_000;

// POST /api/worker/claim -> 204 (nothing to do) or 200 with the job and its lease.
export const POST: RequestHandler = async (event) => {
	requireWorker(event);
	const { jobs, queue } = services();
	for (let i = 0; i < 5; i++) {
		const claim = await queue.claim(VISIBILITY_MS);
		if (!claim) break;
		const job = await jobs.get(claim.jobId);
		if (!job || job.status === 'done' || job.status === 'failed') {
			await queue.ack(claim.lease); // purged, or delivered twice: drop the message
			continue;
		}
		await jobs.markRunning(job.id);
		return json({
			lease: claim.lease,
			job: {
				job_id: job.id,
				page_id: job.id,
				context: job.context ? JSON.parse(job.context) : {},
				pipeline: job.pipeline ? JSON.parse(job.pipeline) : null,
				debug: job.debug
			}
		});
	}
	return new Response(null, { status: 204 });
};
