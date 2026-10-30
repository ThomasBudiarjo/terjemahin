// The job queue carries only job ids; job state lives in the JobStore. The Python worker never
// talks to the queue itself: it calls /api/worker/claim, and this app claims for it, so swapping
// the queue (another broker, a self-hosted one) touches only this file.

export interface Claim {
	jobId: string;
	lease: string; // passed back to ack(); opaque to callers
}

export interface JobQueue {
	enqueue(jobId: string): Promise<void>;
	/** One job for a worker, hidden from other workers for `visibilityMs`, or null. */
	claim(visibilityMs: number): Promise<Claim | null>;
	/** Removes a claimed job for good (done or failed). */
	ack(lease: string): Promise<void>;
}

/** Cloudflare Queues: send through the binding, pull and ack through the REST API
 * (https://developers.cloudflare.com/queues/configuration/pull-consumers/). */
export function cloudflareQueue(
	binding: Queue,
	api: { accountId: string; queueId: string; token: string }
): JobQueue {
	const base = `https://api.cloudflare.com/client/v4/accounts/${api.accountId}/queues/${api.queueId}/messages`;
	const call = async (action: 'pull' | 'ack', body: unknown) => {
		const res = await fetch(`${base}/${action}`, {
			method: 'POST',
			headers: { authorization: `Bearer ${api.token}`, 'content-type': 'application/json' },
			body: JSON.stringify(body)
		});
		if (!res.ok) throw new Error(`queues ${action}: HTTP ${res.status} ${await res.text()}`);
		return res.json<{ result: { messages: { body: string; lease_id: string }[] } }>();
	};
	return {
		// `text` bodies reach pull consumers as plain strings (json/bytes would be base64)
		async enqueue(jobId) {
			await binding.send(jobId, { contentType: 'text' });
		},
		async claim(visibilityMs) {
			const { result } = await call('pull', { batch_size: 1, visibility_timeout_ms: visibilityMs });
			const m = result.messages[0];
			return m ? { jobId: m.body, lease: m.lease_id } : null;
		},
		async ack(lease) {
			await call('ack', { acks: [{ lease_id: lease }], retries: [] });
		}
	};
}

/** The jobs table as the queue: for local dev (no Queues pull API there) or a setup without
 * Cloudflare Queues. A claim is a conditional UPDATE, so two workers never get the same job. */
export function d1Queue(db: D1Database): JobQueue {
	return {
		async enqueue() {}, // the queued row written by the JobStore is the message
		async claim(visibilityMs) {
			const now = Date.now();
			const lease = crypto.randomUUID();
			const row = await db
				.prepare(
					`UPDATE jobs SET lease_id = ?1, lease_until = ?2
					 WHERE id = (SELECT id FROM jobs
					             WHERE status IN ('queued', 'running') AND coalesce(lease_until, 0) < ?3
					             ORDER BY created_at LIMIT 1)
					 RETURNING id`
				)
				.bind(lease, now + visibilityMs, now)
				.first<{ id: string }>();
			return row ? { jobId: row.id, lease } : null;
		},
		async ack(lease) {
			await db.prepare('UPDATE jobs SET lease_id = NULL WHERE lease_id = ?1').bind(lease).run();
		}
	};
}
