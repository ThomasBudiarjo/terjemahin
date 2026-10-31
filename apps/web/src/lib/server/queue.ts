// Workers take jobs through a lease on the job document. The Python worker never talks to the
// database itself: it calls /api/worker/claim, and this app claims for it, so swapping the queue
// (a broker, another database) touches only this file.
import type { Collection } from 'mongodb';
import { STALE_MS, type JobDoc } from './jobs.js';

export interface Claim {
	jobId: string;
	lease: string; // passed back to ack(); opaque to callers
}

export interface JobQueue {
	/** One job for a worker, hidden from other workers for `visibilityMs`, or null. */
	claim(visibilityMs: number): Promise<Claim | null>;
	/** Releases a claimed job for good (done or failed). */
	ack(lease: string): Promise<void>;
}

/** The jobs collection as the queue: a queued document is the message. A claim is one atomic
 * findOneAndUpdate on the oldest unfinished job without a live lease, so two workers never get
 * the same job, and a job whose lease runs out goes to the next worker. */
export function mongoQueue(col: Collection<JobDoc>): JobQueue {
	return {
		async claim(visibilityMs) {
			const now = Date.now();
			const lease = crypto.randomUUID();
			const doc = await col.findOneAndUpdate(
				{
					status: { $in: ['queued', 'running'] },
					createdAt: { $gt: new Date(now - STALE_MS) },
					leaseUntil: { $not: { $gt: new Date(now) } } // none, or expired
				},
				{ $set: { lease, leaseUntil: new Date(now + visibilityMs) } },
				{ sort: { createdAt: 1 }, projection: { _id: 1 } }
			);
			return doc ? { jobId: doc._id, lease } : null;
		},
		async ack(lease) {
			await col.updateOne({ lease }, { $unset: { lease: '', leaseUntil: '' } });
		}
	};
}
