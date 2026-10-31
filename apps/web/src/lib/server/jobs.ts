// Job state: one document per page in the MongoDB `jobs` collection, which is also the queue
// (queue.ts). A TTL index deletes documents RETENTION_MS after creation; their R2 files expire at
// the same age.
import type { Collection } from 'mongodb';

export const RETENTION_MS = 2 * 24 * 3600_000;
// A job not finished after this is reported `expired` and no longer handed to workers.
export const STALE_MS = 24 * 3600_000;

export type JobStatus = 'queued' | 'running' | 'done' | 'failed';

export interface Job {
	id: string;
	userId: string;
	status: JobStatus;
	error: string | null;
	inputKey: string;
	context: string | null;
	pipeline: string | null;
	debug: boolean;
	bubbles: number | null;
	createdAt: number;
	updatedAt: number;
}

export interface JobStore {
	create(job: Job): Promise<void>;
	get(id: string): Promise<Job | null>;
	markRunning(id: string): Promise<void>;
	finish(id: string, result: { status: 'done' | 'failed'; error?: string; bubbles?: number }): Promise<void>;
}

/** What GET /v1/jobs/:id answers. */
export function publicJob(j: Job, now = Date.now()) {
	const stale = (j.status === 'queued' || j.status === 'running') && now - j.createdAt > STALE_MS;
	return {
		job_id: j.id,
		status: stale ? 'expired' : j.status,
		error: j.error ?? undefined,
		bubbles: j.bubbles ?? undefined,
		debug: j.debug,
		created_at: new Date(j.createdAt).toISOString(),
		updated_at: new Date(j.updatedAt).toISOString()
	};
}

export type JobDoc = Omit<Job, 'id' | 'createdAt' | 'updatedAt'> & {
	_id: string;
	createdAt: Date; // a Date, for the TTL index
	updatedAt: Date;
	lease?: string; // set while a worker holds the job (queue.ts)
	leaseUntil?: Date;
};

// createIndexes is a no-op when the indexes exist; once per isolate is enough.
let indexed = false;

export function mongoJobStore(col: Collection<JobDoc>): JobStore {
	return {
		async create(j) {
			if (!indexed) {
				await col.createIndexes([
					{ key: { createdAt: 1 }, expireAfterSeconds: RETENTION_MS / 1000 },
					{ key: { status: 1, createdAt: 1 } } // queue.ts claim
				]);
				indexed = true;
			}
			const { id, createdAt, updatedAt, ...rest } = j;
			await col.insertOne({ _id: id, ...rest, createdAt: new Date(createdAt), updatedAt: new Date(updatedAt) });
		},
		async get(id) {
			const d = await col.findOne({ _id: id });
			if (!d) return null;
			return {
				id: d._id,
				userId: d.userId,
				status: d.status,
				error: d.error,
				inputKey: d.inputKey,
				context: d.context,
				pipeline: d.pipeline,
				debug: d.debug,
				bubbles: d.bubbles,
				createdAt: d.createdAt.getTime(),
				updatedAt: d.updatedAt.getTime()
			};
		},
		async markRunning(id) {
			await col.updateOne({ _id: id, status: 'queued' }, { $set: { status: 'running', updatedAt: new Date() } });
		},
		async finish(id, res) {
			await col.updateOne(
				{ _id: id },
				{ $set: { status: res.status, error: res.error ?? null, bubbles: res.bubbles ?? null, updatedAt: new Date() } }
			);
		}
	};
}
