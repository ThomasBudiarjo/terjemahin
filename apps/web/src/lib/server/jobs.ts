// Job state. Rows older than RETENTION_MS are deleted (their R2 files expire at the same age).

export const RETENTION_MS = 2 * 24 * 3600_000;
// A job not finished after this is reported `expired`: Cloudflare Queues (free plan) drops
// messages after 24 hours, so it will never run.
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
	/** Deletes rows older than `before` (unix ms). */
	purge(before: number): Promise<void>;
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

type Row = {
	id: string;
	user_id: string;
	status: JobStatus;
	error: string | null;
	input_key: string;
	context: string | null;
	pipeline: string | null;
	debug: number;
	bubbles: number | null;
	created_at: number;
	updated_at: number;
};

export function d1JobStore(db: D1Database): JobStore {
	return {
		async create(j) {
			await db
				.prepare(
					`INSERT INTO jobs (id, user_id, status, error, input_key, context, pipeline, debug,
					                   bubbles, created_at, updated_at)
					 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)`
				)
				.bind(j.id, j.userId, j.status, j.error, j.inputKey, j.context, j.pipeline,
					j.debug ? 1 : 0, j.bubbles, j.createdAt, j.updatedAt)
				.run();
		},
		async get(id) {
			const r = await db.prepare('SELECT * FROM jobs WHERE id = ?1').bind(id).first<Row>();
			if (!r) return null;
			return {
				id: r.id,
				userId: r.user_id,
				status: r.status,
				error: r.error,
				inputKey: r.input_key,
				context: r.context,
				pipeline: r.pipeline,
				debug: r.debug === 1,
				bubbles: r.bubbles,
				createdAt: r.created_at,
				updatedAt: r.updated_at
			};
		},
		async markRunning(id) {
			await db
				.prepare(`UPDATE jobs SET status = 'running', updated_at = ?2 WHERE id = ?1 AND status = 'queued'`)
				.bind(id, Date.now())
				.run();
		},
		async finish(id, res) {
			await db
				.prepare('UPDATE jobs SET status = ?2, error = ?3, bubbles = ?4, updated_at = ?5 WHERE id = ?1')
				.bind(id, res.status, res.error ?? null, res.bubbles ?? null, Date.now())
				.run();
		},
		async purge(before) {
			await db.prepare('DELETE FROM jobs WHERE created_at < ?1').bind(before).run();
		}
	};
}
