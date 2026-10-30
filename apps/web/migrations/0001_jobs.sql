-- One row per translated page. Rows and their R2 files are removed after 2 days.
CREATE TABLE jobs (
	id TEXT PRIMARY KEY,
	user_id TEXT NOT NULL,
	status TEXT NOT NULL, -- queued | running | done | failed
	error TEXT,
	input_key TEXT NOT NULL, -- R2 key of the uploaded page
	context TEXT, -- PageContext JSON
	pipeline TEXT, -- PipelineConfig JSON
	debug INTEGER NOT NULL DEFAULT 0, -- keep per-stage images
	bubbles INTEGER,
	-- only used when QUEUE_DRIVER=d1 (the table is the queue)
	lease_id TEXT,
	lease_until INTEGER,
	created_at INTEGER NOT NULL, -- unix ms
	updated_at INTEGER NOT NULL
);

CREATE INDEX jobs_created_at ON jobs (created_at);
CREATE INDEX jobs_queued ON jobs (status, created_at);
