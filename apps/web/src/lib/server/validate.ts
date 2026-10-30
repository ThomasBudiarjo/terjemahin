// Shape checks for the optional JSON fields of POST /v1/pages. They mirror the worker's
// PageContext and PipelineConfig (apps/worker/terjemahin_worker/models.py): a job the worker
// cannot parse would fail later, so it is answered 400 here instead.

const STAGES = ['detector', 'ocr', 'translator', 'mask_refiner', 'inpainter', 'renderer'];

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === 'object' && v !== null && !Array.isArray(v);
const isStrings = (v: unknown) => Array.isArray(v) && v.every((s) => typeof s === 'string');

function parseObject(raw: string, what: string): Obj {
	let v: unknown;
	try {
		v = JSON.parse(raw);
	} catch {
		throw new Error(`${what} must be JSON`);
	}
	if (!isObj(v)) throw new Error(`${what} must be a JSON object`);
	return v;
}

function optional(o: Obj, key: string, check: (v: unknown) => boolean, type: string, what: string) {
	const v = o[key];
	if (v !== undefined && v !== null && !check(v)) throw new Error(`${what}: ${key} must be ${type}`);
}

/** PageContext JSON, re-serialised; throws with a message for the client. */
export function checkContext(raw: string): string {
	const c = parseObject(raw, 'context');
	for (const k of ['series_title', 'series_summary', 'style_notes']) {
		optional(c, k, (v) => typeof v === 'string', 'a string', 'context');
	}
	optional(c, 'previous_pages', (v) => Array.isArray(v) && v.every(isStrings), 'a list of string lists', 'context');
	optional(c, 'glossary', Array.isArray, 'a list', 'context');
	((c.glossary as unknown[] | undefined) ?? []).forEach((g, i) => {
		if (!isObj(g) || typeof g.source !== 'string' || typeof g.target !== 'string') {
			throw new Error(`context: glossary[${i}] needs source and target`);
		}
	});
	return JSON.stringify(c);
}

/** PipelineConfig JSON, re-serialised; unknown fields and stages are refused. */
export function checkPipeline(raw: string): string {
	const p = parseObject(raw, 'pipeline');
	const known = [...STAGES, 'source_lang', 'target_lang', 'options', 'min_ocr_confidence'];
	for (const k of Object.keys(p)) {
		if (!known.includes(k)) throw new Error(`pipeline: unknown field "${k}"`);
	}
	for (const k of [...STAGES, 'source_lang', 'target_lang']) {
		optional(p, k, (v) => typeof v === 'string', 'a string', 'pipeline');
	}
	optional(p, 'min_ocr_confidence', (v) => typeof v === 'number', 'a number', 'pipeline');
	optional(p, 'options', isObj, 'an object', 'pipeline');
	for (const [stage, opts] of Object.entries((p.options as Obj | undefined) ?? {})) {
		if (!STAGES.includes(stage)) throw new Error(`pipeline: options for unknown stage "${stage}"`);
		if (!isObj(opts)) throw new Error(`pipeline: options.${stage} must be an object`);
	}
	return JSON.stringify(p);
}

export function parseBool(raw: string | null, what: string): boolean {
	const v = (raw ?? '').trim().toLowerCase();
	if (['', '0', 'false', 'no', 'off'].includes(v)) return false;
	if (['1', 'true', 'yes', 'on'].includes(v)) return true;
	throw new Error(`${what} must be a boolean`);
}
