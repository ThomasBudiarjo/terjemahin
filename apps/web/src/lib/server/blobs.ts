// Page files (input, output, result, debug images). Keys look like `jobs/<id>/output.png`; an R2
// lifecycle rule on the `jobs/` prefix deletes them after 2 days (README "Cloud (Cloudflare)").

export interface Blob {
	body: ReadableStream;
	contentType: string;
}

export interface BlobStore {
	put(key: string, body: ArrayBuffer | ReadableStream, contentType: string): Promise<void>;
	get(key: string): Promise<Blob | null>;
}

export function r2Store(bucket: R2Bucket): BlobStore {
	return {
		async put(key, body, contentType) {
			await bucket.put(key, body, { httpMetadata: { contentType } });
		},
		async get(key) {
			const obj = await bucket.get(key);
			if (!obj) return null;
			return {
				body: obj.body,
				contentType: obj.httpMetadata?.contentType ?? 'application/octet-stream'
			};
		}
	};
}
