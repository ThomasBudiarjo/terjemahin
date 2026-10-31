import type { Handle } from '@sveltejs/kit/hooks';
import { waitUntil } from 'cloudflare:workers';

// Closes the MongoDB connection a request opened through services().
export const handle: Handle = async ({ event, resolve }) => {
	try {
		return await resolve(event);
	} finally {
		if (event.locals.mongo) waitUntil(event.locals.mongo.close());
	}
};
