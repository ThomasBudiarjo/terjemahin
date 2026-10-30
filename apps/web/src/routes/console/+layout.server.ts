import { env } from 'cloudflare:workers';
import type { LayoutServerLoad } from './$types';

export const load: LayoutServerLoad = () => ({ clerkKey: env.PUBLIC_CLERK_PUBLISHABLE_KEY ?? '' });
