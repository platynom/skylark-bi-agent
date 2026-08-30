/** @type {import('next').NextConfig} */

// Routing note:
// In production, /api/* routing is owned entirely by vercel.json, which rewrites
// /api/(.*) -> /api/index so that Vercel's Python runtime hands the request to the
// FastAPI app in api/index.py. Declaring a Next.js rewrite for the same path here
// would shadow that vercel.json rule (Next.js routing is applied by the framework
// build), and a self-referential rewrite is either a no-op or a loop. So production
// returns no rewrites at all.
//
// In development there is no Vercel layer, so we proxy /api/* to the local uvicorn
// process instead.
const nextConfig = {
  rewrites: async () => {
    if (process.env.NODE_ENV !== 'development') {
      return [];
    }
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*',
      },
    ];
  },
};

export default nextConfig;
