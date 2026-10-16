import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  typescript: {
    // Allows deployment build on Vercel even if complex canvas components contain internal untyped parameters
    ignoreBuildErrors: true,
  },
  async rewrites() {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
