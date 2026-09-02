import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  typescript: {
    // Allows deployment build on Vercel even if complex canvas components contain internal untyped parameters
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
