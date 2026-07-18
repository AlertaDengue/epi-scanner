import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  trailingSlash: true,
  basePath: process.env.NEXT_PUBLIC_URL_PREFIX || "",
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "info.dengue.mat.br",
      },
    ],
  },
};

export default nextConfig;
