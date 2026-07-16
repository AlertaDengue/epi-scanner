import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
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
