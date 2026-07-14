import type { NextConfig } from "next";

const nextConfig: NextConfig = {
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
