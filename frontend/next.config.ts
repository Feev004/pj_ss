import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: "http://127.0.0.1:8080/:path*",
      },
      {
        source: "/video_feed/:path*",
        destination: "http://127.0.0.1:8080/video_feed/:path*",
      },
      {
        source: "/video/:path*",
        destination: "http://127.0.0.1:8080/video/:path*",
      },
      {
        source: "/detection_frame/:path*",
        destination: "http://127.0.0.1:8080/detection_frame/:path*",
      },
      {
        source: "/accident_frame/:path*",
        destination: "http://127.0.0.1:8080/accident_frame/:path*",
      },
    ];
  },
};

export default nextConfig;
