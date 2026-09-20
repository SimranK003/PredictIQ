import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone output produces a minimal, self-contained server bundle
  // (node_modules pruned to only what's actually imported) — this is
  // what the production Docker image runs; `next dev`/`next start`
  // outside Docker are unaffected.
  output: "standalone",
};

export default nextConfig;
