import { defineConfig } from "vitest/config";

export default defineConfig({
  // GitHub Pages serves the site from https://dkirkby.github.io/data-model-loss/,
  // so every asset path has to be base-relative. Runtime fetches must use
  // import.meta.env.BASE_URL rather than a leading slash -- see model/artifact.ts.
  base: "/data-model-loss/",
  test: {
    environment: "jsdom",
    include: ["tests/**/*.test.ts"],
  },
});
