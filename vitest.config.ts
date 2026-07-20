import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    globals: true,
    onConsoleLog(log: string) {
      if (log.includes("not wrapped in act(")) return false;
    },
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "lcov"],
      include: [
        "src/lib/richards.ts",
        "src/lib/base-path.ts",
        "src/lib/constants.ts",
        "src/lib/cache.ts",
        "src/lib/api-client.ts",
        "src/components/charts/stat-cards.tsx",
        "src/components/layout/header.tsx",
        "src/components/layout/footer.tsx",
        "src/components/layout/sidebar.tsx",
        "src/components/ui/slider.tsx",
        "src/components/ui/badge.tsx",
        "src/components/ui/select.tsx",
        "src/components/ui/separator.tsx",
        "src/components/ui/table.tsx",
        "src/components/tables/rank-table.tsx",
      ],
      exclude: [
        "src/**/*.d.ts",
        "src/**/__tests__/**",
        "src/types/**",
      ],
      thresholds: {
        lines: 80,
        functions: 75,
        branches: 70,
        statements: 80,
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
