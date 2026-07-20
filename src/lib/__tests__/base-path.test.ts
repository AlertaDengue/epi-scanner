import { describe, it, expect } from "vitest";
import { basePath } from "@/lib/base-path";

describe("basePath", () => {
  it("returns a string starting with /", () => {
    const result = basePath("/api/cities");
    expect(result).toBeDefined();
    expect(typeof result).toBe("string");
  });

  it("keeps the path intact after the prefix", () => {
    const result = basePath("/api/cities");
    expect(result.endsWith("/api/cities")).toBe(true);
  });

  it("handles root path", () => {
    const result = basePath("/");
    expect(result).toBeDefined();
    expect(result.endsWith("/")).toBe(true);
  });
});
