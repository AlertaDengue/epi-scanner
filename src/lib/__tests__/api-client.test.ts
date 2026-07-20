import { describe, it, expect, vi, beforeEach } from "vitest";

describe("episcannerFetch", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it("constructs URL with base and endpoint", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: "test" }),
    });
    global.fetch = mockFetch;

    process.env.EPISCANNER_API_URL = "http://test-api:8042/api/datastore/episcanner";

    const { episcannerFetch } = await import("@/lib/api-client");

    await episcannerFetch("timeseries", { disease: "dengue", uf: "RJ" });

    const url = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain("http://test-api:8042/api/datastore/episcanner/timeseries/");
    expect(url).toContain("disease=dengue");
    expect(url).toContain("uf=RJ");
  });

  it("omits undefined params from URL", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({}),
    });
    global.fetch = mockFetch;

    const { episcannerFetch } = await import("@/lib/api-client");

    await episcannerFetch("test", { a: "1", b: undefined });

    const url = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain("a=1");
    expect(url).not.toContain("b=");
  });

  it("sends X-UID-Key header when API_KEY is set", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({}),
    });
    global.fetch = mockFetch;

    process.env.EPISCANNER_API_KEY = "test-key";

    const { episcannerFetch } = await import("@/lib/api-client");

    await episcannerFetch("test");

    const headers = mockFetch.mock.calls[0][1]?.headers as Record<string, string>;
    expect(headers["X-UID-Key"]).toBe("test-key");
  });

  it("throws on non-OK response", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      text: () => Promise.resolve("Server error"),
    });
    global.fetch = mockFetch;

    const { episcannerFetch } = await import("@/lib/api-client");

    await expect(episcannerFetch("test")).rejects.toThrow("Episcanner API error 500");
  });

  it("returns JSON data on success", async () => {
    const mockData = { items: [1, 2, 3] };
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockData),
    });
    global.fetch = mockFetch;

    const { episcannerFetch } = await import("@/lib/api-client");

    const result = await episcannerFetch("test");
    expect(result).toEqual(mockData);
  });
});

describe("cache", () => {
  it("cachedJson sets Cache-Control header", async () => {
    const { cachedJson } = await import("@/lib/cache");

    const response = cachedJson({ data: "test" }) as Response;
    const headers = response.headers;

    expect(headers.get("Cache-Control")).toContain("s-maxage=86400");
    expect(headers.get("Cache-Control")).toContain("stale-while-revalidate=86400");
  });

  it("cachedJson returns JSON response", async () => {
    const { cachedJson } = await import("@/lib/cache");

    const response = cachedJson({ hello: "world" }) as Response;
    const data = await response.json();

    expect(data).toEqual({ hello: "world" });
  });
});
