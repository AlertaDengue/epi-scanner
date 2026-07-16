import { NextResponse } from "next/server";

const CACHE_HEADER = "s-maxage=86400, stale-while-revalidate=86400";

export function cachedJson(data: unknown, init?: ResponseInit) {
  return NextResponse.json(data, {
    ...init,
    headers: {
      ...init?.headers,
      "Cache-Control": CACHE_HEADER,
    },
  });
}
