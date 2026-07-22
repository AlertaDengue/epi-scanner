import { NextRequest } from "next/server";
import { cachedJson } from "@/lib/cache";
import { episcannerFetch } from "@/lib/api-client";

interface DjangoCity {
  geocode: string;
  name: string;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease") || "dengue";
  const uf = searchParams.get("uf") || "CE";
  const year = searchParams.get("year");

  const cities = await episcannerFetch<DjangoCity[]>("cities", { disease, uf, ...(year ? { year } : {}) });
  return cachedJson(cities);
}
