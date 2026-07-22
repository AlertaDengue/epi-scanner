import { NextRequest } from "next/server";
import { cachedJson } from "@/lib/cache";
import { episcannerFetch } from "@/lib/api-client";

interface DjangoWeeksPoint {
  geocode: string;
  transmissao: number;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease") || "dengue";
  const uf = searchParams.get("uf") || "CE";
  const year = searchParams.get("year");

  const data = await episcannerFetch<DjangoWeeksPoint[]>("maps/weeks", { disease, uf, ...(year ? { year } : {}) });
  return cachedJson(
    data.map((d) => ({ geocode: Number(d.geocode), transmissao: d.transmissao }))
  );
}
