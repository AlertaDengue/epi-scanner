import { NextRequest } from "next/server";
import { cachedJson } from "@/lib/cache";
import { episcannerFetch } from "@/lib/api-client";

interface DjangoModelEvalResponse {
  rateMap: { geocode: string; rate: number | null }[];
  table: { range: string; count: number; percentage: number }[];
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease") || "dengue";
  const uf = searchParams.get("uf") || "CE";
  const year = Number(searchParams.get("year")) || new Date().getFullYear();

  const data = await episcannerFetch<DjangoModelEvalResponse>("maps/model-eval", {
    disease,
    uf,
    year,
  });

  return cachedJson({
    rateMap: (data.rateMap || []).map((r) => ({
      geocode: Number(r.geocode),
      rate: r.rate,
    })),
    table: data.table || [],
  });
}
