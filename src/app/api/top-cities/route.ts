import { NextRequest } from "next/server";
import { cachedJson } from "@/lib/cache";
import { episcannerFetch } from "@/lib/api-client";

interface DjangoTopCity {
  name_muni: string;
  transmissao: number;
  geocode: string;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease") || "dengue";
  const uf = searchParams.get("uf") || "CE";
  const limit = Number(searchParams.get("limit")) || 20;

  const cities = await episcannerFetch<DjangoTopCity[]>("top-cities", {
    disease,
    uf,
    limit,
  });

  return cachedJson(
    cities.map((c) => ({
      name_muni: c.name_muni,
      transmissao: c.transmissao,
      geocode: Number(c.geocode),
    }))
  );
}
