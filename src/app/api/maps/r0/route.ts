import { NextRequest, NextResponse } from "next/server";
import { getSIRParameters, getWeeksMap, getTopR0 } from "@/lib/queries";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease") || "dengue";
  const uf = searchParams.get("uf") || "CE";
  const year = Number(searchParams.get("year")) || new Date().getFullYear();

  const [params, weeksMap] = await Promise.all([
    getSIRParameters(disease, uf),
    getWeeksMap(disease, uf),
  ]);

  // Merge R0 into weeks map features
  const yearParams = params.filter((p) => Number(p.year) === year);
  const r0Map = new Map<number, number>();
  for (const p of yearParams) {
    r0Map.set(Number(p.geocode), Number(p.R0));
  }

  const features = weeksMap.features.map((f) => ({
    ...f,
    properties: {
      ...f.properties,
      R0: r0Map.get(f.properties?.code_muni ?? 0) || 0,
    },
  }));

  const topR0 = getTopR0(params, year, new Map(), 10);

  return NextResponse.json({
    geojson: { type: "FeatureCollection", features },
    topR0,
  });
}
