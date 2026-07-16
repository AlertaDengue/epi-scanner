import { NextRequest, NextResponse } from "next/server";
import { cachedJson } from "@/lib/cache";
import { episcannerFetch } from "@/lib/api-client";

interface DjangoTimeSeriesPoint {
  date: string;
  casos: number;
  casos_est: number;
  casos_cum: number;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease") || "dengue";
  const uf = searchParams.get("uf") || "CE";
  const geocode = searchParams.get("geocode");
  const year = searchParams.get("year");

  if (!geocode) {
  return cachedJson(
      { error: "geocode is required" },
      { status: 400 }
    );
  }

  const data = await episcannerFetch<DjangoTimeSeriesPoint[]>("timeseries", {
    disease,
    uf,
    geocode,
    ...(year && year !== "all" ? { year } : {}),
  });

  return NextResponse.json(
    data.map((d) => ({
      date: d.date,
      casos: d.casos,
      casos_cum: d.casos_cum,
    }))
  );
}
