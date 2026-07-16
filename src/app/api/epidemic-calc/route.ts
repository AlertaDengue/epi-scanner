import { NextRequest, NextResponse } from "next/server";
import { cachedJson } from "@/lib/cache";
import { episcannerFetch } from "@/lib/api-client";
import { richards } from "@/lib/richards";

interface DjangoTimeSeriesPoint {
  date: string;
  casos: number;
  casos_est: number;
  casos_cum: number;
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { disease, geocode, peakWeek, R0: r0, totalCases, year } = body;

  if (!disease || !geocode || !year) {
    return NextResponse.json(
      { error: "disease, geocode, and year are required" },
      { status: 400 }
    );
  }

  const timeSeries = await episcannerFetch<DjangoTimeSeriesPoint[]>("timeseries", {
    disease,
    geocode,
    year,
  });

  const dates = timeSeries.map((d) => d.date);
  const dataCumulative = timeSeries.map((d) => d.casos_cum);

  const r = 1 - 1 / (r0 || 2);
  const gamma = 0.3;
  const b = (r * gamma) / (1 - r);
  const a = b / (gamma + b);

  const pw = peakWeek || 10;
  const tc = totalCases || 1000;

  const modelCumulative: number[] = [];
  for (let i = 0; i < 52; i++) {
    modelCumulative.push(richards(tc, a, b, i, pw));
  }

  const startDateObj = new Date(dates[0] || `${year}-01-01`);
  const peakDate = new Date(startDateObj);
  peakDate.setDate(peakDate.getDate() + pw * 7);
  const peakWeekDate = peakDate.toISOString().split("T")[0];

  return cachedJson({
    dates,
    dataCumulative,
    modelCumulative,
    peakWeekDate,
  });
}
