import { NextRequest, NextResponse } from "next/server";
import { getTimeSeriesData } from "@/lib/queries";
import { getIniEndWeek } from "@/lib/richards";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease") || "dengue";
  const uf = searchParams.get("uf") || "CE";
  const geocode = Number(searchParams.get("geocode"));
  const year = searchParams.get("year");

  if (!geocode) {
    return NextResponse.json(
      { error: "geocode is required" },
      { status: 400 }
    );
  }

  let startDate: string;
  let endDate: string;

  if (year && year !== "all") {
    const { startDate: sd, endDate: ed } = getIniEndWeek(Number(year));
    startDate = sd;
    endDate = ed;
  } else {
    const { startDate: sd, endDate: ed } = getIniEndWeek(2011, new Date().getFullYear());
    startDate = sd;
    endDate = ed;
  }

  const data = await getTimeSeriesData(disease, uf, geocode, startDate, endDate);
  return NextResponse.json(data);
}
