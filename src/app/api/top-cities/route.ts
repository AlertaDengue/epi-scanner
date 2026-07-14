import { NextRequest, NextResponse } from "next/server";
import { getWeeksMap, getTopCities } from "@/lib/queries";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease") || "dengue";
  const uf = searchParams.get("uf") || "CE";
  const limit = Number(searchParams.get("limit")) || 20;

  const weeksMap = await getWeeksMap(disease, uf);
  const topCities = getTopCities(weeksMap, limit);

  return NextResponse.json(topCities);
}
