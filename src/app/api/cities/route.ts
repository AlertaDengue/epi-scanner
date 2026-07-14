import { NextRequest, NextResponse } from "next/server";
import { getCitiesForState } from "@/lib/queries";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease") || "dengue";
  const uf = searchParams.get("uf") || "CE";

  const cities = await getCitiesForState(disease, uf);
  return NextResponse.json(cities);
}
