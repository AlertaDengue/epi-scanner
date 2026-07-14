import { NextRequest, NextResponse } from "next/server";
import { getRateMap, getModelEvalTable, getAlertData, getSIRParameters } from "@/lib/queries";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease") || "dengue";
  const uf = searchParams.get("uf") || "CE";
  const year = Number(searchParams.get("year")) || new Date().getFullYear();

  const [rateMap, alertData, params] = await Promise.all([
    getRateMap(disease, uf, year),
    getAlertData(disease, uf),
    getSIRParameters(disease, uf),
  ]);

  const table = getModelEvalTable(alertData, params, year);

  return NextResponse.json({ rateMap, table });
}
