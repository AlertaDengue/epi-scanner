import { NextRequest, NextResponse } from "next/server";
import { getSIRParameters } from "@/lib/queries";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease") || "dengue";
  const uf = searchParams.get("uf") || "CE";

  const params = await getSIRParameters(disease, uf);
  return NextResponse.json(params);
}
