import { NextResponse } from "next/server";
import { getStatesList } from "@/lib/queries";

export async function GET() {
  const states = await getStatesList();
  return NextResponse.json(states);
}
