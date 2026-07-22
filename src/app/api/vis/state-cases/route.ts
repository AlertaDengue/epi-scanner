import { NextRequest, NextResponse } from "next/server";

const EPISCANNER_URL =
  process.env.EPISCANNER_API_URL ||
  "http://localhost:8042/api/datastore/episcanner";
const API_BASE = EPISCANNER_URL.replace(/\/?datastore\/episcanner\/?$/, "");
const API_KEY = process.env.EPISCANNER_API_KEY || "";

function getWeekStartDate(year: number, week: number): Date {
  const jan4 = new Date(year, 0, 4);
  const dayOfWeek = jan4.getDay();
  const sundayBefore = new Date(jan4);
  sundayBefore.setDate(jan4.getDate() - ((dayOfWeek + 6) % 7));
  const weekStart = new Date(sundayBefore);
  weekStart.setDate(sundayBefore.getDate() + (week - 1) * 7);
  return weekStart;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease");
  const adm1 = searchParams.get("adm_1");
  const year = searchParams.get("year");

  if (!disease || !adm1) {
    return NextResponse.json(
      { error: "disease and adm_1 are required" },
      { status: 400 }
    );
  }

  const y = year ? Number(year) : new Date().getFullYear();
  const start = getWeekStartDate(y - 1, 45);
  const end = getWeekStartDate(y, 45);

  const params = new URLSearchParams({
    sprint: "false",
    case_definition: "reported",
    disease,
    adm_level: "1",
    adm_1: adm1,
    start: start.toISOString().split("T")[0],
    end: end.toISOString().split("T")[0],
  });

  try {
    const url = `${API_BASE}/vis/dashboard/cases/?${params.toString()}`;
    const res = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        "X-UID-Key": API_KEY,
      },
      cache: "no-store",
    });

    if (!res.ok) {
      const text = await res.text();
      return NextResponse.json(
        { error: `Data-Platform API error ${res.status}: ${text}` },
        { status: res.status }
      );
    }

    const data = await res.json();
    const total = data.reduce(
      (sum: number, d: { cases: number | null }) => sum + (d.cases ?? 0),
      0
    );
    return NextResponse.json({ total });
  } catch {
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
