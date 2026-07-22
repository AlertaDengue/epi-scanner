import { NextRequest, NextResponse } from "next/server";

const EPISCANNER_URL =
  process.env.EPISCANNER_API_URL ||
  "http://localhost:8042/api/datastore/episcanner";
const API_BASE = EPISCANNER_URL.replace(/\/?datastore\/episcanner\/?$/, "");
const API_KEY = process.env.EPISCANNER_API_KEY || "";

function getEndOfWeek(): string {
  const now = new Date();
  const day = now.getUTCDay();
  const sunday = new Date(now);
  sunday.setUTCDate(now.getUTCDate() + ((7 - day) % 7 || 7));
  sunday.setUTCHours(23, 59, 59, 999);
  return sunday.toISOString().split("T")[0];
}

function getSecondsUntilSunday(): number {
  const now = new Date();
  const day = now.getUTCDay();
  const daysUntilSunday = (7 - day) % 7 || 7;
  const sunday = new Date(now);
  sunday.setUTCDate(now.getUTCDate() + daysUntilSunday);
  sunday.setUTCHours(23, 59, 59, 999);
  return Math.max(0, Math.floor((sunday.getTime() - now.getTime()) / 1000));
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease");
  const adm2 = searchParams.get("adm_2");

  if (!disease || !adm2) {
    return NextResponse.json(
      { error: "disease and adm_2 are required" },
      { status: 400 }
    );
  }

  const start = "2010-01-01";
  const end = getEndOfWeek();

  const params = new URLSearchParams({
    sprint: "false",
    case_definition: "reported",
    disease,
    adm_level: "2",
    adm_2: adm2,
    start,
    end,
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
    const maxAge = getSecondsUntilSunday();

    return NextResponse.json(data, {
      headers: {
        "Cache-Control": `public, max-age=${maxAge}, s-maxage=${maxAge}`,
      },
    });
  } catch {
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
