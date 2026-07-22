import { NextRequest, NextResponse } from "next/server";

const EPISCANNER_URL =
  process.env.EPISCANNER_API_URL ||
  "http://localhost:8042/api/datastore/episcanner";
const API_BASE = EPISCANNER_URL.replace(/\/?datastore\/episcanner\/?$/, "");
const API_KEY = process.env.EPISCANNER_API_KEY || "";

const DEFAULT_UF = "RJ";

const STATE_NAME_TO_UF: Record<string, string> = {
  Acre: "AC", Alagoas: "AL", Amapá: "AP", Amazonas: "AM", Bahia: "BA",
  Ceará: "CE", "Distrito Federal": "DF", "Espírito Santo": "ES", "Goiás": "GO",
  Maranhão: "MA", "Mato Grosso": "MT", "Mato Grosso do Sul": "MS",
  "Minas Gerais": "MG", Pará: "PA", Paraíba: "PB", Paraná: "PR",
  Pernambuco: "PE", Piauí: "PI", "Rio de Janeiro": "RJ",
  "Rio Grande do Norte": "RN", "Rio Grande do Sul": "RS",
  Rondônia: "RO", Roraima: "RR", "Santa Catarina": "SC",
  "São Paulo": "SP", Sergipe: "SE", Tocantins: "TO",
};

function normalize(str: string): string {
  return str
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim();
}

interface CityEntry {
  geocode: string;
  name: string;
}

export async function GET(request: NextRequest) {
  try {
    const forwarded = request.headers.get("x-forwarded-for");
    const clientIp = forwarded?.split(",")[0]?.trim() ?? "8.8.8.8";

    const geoRes = await fetch(
      `http://ip-api.com/json/${clientIp}?fields=countryCode,regionName,city`
    );
    if (!geoRes.ok) throw new Error("IP geolocation API error");

    const geoData = (await geoRes.json()) as {
      countryCode?: string;
      regionName?: string;
      city?: string;
    };

    if (geoData.countryCode !== "BR" || !geoData.regionName || !geoData.city) {
      return NextResponse.json({ uf: DEFAULT_UF, geocode: null, cityName: null });
    }

    const uf = STATE_NAME_TO_UF[geoData.regionName] ?? DEFAULT_UF;
    const cityName = normalize(geoData.city);

    const citiesRes = await fetch(
      `${API_BASE}/datastore/episcanner/cities/?disease=dengue&uf=${uf}&year=${new Date().getFullYear()}`,
      {
        headers: {
          "Content-Type": "application/json",
          "X-UID-Key": API_KEY,
        },
        cache: "no-store",
      }
    );

    if (!citiesRes.ok) {
      return NextResponse.json({ uf, geocode: null, cityName: geoData.city });
    }

    const cities: CityEntry[] = await citiesRes.json();
    const match = cities.find((c) => normalize(c.name) === cityName);

    return NextResponse.json({
      uf,
      geocode: match?.geocode ?? null,
      cityName: match?.name ?? geoData.city,
    });
  } catch {
    return NextResponse.json({ uf: DEFAULT_UF, geocode: null, cityName: null });
  }
}
