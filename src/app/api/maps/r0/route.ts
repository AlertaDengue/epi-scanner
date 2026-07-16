import { NextRequest } from "next/server";
import { cachedJson } from "@/lib/cache";
import { episcannerFetch } from "@/lib/api-client";
import { readFileSync } from "fs";
import { join } from "path";

interface DjangoR0Response {
  r0Data: { geocode: string; R0: number }[];
  topR0: { geocode: string; R0: number }[];
}

function getMuniNames(uf: string): Map<number, string> {
  const names = new Map<number, string>();
  try {
    const path = join(process.cwd(), "public", "states", `${uf}.json`);
    const raw = readFileSync(path, "utf-8");
    const geojson = JSON.parse(raw);
    for (const f of geojson.features || []) {
      const props = f.properties;
      if (props?.code_muni && props?.name_muni) {
        names.set(Number(props.code_muni), props.name_muni);
      }
    }
  } catch { /* fallback to numeric code */ }
  return names;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease") || "dengue";
  const uf = searchParams.get("uf") || "CE";
  const year = Number(searchParams.get("year")) || new Date().getFullYear();

  const r0Res = await episcannerFetch<DjangoR0Response>("maps/r0", { disease, uf, year });
  const names = getMuniNames(uf);

  return cachedJson({
    r0Data: (r0Res.r0Data || []).map((r) => ({
      geocode: Number(r.geocode),
      R0: r.R0,
    })),
    topR0: (r0Res.topR0 || []).map((r) => {
      const code = Number(r.geocode);
      return {
        name: names.get(code) ?? String(code),
        geocode: code,
        R0: r.R0,
      };
    }),
  });
}
