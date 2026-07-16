import { NextRequest, NextResponse } from "next/server";

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

const DEFAULT_UF = "RJ";

export async function GET(request: NextRequest) {
  try {
    const forwarded = request.headers.get("x-forwarded-for");
    const clientIp = forwarded?.split(",")[0]?.trim() ?? "8.8.8.8";

    const res = await fetch(`http://ip-api.com/json/${clientIp}?fields=countryCode,region`);
    if (!res.ok) throw new Error("Geolocation API error");

    const data = (await res.json()) as { countryCode?: string; region?: string };

    if (data.countryCode !== "BR") {
      return NextResponse.json({ uf: DEFAULT_UF });
    }

    const uf = data.region ? STATE_NAME_TO_UF[data.region] ?? DEFAULT_UF : DEFAULT_UF;
    return NextResponse.json({ uf });
  } catch {
    return NextResponse.json({ uf: DEFAULT_UF });
  }
}
