import { NextRequest } from "next/server";
import { cachedJson } from "@/lib/cache";
import { episcannerFetch, mapCID10ToDisease } from "@/lib/api-client";

interface DjangoSIRParams {
  cid10: string;
  geocode: string;
  year: number;
  ep_ini: string | null;
  ep_pw: string;
  ep_end: string | null;
  ep_dur: number | null;
  peak_week: number;
  beta: number;
  gamma: number;
  r0: number;
  total_cases: number;
  alpha: number;
  sum_res: number;
  reported_cases: number;
}

interface SIRParamsOutput {
  disease: string;
  CID10: string;
  geocode: number;
  year: number;
  ep_ini: string | null;
  ep_pw: string;
  ep_end: string | null;
  ep_dur: number | null;
  peak_week: number;
  beta: number;
  gamma: number;
  R0: number;
  total_cases: number;
  alpha: number;
  sum_res: number;
  reported_cases: number;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const disease = searchParams.get("disease") || "dengue";
  const uf = searchParams.get("uf") || "CE";
  const year = searchParams.get("year");

  const params = await episcannerFetch<DjangoSIRParams[]>("parameters", { disease, uf, ...(year ? { year } : {}) });

  const output: SIRParamsOutput[] = params.map((p) => ({
    disease: mapCID10ToDisease(p.cid10),
    CID10: p.cid10,
    geocode: Number(p.geocode),
    year: p.year,
    ep_ini: p.ep_ini,
    ep_pw: p.ep_pw,
    ep_end: p.ep_end,
    ep_dur: p.ep_dur,
    peak_week: p.peak_week,
    beta: p.beta,
    gamma: p.gamma,
    R0: p.r0,
    total_cases: p.total_cases,
    alpha: p.alpha,
    sum_res: p.sum_res,
    reported_cases: p.reported_cases
  }));

  return cachedJson(output);
}
