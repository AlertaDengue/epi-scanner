import {
  readAlertData,
  readDuckDBExport,
  loadGeoJSON,
} from "./db";
import { STATES } from "./constants";
import { getIniEndWeek } from "./richards";
import type { GeoJSON, GeoFeature, SIRParams } from "./types";

// Simple in-memory cache
const cache = new Map<string, unknown>();

function cacheKey(...args: string[]): string {
  return args.join("::");
}

export async function getGeoJSON(): Promise<GeoJSON.FeatureCollection> {
  const key = cacheKey("geojson");
  if (cache.has(key)) return cache.get(key) as GeoJSON.FeatureCollection;
  const data = await loadGeoJSON();
  cache.set(key, data);
  return data;
}

export async function getStatesList() {
  return Object.entries(STATES).map(([code, name]) => ({ code, name }));
}

export async function getCitiesForState(
  disease: string,
  uf: string
): Promise<{ geocode: number; name: string }[]> {
  const key = cacheKey("cities", disease, uf);
  if (cache.has(key)) return cache.get(key) as { geocode: number; name: string }[];

  const brmap = await getGeoJSON();
  const alertData = await getAlertData(disease, uf);

  const geocodes = new Set(
    alertData.map((row) => Number(row.municipio_geocodigo))
  );

  const cities: { geocode: number; name: string }[] = [];
  for (const gc of geocodes) {
    const feature = brmap.features.find(
      (f) => Number(f.properties?.code_muni) === gc
    );
    cities.push({
      geocode: gc,
      name: feature?.properties?.name_muni || "",
    });
  }

  cache.set(key, cities);
  return cities;
}

export async function getAlertData(
  disease: string,
  uf: string
): Promise<Record<string, unknown>[]> {
  const key = cacheKey("alert", disease, uf);
  if (cache.has(key)) return cache.get(key) as Record<string, unknown>[];
  const data = await readAlertData(disease, uf);
  cache.set(key, data);
  return data;
}

export async function getSIRParameters(
  disease: string,
  uf: string
): Promise<SIRParams[]> {
  const key = cacheKey("params", disease, uf);
  if (cache.has(key)) return cache.get(key) as SIRParams[];
  const data = await readDuckDBExport(disease, uf);
  cache.set(key, data);
  return data as unknown as SIRParams[];
}

export async function getWeeksMap(
  disease: string,
  uf: string
): Promise<GeoJSON.FeatureCollection> {
  const key = cacheKey("weeksmap", disease, uf);
  if (cache.has(key)) return cache.get(key) as GeoJSON.FeatureCollection;

  const brmap = await getGeoJSON();
  const alertData = await getAlertData(disease, uf);

  const stateFeatures = brmap.features.filter(
    (f) => f.properties?.abbrev_state === uf
  );

  // Sum transmission weeks per city
  const transmissaoMap = new Map<number, number>();
  for (const row of alertData) {
    const gc = Number(row.municipio_geocodigo);
    const trans = Number(row.transmissao) || 0;
    transmissaoMap.set(gc, (transmissaoMap.get(gc) || 0) + trans);
  }

  const features: GeoFeature[] = stateFeatures.map((f) => ({
    type: "Feature" as const,
    geometry: f.geometry,
    properties: {
      code_muni: f.properties?.code_muni ?? 0,
      name_muni: f.properties?.name_muni ?? "",
      abbrev_state: f.properties?.abbrev_state ?? "",
      transmissao: transmissaoMap.get(Number(f.properties?.code_muni)) || 0,
    },
  }));

  const result = { type: "FeatureCollection", features } as unknown as GeoJSON.FeatureCollection;
  cache.set(key, result);
  return result;
}

export async function getRateMap(
  disease: string,
  uf: string,
  year: number
): Promise<Record<string, unknown>[]> {
  const key = cacheKey("ratemap", disease, uf, String(year));
  if (cache.has(key)) return cache.get(key) as Record<string, unknown>[];

  const alertData = await getAlertData(disease, uf);
  const params = await getSIRParameters(disease, uf);

  // Filter alert data to SE range for this year
  const seMin = (year - 1) * 100 + 45;
  const seMax = year * 100 + 45;
  const filtered = alertData.filter((row) => {
    const se = Number(row.SE);
    return se >= seMin && se < seMax;
  });

  // Sum cases per city
  const casosMap = new Map<number, number>();
  for (const row of filtered) {
    const gc = Number(row.municipio_geocodigo);
    casosMap.set(gc, (casosMap.get(gc) || 0) + (Number(row.casos) || 0));
  }

  // Get parameters for this year
  const yearParams = params.filter((p) => Number(p.year) === year);
  const paramsMap = new Map<number, SIRParams>();
  for (const p of yearParams) {
    paramsMap.set(Number(p.geocode), p);
  }

  const result: Record<string, unknown>[] = [];
  for (const [gc, observed] of casosMap) {
    const param = paramsMap.get(gc);
    const totalCases = param ? Number(param.total_cases) : undefined;
    result.push({
      code_muni: gc,
      observed_cases: observed,
      total_cases: totalCases,
      rate: totalCases ? observed / totalCases : null,
    });
  }

  cache.set(key, result);
  return result;
}

export function getTopCities(
  weeksMap: GeoJSON.FeatureCollection,
  limit = 10
): { name_muni: string; transmissao: number; code_muni: number }[] {
  return weeksMap.features
    .map((f) => ({
      name_muni: f.properties?.name_muni || "",
      transmissao: f.properties?.transmissao || 0,
      code_muni: f.properties?.code_muni || 0,
    }))
    .sort((a, b) => b.transmissao - a.transmissao)
    .slice(0, limit);
}

export function getTopR0(
  params: SIRParams[],
  year: number,
  cities: Map<number, string>,
  limit = 10
): { name: string; geocode: number; R0: number }[] {
  return params
    .filter((p) => Number(p.year) === year)
    .sort((a, b) => Number(b.R0) - Number(a.R0))
    .slice(0, limit)
    .map((p) => ({
      name: cities.get(Number(p.geocode)) || "",
      geocode: Number(p.geocode),
      R0: Number(p.R0),
    }));
}

export function getModelEvalTable(
  alertData: Record<string, unknown>[],
  params: SIRParams[],
  year: number
): { range: string; count: number; percentage: number }[] {
  const seMin = (year - 1) * 100 + 45;
  const seMax = year * 100 + 45;

  const filtered = alertData.filter((row) => {
    const se = Number(row.SE);
    return se >= seMin && se < seMax;
  });

  // Sum cases per city
  const casosMap = new Map<number, number>();
  for (const row of filtered) {
    const gc = Number(row.municipio_geocodigo);
    casosMap.set(gc, (casosMap.get(gc) || 0) + (Number(row.casos) || 0));
  }

  const yearParams = params.filter((p) => Number(p.year) === year);
  const rates: number[] = [];

  for (const [gc, observed] of casosMap) {
    const param = yearParams.find((p) => Number(p.geocode) === gc);
    if (param && Number(param.total_cases) > 0) {
      rates.push(observed / Number(param.total_cases));
    }
  }

  const bins = [0, 0.5, 0.95, 1.05, 2, Infinity];
  const ranges = [
    "0 - 0.5",
    "0.5 - 0.95",
    "0.95 - 1.05",
    "1.05 - 2",
    "> 2",
  ];

  const counts = ranges.map((range, i) => {
    const count = rates.filter(
      (r) => r >= bins[i] && r < bins[i + 1]
    ).length;
    return { range, count, percentage: 0 };
  });

  const total = rates.length;
  for (const c of counts) {
    c.percentage = total > 0 ? Math.round((c.count / total) * 10000) / 100 : 0;
  }

  return counts;
}

export async function getTimeSeriesData(
  disease: string,
  uf: string,
  geocode: number,
  startDate: string,
  endDate: string
): Promise<{ date: string; casos: number; casos_cum: number }[]> {
  const alertData = await getAlertData(disease, uf);

  const cityData = alertData
    .filter((row) => Number(row.municipio_geocodigo) === geocode)
    .filter((row) => {
      const d = String(row.data_iniSE);
      return d >= startDate && d <= endDate;
    })
    .sort((a, b) => String(a.data_iniSE).localeCompare(String(b.data_iniSE)));

  let cumulative = 0;
  return cityData.map((row) => {
    const casos = Number(row.casos) || 0;
    cumulative += casos;
    return {
      date: String(row.data_iniSE),
      casos,
      casos_cum: cumulative,
    };
  });
}

export function getSIRParamsForCity(
  params: SIRParams[],
  geocode: number
): SIRParams[] {
  return params.filter((p) => Number(p.geocode) === geocode);
}

export async function getMedianParams(
  params: SIRParams[],
  geocode: number,
  year: number,
  currentCases: number
): Promise<{
  medianR0: number;
  medianPeak: number;
  medianCases: number;
  minCases: number;
  maxCases: number;
  step: number;
}> {
  const pastParams = params.filter(
    (p) => Number(p.geocode) === geocode && Number(p.year) < year
  );
  const currentParams = params.filter(
    (p) => Number(p.geocode) === geocode && Number(p.year) === year
  );

  let medianR0 = 2;
  let medianPeak = 10;
  let medianCases = currentCases;

  if (pastParams.length > 0) {
    const r0Values = pastParams.map((p) => Number(p.R0)).sort((a, b) => a - b);
    const peakValues = pastParams
      .map((p) => Number(p.peak_week))
      .sort((a, b) => a - b);
    const casesValues = pastParams
      .map((p) => Number(p.total_cases))
      .sort((a, b) => a - b);

    medianR0 = r0Values[Math.floor(r0Values.length / 2)];
    medianPeak = peakValues[Math.floor(peakValues.length / 2)];
    medianCases = casesValues[Math.floor(casesValues.length / 2)];
  }

  const minCases = 0.85 * currentCases;
  let maxCases = Math.max(1.25 * currentCases, 1.25 * medianCases);

  if (currentParams.length > 0) {
    maxCases = Math.max(
      maxCases,
      Number(currentParams[0].total_cases)
    );
  }

  const step = Math.max(1, Math.floor((maxCases - minCases) / 20));

  return { medianR0, medianPeak, medianCases, minCases, maxCases, step };
}
