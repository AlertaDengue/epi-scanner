export interface AlertData {
  id?: number;
  municipio_geocodigo: number;
  municipio_nome?: string;
  data_iniSE: string;
  SE: number;
  casos: number;
  transmissao: number;
  pop?: number;
  tweet?: number;
  rt?: number;
  epi?: number;
  pr?: number;
  epi_pr?: number;
  p_rt1?: number;
  p_sm1?: number;
  level?: string;
}

export interface SIRParams {
  geocode: number;
  year: number;
  disease: string;
  beta: number;
  gamma: number;
  R0: number;
  peak_week: number;
  ep_pw: number;
  ep_ini: number;
  ep_end: number;
  ep_dur: number;
  total_cases: number;
}

export interface CityInfo {
  code_muni: number;
  name_muni: string;
  abbrev_state: string;
  geometry?: unknown;
}

export interface GeoFeature {
  type: "Feature";
  geometry: unknown;
  properties: {
    code_muni: number;
    name_muni: string;
    abbrev_state: string;
    transmissao?: number;
    R0?: number;
    rate?: number;
    observed_cases?: number;
    total_cases?: number;
    year?: number;
    [key: string]: unknown;
  } | null;
}

export interface GeoJSON {
  type: "FeatureCollection";
  features: GeoFeature[];
}

export interface WeeksMapData extends GeoFeature {
  properties: {
    code_muni: number;
    name_muni: string;
    abbrev_state: string;
    transmissao: number;
  };
}

export interface TimeSeriesPoint {
  data_iniSE: string;
  casos: number;
  casos_cum: number;
  municipio_geocodigo: number;
}

export interface TopCity {
  name_muni: string;
  transmissao: number;
  code_muni: number;
}

export interface TopR0City {
  name: string;
  geocode: number;
  R0: number;
}

export interface ModelEvalBin {
  range: string;
  count: number;
  percentage: number;
}

export interface EpidemicCalcParams {
  disease: string;
  city: string;
  geocode: number;
  peakWeek: number;
  R0: number;
  totalCases: number;
  startDate: string;
}

export interface EpidemicCalcResult {
  dates: string[];
  dataCumulative: number[];
  modelCumulative: number[];
  peakWeekDate: string | null;
}

export interface MapWithRate {
  name_muni: string;
  geometry: unknown;
  observed_cases: number;
  total_cases: number;
  rate: number;
  code_muni?: number;
}
