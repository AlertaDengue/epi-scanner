export const STATES: Record<string, string> = {
  AC: "Acre",
  AL: "Alagoas",
  AM: "Amazonas",
  AP: "Amapá",
  BA: "Bahia",
  CE: "Ceará",
  DF: "Distrito Federal",
  ES: "Espírito Santo",
  GO: "Goiás",
  MA: "Maranhão",
  MG: "Minas Gerais",
  MS: "Mato Grosso do Sul",
  MT: "Mato Grosso",
  PA: "Pará",
  PB: "Paraíba",
  PE: "Pernambuco",
  PI: "Piauí",
  PR: "Paraná",
  RJ: "Rio de Janeiro",
  RN: "Rio Grande do Norte",
  RO: "Rondônia",
  RR: "Roraima",
  RS: "Rio Grande do Sul",
  SC: "Santa Catarina",
  SE: "Sergipe",
  SP: "São Paulo",
  TO: "Tocantins",
};

export const STATE_CHOICES = Object.entries(STATES).map(([code, name]) => ({
  value: code,
  label: name,
}));

export const DISEASES = [
  { value: "dengue", label: "Dengue" },
  { value: "chikungunya", label: "Chikungunya" },
];

export const DISEASE_SUFFIX: Record<string, string> = {
  dengue: "",
  chikungunya: "_chik",
  zika: "_zika",
};

export const DUCKDB_FILE = process.env.CTNR_EPISCANNER_DUCKDB_DIR
  ? `${process.env.CTNR_EPISCANNER_DUCKDB_DIR}/episcanner.duckdb`
  : null;

export const DATA_DIR = process.env.CTNR_EPISCANNER_DATA_DIR || "./data";

export const MODEL_EVAL_COLORS = [
  "#006aea",
  "#00b4ca",
  "#48d085",
  "#dc7080",
  "#cb2b2b",
];

export const MODEL_EVAL_BINS = [0.5, 0.95, 1.05, 2];

export const CURRENT_YEAR = new Date().getFullYear();

export const MIN_YEAR = 2010;
