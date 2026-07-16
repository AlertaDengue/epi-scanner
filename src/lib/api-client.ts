const BASE_URL = process.env.EPISCANNER_API_URL || "http://localhost:8042/api/datastore/episcanner";
const API_KEY = process.env.EPISCANNER_API_KEY || "";

export async function episcannerFetch<T>(
  endpoint: string,
  params: Record<string, string | number | undefined> = {},
  options: RequestInit = {}
): Promise<T> {
  const url = new URL(`${BASE_URL}/${endpoint}/`);

  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") {
      url.searchParams.set(key, String(value));
    }
  }

  const headers: Record<string, string> = {
    "X-UID-Key": API_KEY,
    ...(options.headers as Record<string, string>),
  };

  const res = await fetch(url.toString(), {
    ...options,
    headers,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Episcanner API error ${res.status}: ${text}`);
  }

  return res.json();
}

export function mapCID10ToDisease(cid10: string): string {
  const mapping: Record<string, string> = {
    A90: "dengue",
    "A92.0": "chikungunya",
    "A92.8": "zika",
  };
  return mapping[cid10] || cid10;
}

export function mapDiseaseToCID10(disease: string): string {
  const mapping: Record<string, string> = {
    dengue: "A90",
    chikungunya: "A92.0",
    zika: "A92.8",
  };
  return mapping[disease] || disease;
}
