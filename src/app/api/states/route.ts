import { cachedJson } from "@/lib/cache";
import { episcannerFetch } from "@/lib/api-client";

interface DjangoState {
  code: string;
  name: string;
}

export async function GET() {
  const states = await episcannerFetch<DjangoState[]>("states");
  return cachedJson(states);
}
