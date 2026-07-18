const BASE = process.env.NEXT_PUBLIC_URL_PREFIX ?? "";

export function basePath(path: string): string {
  return `${BASE}${path}`;
}
