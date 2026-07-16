import type { GeoJsonObject } from "geojson";

export function filterBoundsGeo(geo: Record<string, unknown>, uf: string): GeoJsonObject {
  if (uf !== "ES") return geo as unknown as GeoJsonObject;
  const features = (geo.features as Array<{ properties?: Record<string, unknown> }>) || [];
  return {
    ...geo,
    features: features.filter((f) => f.properties?.code_muni !== 3205309),
  } as unknown as GeoJsonObject;
}
