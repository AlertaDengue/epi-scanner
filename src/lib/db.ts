import path from "path";
import fs from "fs";
import { DATA_DIR } from "./constants";

// Simple parquet reader using apache-arrow or manual parsing
// We'll read parquet files using a subprocess call to duckdb CLI or use a JS reader

const parquetCache = new Map<string, Record<string, unknown>[]>();

/**
 * Read a parquet file and return rows as an array of objects.
 * Uses dynamic import for parquetjs to avoid SSR issues.
 */
async function readParquetFile(filePath: string): Promise<Record<string, unknown>[]> {
  const absPath = path.resolve(filePath);

  if (parquetCache.has(absPath)) {
    return parquetCache.get(absPath)!;
  }

  if (!fs.existsSync(absPath)) {
    throw new Error(`File not found: ${absPath}`);
  }

  // Use parquetjs to read the file
  const parquet = await import("parquetjs");
  const reader = await parquet.ParquetReader.openFile(absPath);
  const cursor = reader.getCursor();
  const records: Record<string, unknown>[] = [];

  let record = await cursor.next();
  while (record) {
    records.push(record as Record<string, unknown>);
    record = await cursor.next();
  }

  await reader.close();
  parquetCache.set(absPath, records);
  return records;
}

export function clearCache(): void {
  parquetCache.clear();
}

export async function readAlertData(
  disease: string,
  uf: string
): Promise<Record<string, unknown>[]> {
  const actualDisease = disease === "chik" ? "chikungunya" : disease;
  const filePath = path.join(DATA_DIR, `${uf}_${actualDisease}.parquet`);
  return readParquetFile(filePath);
}

export async function readGeoPackage(
  filePath: string
): Promise<{ code_muni: number; name_muni: string; abbrev_state: string; geometry: unknown }[]> {
  // GeoPackage is SQLite-based. For now, try reading as parquet
  // The actual GPKG reading should be done by the data pipeline converting to GeoJSON
  const geoJsonPath = path.join(DATA_DIR, "muni_br.geojson");
  if (fs.existsSync(geoJsonPath)) {
    const data = JSON.parse(fs.readFileSync(geoJsonPath, "utf-8"));
    return data.features?.map((f: { properties: { code_muni: number; name_muni: string; abbrev_state: string }; geometry: unknown }) => ({
      code_muni: f.properties.code_muni,
      name_muni: f.properties.name_muni,
      abbrev_state: f.properties.abbrev_state,
      geometry: f.geometry,
    })) || [];
  }
  return [];
}

export async function loadGeoJSON(): Promise<GeoJSON.FeatureCollection> {
  const geoJsonPath = path.join(DATA_DIR, "muni_br.geojson");
  if (fs.existsSync(geoJsonPath)) {
    const data = JSON.parse(fs.readFileSync(geoJsonPath, "utf-8"));
    return data as GeoJSON.FeatureCollection;
  }

  // Try reading GPKG and converting
  const gpkgPath = path.join(DATA_DIR, "muni_br.gpkg");
  if (fs.existsSync(gpkgPath)) {
    // GPKG can't be read without spatial extension
    // The data pipeline should convert this to GeoJSON
    console.warn("GPKG file found but no GeoJSON. Run the data pipeline to convert.");
  }

  return { type: "FeatureCollection", features: [] };
}

/**
 * Read DuckDB data from a pre-exported JSON file
 * The data pipeline should export DuckDB tables to JSON
 */
export async function readDuckDBExport(
  disease: string,
  uf: string
): Promise<Record<string, unknown>[]> {
  const jsonPath = path.join(DATA_DIR, `${uf}_params.json`);

  if (fs.existsSync(jsonPath)) {
    const data = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
    return data.filter((row: Record<string, unknown>) => row.disease === disease);
  }

  // Fallback: try parquet
  const parquetPath = path.join(DATA_DIR, `${uf}_params.parquet`);
  if (fs.existsSync(parquetPath)) {
    return readParquetFile(parquetPath);
  }

  return [];
}
