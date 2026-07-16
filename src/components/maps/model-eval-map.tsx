"use client";

import { useEffect, useState } from "react";
import { MapContainer } from "react-leaflet";
import { StyledGeoJSON } from "./styled-geojson";
import { FitBounds } from "./fit-bounds";
import { RecenterButton } from "./recenter-button";
import L from "leaflet";
import { Spinner } from "@/components/ui/spinner";
import "leaflet/dist/leaflet.css";
import "@/lib/leaflet-config";

interface ModelEvalMapData {
  geocode: number;
  rate: number | null;
}

interface ModelEvalMapProps {
  data: ModelEvalMapData[];
  year: number;
  uf: string;
}

const MODEL_EVAL_COLORS = ["#006aea", "#00b4ca", "#48d085", "#dc7080", "#cb2b2b"];

function getEvalColor(rate: number | null): string {
  if (rate === null) return "#d3d3d3";
  if (rate < 0.5) return MODEL_EVAL_COLORS[0];
  if (rate < 0.95) return MODEL_EVAL_COLORS[1];
  if (rate < 1.05) return MODEL_EVAL_COLORS[2];
  if (rate < 2) return MODEL_EVAL_COLORS[3];
  return MODEL_EVAL_COLORS[4];
}

export function ModelEvalMap({ data, year, uf }: ModelEvalMapProps) {
  const [geojson, setGeojson] = useState<GeoJSON.FeatureCollection | null>(null);
  const [bounds, setBounds] = useState<L.LatLngBounds | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/states/${uf}.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load GeoJSON for ${uf}`);
        return r.json();
      })
      .then((geo) => {
        setGeojson(geo);
        try {
          const layer = L.geoJSON(geo);
          const b = layer.getBounds();
          if (b.isValid()) setBounds(b);
          else setBounds(new L.LatLngBounds([-33.75, -73.99], [5.27, -32.38]));
        } catch {
          setBounds(new L.LatLngBounds([-33.75, -73.99], [5.27, -32.38]));
        }
        setError(null);
      })
      .catch((err) => {
        console.error("Error loading map data:", err);
        setError("Failed to load map data");
      });
  }, [uf]);

  if (error) {
    return (
      <div className="rounded-lg border bg-white p-2">
        <h3 className="mb-2 text-sm font-semibold">
          Observed Cases/Estimated Cases by city in {year}
        </h3>
        <div className="flex h-[400px] items-center justify-center text-sm text-red-600">
          {error}
        </div>
      </div>
    );
  }

  if (!geojson || !bounds) {
    return (
      <div className="rounded-lg border bg-white p-2">
        <h3 className="mb-2 text-sm font-semibold">
          Observed Cases/Estimated Cases by city in {year}
        </h3>
        <div className="flex h-[400px] items-center justify-center">
          <Spinner className="size-8" />
        </div>
      </div>
    );
  }

  const dataMap = new Map(data.map((d) => [d.geocode, d.rate]));

  const style = (feature: GeoJSON.Feature | undefined) => {
    if (!feature || !feature.properties) return {};
    const code = feature.properties.code_muni as number;
    const rate = dataMap.get(code) ?? null;
    return {
      fillColor: getEvalColor(rate),
      weight: 1,
      opacity: 1,
      color: "#000",
      fillOpacity: 0.6,
    };
  };

  const onEachFeature = (feature: GeoJSON.Feature, layer: L.Layer) => {
    if (!feature.properties) return;
    const name = feature.properties.name_muni as string;
    const code = feature.properties.code_muni as number;
    const rate = dataMap.get(code) ?? null;
    const rateText = rate !== null ? rate.toFixed(2) : "N/A";
    layer.bindPopup(`<b>${name}</b><br/>Rate: ${rateText}`);
  };

  const evalLabels: [string, string, string][] = [
    ["< 0.5", MODEL_EVAL_COLORS[0], "Overestimated"],
    ["0.5 - 0.95", MODEL_EVAL_COLORS[1], ""],
    ["0.95 - 1.05", MODEL_EVAL_COLORS[2], "Good"],
    ["1.05 - 2", MODEL_EVAL_COLORS[3], ""],
    ["> 2", MODEL_EVAL_COLORS[4], "Underestimated"],
  ];

  return (
    <div className="rounded-lg border bg-white p-2">
      <h3 className="mb-2 text-sm font-semibold">
        Observed Cases/Estimated Cases by city in {year}
      </h3>
      <div className="relative z-0">
        <MapContainer key={uf}
          bounds={bounds}
          className="h-[400px] w-full rounded"
          scrollWheelZoom={true}
          zoomSnap={0.25}
          zoomDelta={0.5}
          zoomControl={true}
          attributionControl={false}
        >
          <FitBounds bounds={bounds} />
          <RecenterButton bounds={bounds} />
          <StyledGeoJSON
            key={`${uf}-${year}-${data.map((d) => d.geocode).slice(0, 3).join(",")}`}
            data={geojson}
            style={style}
            onEachFeature={onEachFeature}
            dataDeps={[data]}
          />
        </MapContainer>
        <div
          className="absolute bottom-3 left-3 z-[1010] rounded bg-white px-2 py-1.5 text-xs shadow-md"
          style={{ maxWidth: "220px" }}
        >
          <div className="mb-1 font-semibold">Model Evaluation</div>
          <div className="mb-1 text-[10px] text-gray-600">Observed/Estimated Cases</div>
          {evalLabels.map(([range, color, note]) => (
            <div key={range} className="flex items-center gap-1">
              <span
                className="inline-block h-3 w-3 rounded-sm"
                style={{ background: color }}
              />
              <span>
                {range} {note ? `(${note})` : ""}
              </span>
            </div>
          ))}
          <div className="mt-1 text-[10px] text-gray-500">* Gray: no epidemic detected</div>
        </div>
      </div>
    </div>
  );
}
