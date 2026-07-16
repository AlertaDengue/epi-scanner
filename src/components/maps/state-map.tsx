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

interface StateMapData {
  geocode: number;
  transmissao: number;
}

interface StateMapProps {
  data: StateMapData[];
  title: string;
  uf: string;
}

const COLORS = ["#f7fbff", "#d0d1e6", "#74a9cf", "#2b8cbe", "#045a8d"];

function getColor(value: number, max: number): string {
  const ratio = max > 0 ? value / max : 0;
  if (ratio < 0.2) return COLORS[0];
  if (ratio < 0.4) return COLORS[1];
  if (ratio < 0.6) return COLORS[2];
  if (ratio < 0.8) return COLORS[3];
  return COLORS[4];
}

export function StateMap({ data, title, uf }: StateMapProps) {
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
        {title && <h3 className="mb-2 text-sm font-semibold">{title}</h3>}
        <div className="flex h-[400px] items-center justify-center text-sm text-red-600">
          {error}
        </div>
      </div>
    );
  }

  if (!geojson || !bounds) {
    return (
      <div className="rounded-lg border bg-white p-2">
        {title && <h3 className="mb-2 text-sm font-semibold">{title}</h3>}
        <div className="flex h-[400px] items-center justify-center">
          <Spinner className="size-8" />
        </div>
      </div>
    );
  }

  const maxVal = Math.max(...data.map((d) => d.transmissao), 1);
  const dataMap = new Map(data.map((d) => [d.geocode, d.transmissao]));

  const style = (feature: GeoJSON.Feature | undefined) => {
    if (!feature || !feature.properties) return {};
    const code = feature.properties.code_muni as number;
    const transmissao = dataMap.get(code) ?? 0;
    return {
      fillColor: getColor(transmissao, maxVal),
      weight: 1,
      opacity: 1,
      color: "white",
      fillOpacity: 0.7,
    };
  };

  const onEachFeature = (feature: GeoJSON.Feature, layer: L.Layer) => {
    if (!feature.properties) return;
    const name = feature.properties.name_muni as string;
    const code = feature.properties.code_muni as number;
    const transmissao = dataMap.get(code) ?? 0;
    layer.bindPopup(`<b>${name}</b><br/>Weeks: ${transmissao}`);
  };

  return (
    <div className="rounded-lg border bg-white p-2">
      {title && <h3 className="mb-2 text-sm font-semibold">{title}</h3>}
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
            key={`${uf}-${data.map((d) => d.geocode).slice(0, 3).join(",")}`}
            data={geojson}
            style={style}
            onEachFeature={onEachFeature}
            dataDeps={[data]}
          />
        </MapContainer>
        <div
          className="absolute bottom-3 left-3 z-[1010] rounded bg-white px-2 py-1.5 text-xs shadow-md"
          style={{ maxWidth: "200px" }}
        >
          <div className="mb-1 font-semibold">Weeks</div>
          {COLORS.map((color, i) => (
            <div key={i} className="flex items-center gap-1">
              <span
                className="inline-block h-3 w-3 rounded-sm"
                style={{ background: color }}
              />
              <span>{Math.round((maxVal / COLORS.length) * i)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
