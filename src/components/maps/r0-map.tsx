"use client";

import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { GeoJSON } from "@/lib/types";
import { createLegendControl } from "./legend";

interface R0MapProps {
  geojson: GeoJSON.FeatureCollection;
  year: number;
}

const R0_STOPS: [number, string][] = [
  [0, "#f7fbff"],
  [0.5, "#d0d1e6"],
  [1, "#74a9cf"],
  [1.5, "#2b8cbe"],
  [2, "#045a8d"],
  [3, "#023858"],
];

export function R0Map({ geojson, year }: R0MapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<maplibregl.Map | null>(null);

  useEffect(() => {
    if (!mapContainer.current || !geojson?.features?.length) return;

    if (mapInstance.current) {
      mapInstance.current.remove();
      mapInstance.current = null;
    }

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {},
        layers: [],
      },
      attributionControl: false,
    });

    mapInstance.current = map;

    map.addControl(new maplibregl.NavigationControl(), "top-left");

    map.on("load", () => {
      map.addSource("municipalities", {
        type: "geojson",
        data: geojson as unknown as GeoJSON.FeatureCollection,
      });

      map.addLayer({
        id: "municipalities-fill",
        type: "fill",
        source: "municipalities",
        paint: {
          "fill-color": [
            "interpolate",
            ["linear"],
            ["coalesce", ["get", "R0"], 0],
            ...R0_STOPS.flat(),
          ],
          "fill-opacity": 0.7,
        },
      });

      map.addLayer({
        id: "municipalities-border",
        type: "line",
        source: "municipalities",
        paint: {
          "line-color": "white",
          "line-width": 1,
        },
      });

      const popup = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false,
      });

      map.on("mouseenter", "municipalities-fill", (e) => {
        map.getCanvas().style.cursor = "pointer";
        const feature = e.features?.[0];
        if (!feature) return;
        const props = feature.properties as Record<string, unknown>;
        const r0 = Number(props.R0) || 0;
        popup
          .setLngLat(
            (e as maplibregl.MapMouseEvent & { features?: maplibregl.MapGeoJSONFeature[] }).lngLat
          )
          .setHTML(
            `<b>${props.name_muni}</b><br/>R0: ${r0.toFixed(2)}`
          )
          .addTo(map);
      });

      map.on("mouseleave", "municipalities-fill", () => {
        map.getCanvas().style.cursor = "";
        popup.remove();
      });

      const bounds = new maplibregl.LngLatBounds();
      for (const feature of geojson.features) {
        if (feature.geometry?.type === "Polygon") {
          for (const coord of feature.geometry.coordinates[0]) {
            bounds.extend(coord as [number, number]);
          }
        } else if (feature.geometry?.type === "MultiPolygon") {
          for (const polygon of feature.geometry.coordinates) {
            for (const coord of polygon[0]) {
              bounds.extend(coord as [number, number]);
            }
          }
        }
      }
      map.fitBounds(bounds, { padding: 20 });

      const legendHtml =
        `<b>R0</b><br/>` +
        R0_STOPS.map(
          ([val, color]) =>
            `<i style="background:${color};width:18px;height:18px;display:inline-block;margin-right:4px"></i> ${val}+`
        ).join("<br/>");
      map.addControl(createLegendControl(legendHtml));
    });

    return () => {
      map.remove();
      mapInstance.current = null;
    };
  }, [geojson, year]);

  return (
    <div className="rounded-lg border bg-white p-2">
      <h3 className="mb-2 text-sm font-semibold">R0 by city in {year}</h3>
      <div ref={mapContainer} className="h-[400px] w-full rounded" />
    </div>
  );
}
