"use client";

import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { GeoJSON } from "@/lib/types";
import { createLegendControl } from "./legend";

interface StateMapProps {
  geojson: GeoJSON.FeatureCollection;
  title: string;
  colorField: string;
  colorScheme?: string;
  legendTitle?: string;
}

const COLOR_STOPS = ["#f7fbff", "#d0d1e6", "#74a9cf", "#2b8cbe", "#045a8d"];

export function StateMap({
  geojson,
  title,
  colorField,
  legendTitle = "Weeks",
}: StateMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<maplibregl.Map | null>(null);

  useEffect(() => {
    if (!mapContainer.current || !geojson?.features?.length) return;

    if (mapInstance.current) {
      mapInstance.current.remove();
      mapInstance.current = null;
    }

    const values = geojson.features.map(
      (f) => Number(f.properties?.[colorField]) || 0
    );
    const maxVal = Math.max(...values, 1);

    const colorStops: [number, string][] = COLOR_STOPS.map((color, i) => [
      (maxVal / COLOR_STOPS.length) * i,
      color,
    ]);

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
            ["coalesce", ["get", colorField], 0],
            ...colorStops.flat(),
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
        const value = Number(props[colorField]) || 0;
        popup
          .setLngLat(
            (e as maplibregl.MapMouseEvent & { features?: maplibregl.MapGeoJSONFeature[] }).lngLat
          )
          .setHTML(
            `<b>${props.name_muni}</b><br/>${legendTitle}: ${Math.round(value)}`
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

      const steps = COLOR_STOPS.length;
      const legendLabels: string[] = [];
      for (let i = 0; i < steps; i++) {
        const val = (maxVal / steps) * i;
        legendLabels.push(
          `<i style="background:${COLOR_STOPS[i]};width:18px;height:18px;display:inline-block;margin-right:4px"></i> ${Math.round(val)}`
        );
      }
      const legendHtml = `<b>${legendTitle}</b><br/>${legendLabels.join("<br/>")}`;
      map.addControl(createLegendControl(legendHtml));
    });

    return () => {
      map.remove();
      mapInstance.current = null;
    };
  }, [geojson, colorField, legendTitle]);

  return (
    <div className="rounded-lg border bg-white p-2">
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      <div ref={mapContainer} className="h-[400px] w-full rounded" />
    </div>
  );
}
