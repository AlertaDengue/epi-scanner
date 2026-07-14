"use client";

import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { GeoJSON } from "@/lib/types";
import { createLegendControl } from "./legend";

interface ModelEvalMapProps {
  geojson: GeoJSON.FeatureCollection;
  rateMap: { code_muni: number; rate: number | null }[];
  year: number;
}

const MODEL_EVAL_COLORS = ["#006aea", "#00b4ca", "#48d085", "#dc7080", "#cb2b2b"];
const BINS = [0.5, 0.95, 1.05, 2];

export function ModelEvalMap({ geojson, rateMap, year }: ModelEvalMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<maplibregl.Map | null>(null);

  useEffect(() => {
    if (!mapContainer.current || !geojson?.features?.length) return;

    if (mapInstance.current) {
      mapInstance.current.remove();
      mapInstance.current = null;
    }

    const rateMapObj = new Map(rateMap.map((r) => [r.code_muni, r.rate]));

    const enrichedGeoJSON = {
      ...geojson,
      features: geojson.features.map((f) => ({
        ...f,
        properties: {
          ...f.properties,
          rate: rateMapObj.get(Number(f.properties?.code_muni)) ?? null,
        },
      })),
    };

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
        data: enrichedGeoJSON as unknown as GeoJSON.FeatureCollection,
      });

      map.addLayer({
        id: "municipalities-fill",
        type: "fill",
        source: "municipalities",
        paint: {
          "fill-color": [
            "match",
            [
              "case",
              ["==", ["get", "rate"], null],
              -1,
              ["<", ["get", "rate"], BINS[0]],
              0,
              ["<", ["get", "rate"], BINS[1]],
              1,
              ["<", ["get", "rate"], BINS[2]],
              2,
              ["<", ["get", "rate"], BINS[3]],
              3,
              4,
            ],
            -1,
            "#d3d3d3",
            0,
            MODEL_EVAL_COLORS[0],
            1,
            MODEL_EVAL_COLORS[1],
            2,
            MODEL_EVAL_COLORS[2],
            3,
            MODEL_EVAL_COLORS[3],
            4,
            MODEL_EVAL_COLORS[4],
            "#d3d3d3",
          ],
          "fill-opacity": 0.6,
        },
      });

      map.addLayer({
        id: "municipalities-border",
        type: "line",
        source: "municipalities",
        paint: {
          "line-color": "#000",
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
        const rate = props.rate as number | null;
        popup
          .setLngLat(
            (e as maplibregl.MapMouseEvent & { features?: maplibregl.MapGeoJSONFeature[] }).lngLat
          )
          .setHTML(
            `<b>${props.name_muni}</b><br/>Rate: ${rate != null ? rate.toFixed(2) : "N/A"}`
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

      const evalLabels: [string, string, string][] = [
        ["< 0.5", MODEL_EVAL_COLORS[0], "Overestimated"],
        ["0.5 - 0.95", MODEL_EVAL_COLORS[1], ""],
        ["0.95 - 1.05", MODEL_EVAL_COLORS[2], "Good"],
        ["1.05 - 2", MODEL_EVAL_COLORS[3], ""],
        ["> 2", MODEL_EVAL_COLORS[4], "Underestimated"],
      ];
      const evalHtml =
        `<b>Model Evaluation</b><br/>
         <small>Observed/Estimated Cases</small><br/>` +
        evalLabels
          .map(
            ([range, color, note]) =>
              `<i style="background:${color};width:18px;height:18px;display:inline-block;margin-right:4px"></i> ${range} ${note ? `(${note})` : ""}`
          )
          .join("<br/>") +
        `<br/><small style="color:grey">* Gray: no epidemic detected</small>`;
      map.addControl(createLegendControl(evalHtml));
    });

    return () => {
      map.remove();
      mapInstance.current = null;
    };
  }, [geojson, rateMap, year]);

  return (
    <div className="rounded-lg border bg-white p-2">
      <h3 className="mb-2 text-sm font-semibold">
        Observed Cases/Estimated Cases by city in {year}
      </h3>
      <div ref={mapContainer} className="h-[400px] w-full rounded" />
    </div>
  );
}
