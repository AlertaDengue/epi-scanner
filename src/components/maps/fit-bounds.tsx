"use client";

import { useEffect } from "react";
import { useMap } from "react-leaflet";
import type L from "leaflet";

export function FitBounds({ bounds }: { bounds: L.LatLngBounds }) {
  const map = useMap();

  useEffect(() => {
    if (!bounds || !bounds.isValid()) return;
    map.whenReady(() => {
      if (!bounds.isValid()) return;
      requestAnimationFrame(() => {
        map.fitBounds(bounds, { padding: [10, 10], maxZoom: 10, animate: false });
      });
    });
  }, [map, bounds]);

  return null;
}
