"use client";

import { useMap } from "react-leaflet";
import type L from "leaflet";

interface RecenterButtonProps {
  bounds: L.LatLngBounds;
}

export function RecenterButton({ bounds }: RecenterButtonProps) {
  const map = useMap();

  return (
    <button
      type="button"
      className="absolute left-[3.5rem] top-3 z-[1010] rounded bg-white p-1.5 shadow-md hover:bg-gray-50"
      onClick={() => { if (bounds?.isValid()) map.whenReady(() => map.fitBounds(bounds, { padding: [10, 10], maxZoom: 10, animate: true })); }}
      title="Recenter"
      aria-label="Recenter map"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <circle cx="12" cy="12" r="3" />
      </svg>
    </button>
  );
}
