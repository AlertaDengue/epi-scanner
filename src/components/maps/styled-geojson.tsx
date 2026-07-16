"use client";

import { useEffect, useRef } from "react";
import { GeoJSON, type GeoJSONProps } from "react-leaflet";
import type L from "leaflet";

type Props = GeoJSONProps & { dataDeps: unknown[] };

export function StyledGeoJSON({ dataDeps, ...props }: Props) {
  const layerRef = useRef<L.GeoJSON | null>(null);

  useEffect(() => {
    if (layerRef.current && props.style) {
      layerRef.current.setStyle(props.style);
    }
  }, dataDeps ?? []);

  return (
    <GeoJSON
      {...props}
      ref={layerRef}
    />
  );
}
