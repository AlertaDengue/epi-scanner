import type { IControl, Map as MaplibreMap } from "maplibre-gl";

export function createLegendControl(
  html: string,
  position: "top-left" | "top-right" | "bottom-left" | "bottom-right" = "bottom-right"
): IControl {
  let container: HTMLDivElement | undefined;

  return {
    onAdd(_map: MaplibreMap) {
      container = document.createElement("div");
      container.className = "maplibregl-ctrl";
      container.style.cssText =
        "background:white;padding:8px;border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,0.3);font-size:12px;max-width:200px";
      container.innerHTML = html;
      return container;
    },
    onRemove() {
      container?.remove();
      container = undefined;
    },
    getDefaultPosition() {
      return position;
    },
  };
}
