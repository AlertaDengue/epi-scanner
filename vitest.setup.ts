import "@testing-library/jest-dom/vitest";

// Mock ResizeObserver for Recharts
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Mock SVGElement methods used by Recharts
if (typeof (SVGElement.prototype as unknown as Record<string, unknown>).getBBox === "undefined") {
  (SVGElement.prototype as unknown as Record<string, unknown>).getBBox = () => ({
    x: 0,
    y: 0,
    width: 100,
    height: 100,
  });
}

// Mock scrollTo for Base UI components
Element.prototype.scrollTo = () => {};
