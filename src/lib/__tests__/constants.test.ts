import { describe, it, expect } from "vitest";
import { CURRENT_YEAR, MIN_YEAR, STATES, DISEASES } from "@/lib/constants";

describe("constants", () => {
  it("CURRENT_YEAR is a valid year", () => {
    expect(CURRENT_YEAR).toBeGreaterThanOrEqual(2020);
    expect(CURRENT_YEAR).toBeLessThanOrEqual(2100);
    expect(Number.isInteger(CURRENT_YEAR)).toBe(true);
  });

  it("MIN_YEAR is before CURRENT_YEAR", () => {
    expect(MIN_YEAR).toBeLessThan(CURRENT_YEAR);
    expect(Number.isInteger(MIN_YEAR)).toBe(true);
  });

  it("STATES contains all 27 Brazilian states", () => {
    expect(Object.keys(STATES)).toHaveLength(27);
    expect(STATES.RJ).toBe("Rio de Janeiro");
    expect(STATES.SP).toBe("São Paulo");
    expect(STATES.DF).toBe("Distrito Federal");
  });

  it("DISEASES is an object with disease names", () => {
    expect(DISEASES).toBeDefined();
    expect(typeof DISEASES).toBe("object");
  });
});
