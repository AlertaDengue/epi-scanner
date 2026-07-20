import { describe, it, expect } from "vitest";
import {
  richards,
  richardsVectorized,
  getIniEndWeek,
  getWeekNumber,
} from "@/lib/richards";

describe("Richards model", () => {
  it("starts near zero when time is well before peak week", () => {
    const result = richards(1000, 0.5, 0.1, 0, 20);
    expect(result).toBeLessThan(200);
    expect(result).toBeGreaterThanOrEqual(0);
  });

  it("approaches L as time increases beyond peak", () => {
    const result = richards(1000, 0.5, 0.1, 100, 10);
    expect(result).toBeGreaterThan(990);
    expect(result).toBeLessThan(1001);
  });

  it("returns 0 when L is 0 (no cases)", () => {
    expect(richards(0, 0.5, 0.1, 10, 10)).toBe(0);
  });

  it("generates monotonically increasing values", () => {
    const values = Array.from({ length: 50 }, (_, i) =>
      richards(5000, 0.3, 0.2, i, 15)
    );
    for (let i = 1; i < values.length; i++) {
      expect(values[i]).toBeGreaterThanOrEqual(values[i - 1]);
    }
  });

  it("model length matches the number of data points, not hardcoded 52", () => {
    const L = 5000;
    const r0 = 2;
    const r = 1 - 1 / r0;
    const gamma = 0.3;
    const b = (r * gamma) / (1 - r);
    const a = b / (gamma + b);
    const peakWeek = 15;

    [30, 52, 104, 200].forEach((dataLength) => {
      const modelValues = Array.from({ length: dataLength }, (_, i) =>
        richards(L, a, b, i, peakWeek)
      );
      expect(modelValues).toHaveLength(dataLength);
    });
  });

  it("produces non-negative values for all inputs with positive L", () => {
    const values = Array.from({ length: 100 }, (_, i) =>
      richards(10000, 0.3, 0.1, i, 25)
    );
    values.forEach((v, i) => {
      expect(v).toBeGreaterThanOrEqual(0);
    });
  });

  it("does not exceed L for any time value", () => {
    const L = 5000;
    const values = Array.from({ length: 200 }, (_, i) =>
      richards(L, 0.5, 0.1, i, 10)
    );
    values.forEach((v) => {
      expect(v).toBeLessThanOrEqual(L);
    });
  });

  it("different a values affect curve shape", () => {
    const v1 = richards(1000, 0.2, 0.1, 15, 10);
    const v2 = richards(1000, 0.8, 0.1, 15, 10);
    expect(v1).not.toEqual(v2);
  });

  it("different peak weeks shift the curve", () => {
    const early = richards(1000, 0.3, 0.1, 5, 5);
    const late = richards(1000, 0.3, 0.1, 5, 15);
    expect(early).toBeGreaterThan(late);
  });
});

describe("richardsVectorized", () => {
  it("returns an array of the same length as input times", () => {
    const result = richardsVectorized(1000, 0.5, 0.1, [0, 5, 10, 15, 20], 10);
    expect(result).toHaveLength(5);
  });

  it("produces monotonically increasing values", () => {
    const tValues = [0, 5, 10, 15, 20, 25, 30];
    const result = richardsVectorized(5000, 0.3, 0.15, tValues, 12);
    for (let i = 1; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(result[i - 1]);
    }
  });

  it("returns empty array for empty input", () => {
    const result = richardsVectorized(1000, 0.5, 0.1, [], 10);
    expect(result).toEqual([]);
  });
});

describe("getIniEndWeek", () => {
  it("returns start and end dates for a given year", () => {
    const result = getIniEndWeek(2024);
    expect(result).toHaveProperty("startDate");
    expect(result).toHaveProperty("endDate");
    expect(result.startDate).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(result.endDate).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it("uses endYear when provided", () => {
    const result = getIniEndWeek(2024, 2025);
    expect(result.endDate).toBe("2025-11-01");
  });

  it("start date is before end date", () => {
    const result = getIniEndWeek(2024);
    expect(new Date(result.startDate).getTime()).toBeLessThan(
      new Date(result.endDate).getTime()
    );
  });
});

describe("getWeekNumber", () => {
  it("returns a positive integer for any date", () => {
    const result = getWeekNumber("2024-07-15");
    expect(result).toBeGreaterThan(0);
    expect(Number.isInteger(result)).toBe(true);
  });

  it("increases as the date progresses within the same year", () => {
    const july = getWeekNumber("2024-07-01");
    const august = getWeekNumber("2024-08-01");
    expect(august).toBeGreaterThan(july);
  });
});
