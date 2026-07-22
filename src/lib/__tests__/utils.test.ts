import { describe, it, expect } from "vitest";
import { formatDateISO, cdcEpiweekStart } from "@/lib/utils";

describe("formatDateISO", () => {
  it("formats a date as YYYY-MM-DD in UTC", () => {
    const date = new Date(Date.UTC(2026, 0, 15));
    expect(formatDateISO(date)).toBe("2026-01-15");
  });

  it("pads single-digit months and days", () => {
    const date = new Date(Date.UTC(2026, 2, 5));
    expect(formatDateISO(date)).toBe("2026-03-05");
  });
});

describe("cdcEpiweekStart", () => {
  it("week 1 of 2024 starts on Sun Dec 31 2023", () => {
    const d = cdcEpiweekStart(2024, 1);
    expect(formatDateISO(d)).toBe("2023-12-31");
  });

  it("week 1 of 2025 starts on Sun Dec 29 2024", () => {
    const d = cdcEpiweekStart(2025, 1);
    expect(formatDateISO(d)).toBe("2024-12-29");
  });

  it("week 45 of 2025 starts on Sun Nov 2 2025", () => {
    const d = cdcEpiweekStart(2025, 45);
    expect(formatDateISO(d)).toBe("2025-11-02");
  });

  it("week 45 of 2026 starts on Sun Nov 8 2026", () => {
    const d = cdcEpiweekStart(2026, 45);
    expect(formatDateISO(d)).toBe("2026-11-08");
  });

  it("epi year 2026 runs from 2025-11-02 to 2026-11-08", () => {
    const start = cdcEpiweekStart(2025, 45);
    const end = cdcEpiweekStart(2026, 45);
    expect(formatDateISO(start)).toBe("2025-11-02");
    expect(formatDateISO(end)).toBe("2026-11-08");
  });
});
