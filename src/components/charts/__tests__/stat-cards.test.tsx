import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { StatCards } from "@/components/charts/stat-cards";

describe("StatCards", () => {
  const baseProps = {
    cumulativeCases: 50000,
    topR0: 1.8,
    peakYear: 2024,
    locationLabel: "State under scan",
    locationValue: "RJ",
    locationHint: "Municipal-level resolution",
  };

  it("renders all four stat cards", () => {
    render(<StatCards {...baseProps} />);
    expect(screen.getByText("Cumulative notified cases")).toBeInTheDocument();
    expect(screen.getByText("Highest current R₀")).toBeInTheDocument();
    expect(screen.getByText("Monitored season")).toBeInTheDocument();
    expect(screen.getByText("State under scan")).toBeInTheDocument();
  });

  it("shows formatted values when not loading", () => {
    render(<StatCards {...baseProps} />);
    expect(screen.getByText("50,000")).toBeInTheDocument();
    expect(screen.getByText("1.80")).toBeInTheDocument();
    expect(screen.getByText("2024")).toBeInTheDocument();
    expect(screen.getByText("RJ")).toBeInTheDocument();
  });

  it("shows loading spinners when loading prop is true", () => {
    render(<StatCards {...baseProps} loading={true} />);
    // Should have spinners (each card shows Loader2)
    const spinners = document.querySelectorAll(".animate-spin");
    expect(spinners.length).toBeGreaterThanOrEqual(3); // 3 data-dependent cards
  });

  it("does not show spinners when loading is false", () => {
    render(<StatCards {...baseProps} loading={false} />);
    const spinners = document.querySelectorAll(".animate-spin");
    expect(spinners.length).toBe(0);
  });
});
