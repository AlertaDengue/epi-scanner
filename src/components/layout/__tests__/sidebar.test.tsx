import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { DashboardSidebar } from "@/components/layout/sidebar";

const baseProps = {
  disease: "dengue",
  state: "RJ",
  city: "3304557",
  cities: [{ geocode: 3304557, name: "Rio de Janeiro" }],
  topCities: [
    { name_muni: "Rio de Janeiro", transmissao: 20, geocode: 3304557 },
    { name_muni: "Niterói", transmissao: 15, geocode: 3303302 },
  ],
  onDiseaseChange: () => {},
  onStateChange: () => {},
  onCityChange: () => {},
};

describe("DashboardSidebar", () => {
  it("renders top cities list", () => {
    render(<DashboardSidebar {...baseProps} />);
    expect(screen.getByText("Top 20 most active cities")).toBeInTheDocument();
    expect(screen.getByText("Rio de Janeiro")).toBeInTheDocument();
  });

  it("shows spinner when loading", () => {
    render(<DashboardSidebar {...baseProps} loading={true} />);
    const spinners = document.querySelectorAll(".animate-spin");
    expect(spinners.length).toBeGreaterThan(0);
  });

  it("does not show spinner when not loading", () => {
    render(<DashboardSidebar {...baseProps} loading={false} />);
    const spinners = document.querySelectorAll(".animate-spin");
    expect(spinners.length).toBe(0);
  });
});
