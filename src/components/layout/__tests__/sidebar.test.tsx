import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DashboardSidebar } from "@/components/layout/sidebar";

function makeTopCities(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    name_muni: `City ${i + 1}`,
    transmissao: 20 - i,
    geocode: 3300000 + i,
  }));
}

const baseProps = {
  disease: "dengue",
  state: "RJ",
  city: "3304557",
  cities: [{ geocode: 3304557, name: "Rio de Janeiro" }],
  topCities: makeTopCities(20),
  onDiseaseChange: () => {},
  onStateChange: () => {},
  onCityChange: () => {},
  epiYear: "all",
  onEpiYearChange: () => {},
  epiYears: [2024, 2025],
};

describe("DashboardSidebar", () => {
  it("renders top cities list", () => {
    render(<DashboardSidebar {...baseProps} />);
    expect(screen.getByText("Top 20 most active cities")).toBeInTheDocument();
    expect(screen.getByText("City 1")).toBeInTheDocument();
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

  it("shows only 5 cities by default", () => {
    render(<DashboardSidebar {...baseProps} />);
    expect(screen.getByText("City 1")).toBeInTheDocument();
    expect(screen.getByText("City 5")).toBeInTheDocument();
    expect(screen.queryByText("City 6")).not.toBeInTheDocument();
  });

  it("expands to show all 20 cities", async () => {
    const user = userEvent.setup();
    render(<DashboardSidebar {...baseProps} />);
    const btn = screen.getByText(/Show all 20/);
    await user.click(btn);
    expect(screen.getByText("City 20")).toBeInTheDocument();
    expect(screen.getByText("Show less")).toBeInTheDocument();
  });

  it("collapses back to 5 cities", async () => {
    const user = userEvent.setup();
    render(<DashboardSidebar {...baseProps} />);
    await user.click(screen.getByText(/Show all 20/));
    await user.click(screen.getByText("Show less"));
    expect(screen.queryByText("City 6")).not.toBeInTheDocument();
  });

  it("renders disease options", () => {
    render(<DashboardSidebar {...baseProps} />);
    expect(screen.getByText("Select disease")).toBeInTheDocument();
    expect(screen.getByText("Select state")).toBeInTheDocument();
  });

  it("renders epidemic year selector", () => {
    render(<DashboardSidebar {...baseProps} />);
    expect(screen.getByText("Epidemic year")).toBeInTheDocument();
  });

  it("shows epidemic year selector even with empty years (All option always visible)", () => {
    render(<DashboardSidebar {...baseProps} epiYears={[]} />);
    expect(screen.getByText("Epidemic year")).toBeInTheDocument();
  });

  it("calls onCityChange when a city is clicked", async () => {
    const user = userEvent.setup();
    const onCityChange = vi.fn();
    render(<DashboardSidebar {...baseProps} onCityChange={onCityChange} />);
    await user.click(screen.getByText("City 1"));
    expect(onCityChange).toHaveBeenCalledWith("3300000");
  });

  it("does not show expand button when 5 or fewer cities", () => {
    const fewCities = makeTopCities(3);
    render(<DashboardSidebar {...baseProps} topCities={fewCities} />);
    expect(screen.queryByText(/Show all/)).not.toBeInTheDocument();
  });

  it("renders city selector label", () => {
    render(<DashboardSidebar {...baseProps} />);
    expect(screen.getByText("Select city")).toBeInTheDocument();
  });

  it("shows selected city name in trigger", () => {
    render(<DashboardSidebar {...baseProps} state="SP" cities={[{ geocode: 3509502, name: "Campinas" }]} city="3509502" />);
    expect(screen.getByText("Campinas")).toBeInTheDocument();
  });

  it("shows default text when no city selected", () => {
    render(<DashboardSidebar {...baseProps} city="" />);
    expect(screen.getByText("Search city...")).toBeInTheDocument();
  });

  it("shows disease label instead of value", () => {
    render(<DashboardSidebar {...baseProps} disease="chikungunya" />);
    expect(screen.getByText("Chikungunya")).toBeInTheDocument();
  });

  it("shows state label instead of code", () => {
    render(<DashboardSidebar {...baseProps} state="SP" city="" />);
    expect(screen.getByText("São Paulo")).toBeInTheDocument();
  });
});
