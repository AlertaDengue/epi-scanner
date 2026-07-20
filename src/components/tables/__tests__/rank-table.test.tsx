import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { RankTable } from "@/components/tables/rank-table";

describe("RankTable", () => {
  const rows = [
    { name: "City A", value: 100 },
    { name: "City B", value: 80 },
    { name: "City C", value: 60 },
  ];

  it("renders all rows", () => {
    render(
      <RankTable
        rows={rows}
        valueLabel="Weeks"
        format={(v) => Math.round(v).toString()}
      />
    );
    expect(screen.getByText("City A")).toBeInTheDocument();
    expect(screen.getByText("City B")).toBeInTheDocument();
    expect(screen.getByText("City C")).toBeInTheDocument();
  });

  it("renders formatted values", () => {
    render(
      <RankTable
        rows={rows}
        valueLabel="Weeks"
        format={(v) => v.toFixed(1)}
      />
    );
    expect(screen.getByText("100.0")).toBeInTheDocument();
    expect(screen.getByText("80.0")).toBeInTheDocument();
  });

  it("renders with bar color", () => {
    const { container } = render(
      <RankTable
        rows={rows}
        valueLabel="R₀"
        format={(v) => v.toFixed(2)}
        barColor="red"
      />
    );
    expect(container.querySelector(".bg-red")).toBeDefined();
  });

  it("renders index numbers", () => {
    render(
      <RankTable
        rows={rows}
        valueLabel="Count"
        format={(v) => v.toString()}
      />
    );
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("handles empty rows", () => {
    const { container } = render(
      <RankTable
        rows={[]}
        valueLabel="Empty"
        format={(v) => v.toString()}
      />
    );
    expect(container.querySelector("ul")).toBeInTheDocument();
  });
});
