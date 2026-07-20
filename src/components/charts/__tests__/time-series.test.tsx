import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { TimeSeriesChart } from "@/components/charts/time-series";

function makeData(length = 30) {
  return Array.from({ length }, (_, i) => {
    const d = new Date(2024, 0, 1);
    d.setDate(d.getDate() + i * 7);
    return {
      date: d.toISOString().split("T")[0],
      casos: 10 + i * 5,
      casos_cum: (i + 1) * 100,
    };
  });
}

describe("TimeSeriesChart", () => {
  it("renders with required props", () => {
    render(<TimeSeriesChart data={makeData()} title="Test Chart" />);
    expect(screen.getByText("Test Chart")).toBeInTheDocument();
  });

  it("renders the responsive chart container", () => {
    const { container } = render(
      <TimeSeriesChart data={makeData()} title="Chart" />
    );
    expect(container.querySelector(".recharts-responsive-container")).toBeInTheDocument();
  });

  it("accepts peakWeekDate prop without crashing", () => {
    const { container } = render(
      <TimeSeriesChart
        data={makeData()}
        peakWeekDate="2024-06-15"
        title="With Peak"
      />
    );
    expect(container.querySelector(".recharts-responsive-container")).toBeInTheDocument();
  });

  it("accepts startDate prop without crashing", () => {
    const { container } = render(
      <TimeSeriesChart
        data={makeData()}
        startDate="2024-01-01"
        title="With Start"
      />
    );
    expect(container.querySelector(".recharts-responsive-container")).toBeInTheDocument();
  });

  it("accepts endDate prop without crashing", () => {
    const { container } = render(
      <TimeSeriesChart
        data={makeData()}
        endDate="2024-12-31"
        title="With End"
      />
    );
    expect(container.querySelector(".recharts-responsive-container")).toBeInTheDocument();
  });

  it("accepts all three date markers simultaneously", () => {
    const { container } = render(
      <TimeSeriesChart
        data={makeData()}
        startDate="2024-03-01"
        peakWeekDate="2024-06-15"
        endDate="2024-09-30"
        title="All Markers"
      />
    );
    expect(container.querySelector(".recharts-responsive-container")).toBeInTheDocument();
  });

  it("does not crash when markers are null", () => {
    const { container } = render(
      <TimeSeriesChart data={makeData()} title="No Markers" />
    );
    expect(container.querySelector(".recharts-responsive-container")).toBeInTheDocument();
  });

  it("renders model data without crashing", () => {
    const data = makeData(20);
    const modelData = data.map((d) => ({ date: d.date, model: 500 }));
    const { container } = render(
      <TimeSeriesChart
        data={data}
        modelData={modelData}
        title="With Model"
      />
    );
    expect(container.querySelector(".recharts-responsive-container")).toBeInTheDocument();
  });

  it("renders title correctly when empty string", () => {
    const { container } = render(
      <TimeSeriesChart data={makeData()} title="" />
    );
    expect(container.querySelector(".recharts-responsive-container")).toBeInTheDocument();
  });
});
