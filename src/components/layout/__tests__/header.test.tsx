import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { DashboardHeader } from "@/components/layout/header";

describe("DashboardHeader", () => {
  it("renders the title", () => {
    render(<DashboardHeader />);
    expect(screen.getByText("Real-time Epidemic Scanner")).toBeInTheDocument();
  });

  it("renders the subtitle with Infodengue", () => {
    render(<DashboardHeader />);
    expect(screen.getByText(/Infodengue/)).toBeInTheDocument();
  });

  it("renders the Live surveillance badge", () => {
    render(<DashboardHeader />);
    expect(screen.getByText("Live surveillance")).toBeInTheDocument();
  });

  it("renders logo image with GitHub URL", () => {
    render(<DashboardHeader />);
    const img = screen.getByAltText("Infodengue");
    expect(img).toBeInTheDocument();
    expect(img.getAttribute("src")).toContain("github.com");
  });
});
