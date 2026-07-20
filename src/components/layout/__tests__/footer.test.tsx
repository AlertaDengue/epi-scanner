import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Footer } from "@/components/layout/footer";

describe("Footer", () => {
  it("renders Infodengue link", () => {
    render(<Footer />);
    expect(screen.getByText("Infodengue")).toBeInTheDocument();
  });

  it("renders Mosqlimate link", () => {
    render(<Footer />);
    expect(screen.getByText("Mosqlimate")).toBeInTheDocument();
  });

  it("renders EpiGraphHub link", () => {
    render(<Footer />);
    expect(screen.getByText("EpiGraphHub")).toBeInTheDocument();
  });

  it("renders EpiScanner Article link", () => {
    render(<Footer />);
    const link = screen.getByText("EpiScanner Article");
    expect(link).toBeInTheDocument();
    expect(link.getAttribute("href")).toContain("royalsocietypublishing.org");
  });

  it("renders the copyright year", () => {
    render(<Footer />);
    const year = new Date().getFullYear().toString();
    expect(screen.getByText(new RegExp(year))).toBeInTheDocument();
  });
});
