import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { Slider } from "@/components/ui/slider";

describe("Slider", () => {
  it("does not have overflow-hidden on the track element", () => {
    const { container } = render(
      <Slider value={[100]} min={0} max={100} />
    );
    const tracks = container.querySelectorAll('[data-slot="slider-track"]');
    expect(tracks.length).toBeGreaterThan(0);
    tracks.forEach((track) => {
      expect(track.className).not.toContain("overflow-hidden");
    });
  });

  it("renders a track element with rounded-full", () => {
    const { container } = render(
      <Slider value={[50]} min={0} max={100} />
    );
    const track = container.querySelector('[data-slot="slider-track"]');
    expect(track).toBeInTheDocument();
    expect(track?.className).toContain("rounded-full");
  });

  it("renders an indicator element with rounded-full", () => {
    const { container } = render(
      <Slider value={[50]} min={0} max={100} />
    );
    const indicator = container.querySelector('[data-slot="slider-range"]');
    expect(indicator).toBeInTheDocument();
    expect(indicator?.className).toContain("rounded-full");
  });

  it("renders a thumb element", () => {
    const { container } = render(
      <Slider value={[50]} min={0} max={100} />
    );
    const thumb = container.querySelector('[data-slot="slider-thumb"]');
    expect(thumb).toBeInTheDocument();
  });

  it("renders without overflow-hidden on track at all value positions", () => {
    [0, 50, 100].forEach((val) => {
      const { container } = render(
        <Slider value={[val]} min={0} max={100} />
      );
      const track = container.querySelector('[data-slot="slider-track"]');
      expect(track?.className).not.toContain("overflow-hidden");
    });
  });

  it("renders with custom className", () => {
    const { container } = render(
      <Slider value={[50]} min={0} max={100} className="my-slider" />
    );
    const root = container.querySelector('[data-slot="slider"]');
    expect(root?.className).toContain("my-slider");
  });
});
