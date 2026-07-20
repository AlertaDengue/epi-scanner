import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  Select,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

describe("Select", () => {
  it("renders trigger with placeholder text", () => {
    render(
      <Select>
        <SelectTrigger>
          <SelectValue placeholder="Choose an option" />
        </SelectTrigger>
      </Select>
    );
    expect(screen.getByText("Choose an option")).toBeInTheDocument();
  });

  it("renders with custom className on trigger", () => {
    const { container } = render(
      <Select>
        <SelectTrigger className="w-40">
          <SelectValue placeholder="Test" />
        </SelectTrigger>
      </Select>
    );
    expect(screen.getByText("Test")).toBeInTheDocument();
    expect(container.querySelector('[data-slot="select-trigger"]')).toBeInTheDocument();
  });
});

describe("Separator", () => {
  it("renders separator element", async () => {
    const { Separator } = await import("@/components/ui/separator");
    const { container } = render(<Separator />);
    expect(container.querySelector('[data-slot="separator"]')).toBeInTheDocument();
  });

  it("renders vertical orientation", async () => {
    const { Separator } = await import("@/components/ui/separator");
    const { container } = render(<Separator orientation="vertical" />);
    const sep = container.querySelector('[data-slot="separator"]');
    expect(sep).toBeInTheDocument();
  });
});
