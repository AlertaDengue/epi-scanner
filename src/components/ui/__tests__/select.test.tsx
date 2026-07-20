import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  SelectGroup,
  SelectLabel,
  SelectSeparator,
} from "@/components/ui/select";

describe("Select components", () => {
  it("renders a Select with trigger and placeholder", () => {
    render(
      <Select>
        <SelectTrigger>
          <SelectValue placeholder="Choose..." />
        </SelectTrigger>
      </Select>
    );
    expect(screen.getByText("Choose...")).toBeInTheDocument();
  });

  it("renders a SelectItem", () => {
    render(
      <Select defaultOpen>
        <SelectTrigger>
          <SelectValue placeholder="Pick" />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectLabel>Group</SelectLabel>
            <SelectItem value="a">Option A</SelectItem>
            <SelectSeparator />
            <SelectItem value="b">Option B</SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
    );
    // The trigger should show the placeholder
    expect(screen.getByText("Pick")).toBeInTheDocument();
  });
});
