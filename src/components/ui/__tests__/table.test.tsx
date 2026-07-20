import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  Table,
  TableHeader,
  TableBody,
  TableFooter,
  TableRow,
  TableHead,
  TableCell,
  TableCaption,
} from "@/components/ui/table";

describe("Table", () => {
  it("renders a table with container", () => {
    const { container } = render(<Table />);
    expect(container.querySelector('[data-slot="table-container"]')).toBeInTheDocument();
    expect(container.querySelector("table")).toBeInTheDocument();
  });

  it("renders TableHeader", () => {
    render(
      <table>
        <TableHeader />
      </table>
    );
    expect(document.querySelector('[data-slot="table-header"]')).toBeInTheDocument();
  });

  it("renders TableBody", () => {
    render(
      <table>
        <TableBody />
      </table>
    );
    expect(document.querySelector('[data-slot="table-body"]')).toBeInTheDocument();
  });

  it("renders TableFooter", () => {
    render(
      <table>
        <TableFooter />
      </table>
    );
    expect(document.querySelector('[data-slot="table-footer"]')).toBeInTheDocument();
  });

  it("renders TableRow", () => {
    render(
      <table>
        <TableBody>
          <TableRow data-testid="row" />
        </TableBody>
      </table>
    );
    expect(document.querySelector('[data-slot="table-row"]')).toBeInTheDocument();
  });

  it("renders TableHead", () => {
    render(
      <table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
          </TableRow>
        </TableHeader>
      </table>
    );
    expect(screen.getByText("Name")).toBeInTheDocument();
  });

  it("renders TableCell", () => {
    render(
      <table>
        <TableBody>
          <TableRow>
            <TableCell>Data</TableCell>
          </TableRow>
        </TableBody>
      </table>
    );
    expect(screen.getByText("Data")).toBeInTheDocument();
  });

  it("renders TableCaption", () => {
    render(
      <table>
        <TableCaption>Table Caption</TableCaption>
      </table>
    );
    expect(screen.getByText("Table Caption")).toBeInTheDocument();
  });
});
