"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface TopR0TableProps {
  cities: { name: string; geocode: number; R0: number }[];
}

export function TopR0Table({ cities }: TopR0TableProps) {
  return (
    <div className="rounded-lg border bg-white p-4">
      <h3 className="mb-2 text-sm font-semibold">Top 10 R0s</h3>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>City</TableHead>
            <TableHead className="text-right">R0</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {cities.map((city, i) => (
            <TableRow key={i}>
              <TableCell className="font-medium">{city.name}</TableCell>
              <TableCell className="text-right">{city.R0.toFixed(2)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
