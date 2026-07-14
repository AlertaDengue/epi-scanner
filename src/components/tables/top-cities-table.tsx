"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface TopCitiesTableProps {
  cities: { name_muni: string; transmissao: number }[];
  title: string;
}

export function TopCitiesTable({ cities, title }: TopCitiesTableProps) {
  return (
    <div className="rounded-lg border bg-white p-4">
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>City</TableHead>
            <TableHead className="text-right">Epi Weeks</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {cities.map((city, i) => (
            <TableRow key={i}>
              <TableCell className="font-medium">{city.name_muni}</TableCell>
              <TableCell className="text-right">{city.transmissao}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
