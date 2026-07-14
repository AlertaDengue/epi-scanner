"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface SIRParamsTableProps {
  params: {
    year: number;
    beta: number;
    gamma: number;
    R0: number;
    peak_week: number;
    ep_ini: number;
    ep_end: number;
    ep_dur: number;
    total_cases: number;
    reported_cases?: number;
  }[];
}

export function SIRParamsTable({ params }: SIRParamsTableProps) {
  return (
    <div className="rounded-lg border bg-white p-4">
      <h3 className="mb-2 text-sm font-semibold">SIR Parameters</h3>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Year</TableHead>
            <TableHead className="text-right">Beta</TableHead>
            <TableHead className="text-right">Gamma</TableHead>
            <TableHead className="text-right">R0</TableHead>
            <TableHead className="text-right">Peak Week</TableHead>
            <TableHead className="text-right">Start</TableHead>
            <TableHead className="text-right">End</TableHead>
            <TableHead className="text-right">Duration</TableHead>
            <TableHead className="text-right">Est. Cases</TableHead>
            <TableHead className="text-right">Reported</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {params.map((row, i) => (
            <TableRow key={i}>
              <TableCell className="font-medium">{row.year}</TableCell>
              <TableCell className="text-right">{row.beta.toFixed(2)}</TableCell>
              <TableCell className="text-right">{row.gamma.toFixed(2)}</TableCell>
              <TableCell className="text-right">{row.R0.toFixed(2)}</TableCell>
              <TableCell className="text-right">{row.peak_week}</TableCell>
              <TableCell className="text-right">{row.ep_ini}</TableCell>
              <TableCell className="text-right">{row.ep_end}</TableCell>
              <TableCell className="text-right">{row.ep_dur}</TableCell>
              <TableCell className="text-right">{row.total_cases.toLocaleString()}</TableCell>
              <TableCell className="text-right">
                {row.reported_cases?.toLocaleString() ?? "—"}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
