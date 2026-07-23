"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { formatDateISO, cdcEpiweekStart } from "@/lib/utils";

interface ReportedCasesChartProps {
  data: { date: string; cases: number | null }[];
  title?: string;
  targetYear?: number;
}

export function ReportedCasesChart({
  data,
  title = "Reported cases",
  targetYear,
}: ReportedCasesChartProps) {
  const currentYear = targetYear ?? new Date().getFullYear();
  const epiStart = cdcEpiweekStart(currentYear - 1, 45);
  const epiEnd = cdcEpiweekStart(currentYear, 45);

  const enriched = data.map((d) => {
    const date = new Date(d.date);
    const inEpiYear = date >= epiStart && date < epiEnd;
    return {
      ...d,
      isCurrent: inEpiYear ? d.cases : null,
      isPrevious: !inEpiYear ? d.cases : null,
    };
  });

  return (
    <div className="rounded-lg border bg-white p-4">
      <h3 className="mb-4 text-sm font-semibold">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={enriched}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11 }}
            tickFormatter={(v) => {
              const d = new Date(String(v));
              return formatDateISO(d);
            }}
          />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip
            labelFormatter={(v) => {
              if (typeof v !== "string") return String(v);
              return formatDateISO(new Date(v));
            }}
            formatter={(value, name) => [
              Math.round(Number(value)).toLocaleString(),
              String(name),
            ]}
          />
          <Area
            type="monotone"
            dataKey="isPrevious"
            name="Cases"
            stroke="#94a3b8"
            fill="#94a3b8"
            fillOpacity={0.1}
            strokeWidth={1.5}
            dot={false}
          />
          <Area
            type="monotone"
            dataKey="isCurrent"
            name="Cases"
            stroke="#2563eb"
            fill="#2563eb"
            fillOpacity={0.2}
            strokeWidth={2.5}
            dot={false}
            connectNulls
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
