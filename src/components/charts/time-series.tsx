"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { getWeekNumber } from "@/lib/richards";
import { formatDateISO } from "@/lib/utils";

interface TimeSeriesChartProps {
  data: { date: string; casos: number; casos_cum: number }[];
  modelData?: { date: string; model: number }[];
  peakWeekDate?: string | null;
  startDate?: string | null;
  endDate?: string | null;
  title: string;
}

export function TimeSeriesChart({
  data,
  modelData,
  peakWeekDate,
  startDate,
  endDate,
  title,
}: TimeSeriesChartProps) {
  // Merge data for the chart
  const mergedData = data.map((d, i) => ({
    ...d,
    model: modelData?.[i]?.model ?? null,
  }));

  return (
    <div className="rounded-lg border bg-white p-4">
      <h3 className="mb-4 text-sm font-semibold">{title}</h3>
      <ResponsiveContainer width="100%" height={400}>
        <AreaChart data={mergedData}>
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
              const wk = getWeekNumber(v);
              const dateStr = formatDateISO(new Date(v));
              return `${dateStr} · Epiweek ${wk}`;
            }}
            formatter={(value, name) => [
              Math.round(Number(value)).toLocaleString(),
              String(name),
            ]}
          />
          <Legend />
          <Area
            type="stepAfter"
            dataKey="casos"
            name="New Cases"
            stroke="#3b82f6"
            fill="#3b82f6"
            fillOpacity={0.3}
          />
          <Area
            type="stepAfter"
            dataKey="casos_cum"
            name="Cumulative Cases"
            stroke="#6366f1"
            fill="#6366f1"
            fillOpacity={0.3}
          />
          {modelData && (
            <Area
              type="monotone"
              dataKey="model"
              name="Model"
              stroke="#ff7f0e"
              fill="none"
              strokeWidth={2}
              dot={false}
            />
          )}
          {startDate && (
            <ReferenceLine
              x={startDate}
              stroke="#22c55e"
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{ value: "Start", position: "top", fontSize: 10 }}
            />
          )}
          {peakWeekDate && (
            <ReferenceLine
              x={peakWeekDate}
              stroke="orange"
              strokeWidth={2}
              label={{ value: "Peak", position: "top", fontSize: 10 }}
            />
          )}
          {endDate && (
            <ReferenceLine
              x={endDate}
              stroke="#a855f7"
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{ value: "End", position: "top", fontSize: 10 }}
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
