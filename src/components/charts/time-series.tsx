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

interface TimeSeriesChartProps {
  data: { date: string; casos: number; casos_cum: number }[];
  modelData?: { date: string; model: number }[];
  peakWeekDate?: string | null;
  title: string;
}

export function TimeSeriesChart({
  data,
  modelData,
  peakWeekDate,
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
              const d = new Date(v);
              return `${d.getMonth() + 1}/${d.getFullYear()}`;
            }}
          />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip
            labelFormatter={(v) => new Date(v).toLocaleDateString()}
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
              stroke="#ef4444"
              fill="none"
              strokeWidth={2}
              dot={false}
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
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
