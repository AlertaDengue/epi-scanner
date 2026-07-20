"use client";

import { useState, useCallback } from "react";
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
  Line,
  ComposedChart,
} from "recharts";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { richards } from "@/lib/richards";

interface EpidemicCalculatorProps {
  disease: string;
  city: string;
  dataCumulative: number[];
  dates: string[];
  initialPeakWeek: number;
  initialR0: number;
  initialTotalCases: number;
  minCases: number;
  maxCases: number;
  step: number;
  startDate?: string | null;
  endDate?: string | null;
}

export function EpidemicCalculator({
  disease,
  city,
  dataCumulative,
  dates,
  initialPeakWeek,
  initialR0,
  initialTotalCases,
  minCases,
  maxCases,
  step,
  startDate,
  endDate,
}: EpidemicCalculatorProps) {
  const [peakWeek, setPeakWeek] = useState(initialPeakWeek);
  const [r0, setR0] = useState(initialR0);
  const [totalCases, setTotalCases] = useState(initialTotalCases);

  const calculateModel = useCallback(() => {
    const r = 1 - 1 / r0;
    const gamma = 0.3;
    const b = (r * gamma) / (1 - r);
    const a = b / (gamma + b);

    return Array.from({ length: 52 }, (_, i) => richards(totalCases, a, b, i, peakWeek));
  }, [peakWeek, r0, totalCases]);

  const modelValues = calculateModel();

  const chartData = dates.map((date, i) => ({
    date,
    data: dataCumulative[i] || 0,
    model: modelValues[i] || 0,
  }));

  const peakDate = dates[Math.round(peakWeek)] || dates[0];

  const title = `${disease.charAt(0).toUpperCase() + disease.slice(1)} weekly cases in ${new Date().getFullYear()} for ${city}`;

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="pt-4">
          <p className="text-xs text-muted-foreground mb-4">
            The section below displays the cumulative cases for the selected
            city in blue, the Richards model in orange, and the peak week in
            red. The sliders allow you to adjust the peak week, reproduction
            number (R0), and total number of cases.
          </p>
          <ResponsiveContainer width="100%" height={350}>
            <ComposedChart data={chartData}>
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
                dataKey="data"
                name="Data"
                stroke="#1f77b4"
                fill="#1f77b4"
                fillOpacity={0.3}
              />
              <Line
                type="monotone"
                dataKey="model"
                name="Model"
                stroke="#ff7f0e"
                strokeWidth={2}
                dot={false}
              />
              {startDate && (
                <ReferenceLine
                  x={startDate}
                  stroke="#22c55e"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  label={{ value: "Start", position: "top", fontSize: 10 }}
                />
              )}
              {peakDate && (
                <ReferenceLine
                  x={peakDate}
                  stroke="red"
                  strokeWidth={2}
                  label={{ value: "Peak week", position: "top", fontSize: 10 }}
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
            </ComposedChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Peak week</CardTitle>
          </CardHeader>
          <CardContent>
            <Slider
              value={[peakWeek]}
              onValueChange={(v) => setPeakWeek(Array.isArray(v) ? v[0] : v)}
              min={5}
              max={45}
              step={1}
            />
            <p className="mt-1 text-xs text-muted-foreground">
              Current: {peakWeek}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">R0</CardTitle>
          </CardHeader>
          <CardContent>
            <Slider
              value={[r0]}
              onValueChange={(v) => setR0(Array.isArray(v) ? v[0] : v)}
              min={0.1}
              max={5}
              step={0.05}
            />
            <p className="mt-1 text-xs text-muted-foreground">
              Current: {r0.toFixed(2)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Total cases</CardTitle>
          </CardHeader>
          <CardContent>
            <Slider
              value={[totalCases]}
              onValueChange={(v) => setTotalCases(Array.isArray(v) ? v[0] : v)}
              min={minCases}
              max={maxCases}
              step={step}
            />
            <p className="mt-1 text-xs text-muted-foreground">
              Current: {totalCases.toLocaleString()}
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
