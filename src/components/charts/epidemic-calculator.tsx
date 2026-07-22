"use client";

import { useState, useCallback } from "react";
import {
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
import { richards, getWeekNumber } from "@/lib/richards";
import { formatDateISO } from "@/lib/utils";
import { Lock, LockOpen } from "lucide-react";

interface EpidemicCalculatorProps {
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
  const [peakWeek, setPeakWeek] = useState(Math.round(initialPeakWeek));
  const [r0, setR0] = useState(initialR0);
  const [totalCases, setTotalCases] = useState(Math.round(initialTotalCases));

  const [lockPeak, setLockPeak] = useState(true);
  const [lockR0, setLockR0] = useState(true);
  const [lockCases, setLockCases] = useState(true);

  const effectivePeakWeek = lockPeak ? initialPeakWeek : peakWeek;
  const effectiveR0 = lockR0 ? initialR0 : r0;
  const effectiveTotalCases = lockCases ? initialTotalCases : totalCases;

  const calculateModel = useCallback(() => {
    const r = 1 - 1 / effectiveR0;
    const gamma = 0.3;
    const b = (r * gamma) / (1 - r);
    const a = b / (gamma + b);

    return Array.from({ length: dates.length }, (_, i) => richards(effectiveTotalCases, a, b, i, effectivePeakWeek));
  }, [effectivePeakWeek, effectiveR0, effectiveTotalCases, dates]);

  const modelValues = calculateModel();

  const chartData = dates.map((date, i) => ({
    date,
    data: dataCumulative[i] || 0,
    model: modelValues[i] || 0,
  }));

  const peakDate = dates[Math.round(effectivePeakWeek)] || dates[0];

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="pt-4">
          <p className="text-xs text-muted-foreground mb-4">
            The chart displays observed cumulative cases in blue, the Richards
            model in orange, and the peak week in red. Each parameter defaults
            to its data-optimized value when locked (🔒). Unlock to adjust
            manually.
          </p>
          <ResponsiveContainer width="100%" height={350}>
            <ComposedChart data={chartData}>
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
                dataKey="data"
                name="Cumulative Cases"
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
            <CardTitle className="flex items-center justify-between text-sm">
              <span>Peak week</span>
              <button
                type="button"
                onClick={() => {
                  setLockPeak(!lockPeak);
                  if (lockPeak) setPeakWeek(Math.round(initialPeakWeek));
                }}
                className="rounded p-0.5 text-muted-foreground hover:bg-muted hover:text-foreground"
                title={lockPeak ? "Unlock to set manually" : "Lock to optimal value"}
              >
                {lockPeak ? <Lock className="size-3.5" /> : <LockOpen className="size-3.5" />}
              </button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Slider
              value={[effectivePeakWeek]}
              onValueChange={(v) => !lockPeak && setPeakWeek(Math.round(Array.isArray(v) ? v[0] : v))}
              min={1}
              max={dates.length - 1}
              step={1}
              disabled={lockPeak}
            />
            <p className="mt-1 text-xs text-muted-foreground">
              {lockPeak ? `Optimal: ${Math.round(effectivePeakWeek)}` : `Current: ${peakWeek}`}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between text-sm">
              <span>R0</span>
              <button
                type="button"
                onClick={() => {
                  setLockR0(!lockR0);
                  if (lockR0) setR0(initialR0);
                }}
                className="rounded p-0.5 text-muted-foreground hover:bg-muted hover:text-foreground"
                title={lockR0 ? "Unlock to set manually" : "Lock to optimal value"}
              >
                {lockR0 ? <Lock className="size-3.5" /> : <LockOpen className="size-3.5" />}
              </button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Slider
              value={[effectiveR0]}
              onValueChange={(v) => !lockR0 && setR0(Array.isArray(v) ? v[0] : v)}
              min={0.1}
              max={5}
              step={0.05}
              disabled={lockR0}
            />
            <p className="mt-1 text-xs text-muted-foreground">
              {lockR0 ? `Optimal: ${effectiveR0.toFixed(2)}` : `Current: ${r0.toFixed(2)}`}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between text-sm">
              <span>Total cases</span>
              <button
                type="button"
                onClick={() => {
                  setLockCases(!lockCases);
                  if (lockCases) setTotalCases(Math.round(initialTotalCases));
                }}
                className="rounded p-0.5 text-muted-foreground hover:bg-muted hover:text-foreground"
                title={lockCases ? "Unlock to set manually" : "Lock to optimal value"}
              >
                {lockCases ? <Lock className="size-3.5" /> : <LockOpen className="size-3.5" />}
              </button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Slider
              value={[effectiveTotalCases]}
              onValueChange={(v) => !lockCases && setTotalCases(Math.round(Array.isArray(v) ? v[0] : v))}
              min={minCases}
              max={maxCases}
              step={step}
              disabled={lockCases}
            />
            <p className="mt-1 text-xs text-muted-foreground">
              {lockCases ? `Optimal: ${effectiveTotalCases.toLocaleString()}` : `Current: ${totalCases.toLocaleString()}`}
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
