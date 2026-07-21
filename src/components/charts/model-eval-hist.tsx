"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface ModelEvalHistProps {
  rates: number[];
}

const COLORS = ["#006aea", "#00b4ca", "#48d085", "#dc7080", "#cb2b2b"];
const BINS = [0, 0.5, 0.95, 1.05, 2, 3];

function binRates(rates: number[]) {
  const bins = BINS.slice(0, -1).map((min, i) => ({
    range: `${min.toFixed(2)}-${BINS[i + 1].toFixed(2)}`,
    count: 0,
    index: i,
  }));

  for (const rate of rates) {
    for (let i = 0; i < BINS.length - 1; i++) {
      if (rate >= BINS[i] && rate < BINS[i + 1]) {
        bins[i].count++;
        break;
      }
      if (i === BINS.length - 2 && rate >= BINS[i]) {
        bins[i].count++;
      }
    }
  }

  return bins;
}

export function ModelEvalHist({ rates }: ModelEvalHistProps) {
  const data = binRates(rates);

  return (
    <div className="rounded-lg border bg-white p-4">
      <h3 className="mb-4 text-sm font-semibold">Rate Distribution</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart
          data={data}
          margin={{ top: 5, right: 5, left: 0, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="range" tick={{ fontSize: 11 }} />
          <YAxis width={28} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Bar dataKey="count" name="Cities">
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
