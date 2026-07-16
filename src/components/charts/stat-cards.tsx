import { Activity, CalendarClock, Layers, TrendingUp } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

type StatCardsProps = {
  cumulativeCases: number;
  topR0: number;
  peakYear: number;
  state: string;
};

export function StatCards({
  cumulativeCases,
  topR0,
  peakYear,
  state,
}: StatCardsProps) {
  const stats = [
    {
      label: "Cumulative notified cases",
      value: cumulativeCases.toLocaleString("en-US"),
      hint: `Since Jan ${peakYear}`,
      icon: Activity,
    },
    {
      label: "Highest current R₀",
      value: topR0.toFixed(2),
      hint: "Basic reproduction number",
      icon: TrendingUp,
    },
    {
      label: "Monitored season",
      value: String(peakYear),
      hint: "Epidemiological year",
      icon: CalendarClock,
    },
    {
      label: "State under scan",
      value: state,
      hint: "Municipal-level resolution",
      icon: Layers,
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {stats.map((stat) => (
        <Card key={stat.label}>
          <CardContent className="flex flex-col gap-2 p-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground text-pretty">
                {stat.label}
              </span>
              <stat.icon
                className="size-4 shrink-0 text-primary"
                aria-hidden="true"
              />
            </div>
            <span className="text-2xl font-bold tracking-tight tabular-nums">
              {stat.value}
            </span>
            <span className="text-xs text-muted-foreground">{stat.hint}</span>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
