import { Activity, CalendarClock, Layers, TrendingUp, Loader2, HelpCircle } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

type StatCardsProps = {
  cumulativeCases: number;
  topR0: number;
  peakYear: number;
  locationLabel: string;
  locationValue: string;
  locationHint: string;
  loading?: boolean;
  displayYear?: string;
};

export function StatCards({
  cumulativeCases,
  topR0,
  peakYear,
  locationLabel,
  locationValue,
  locationHint,
  loading = false,
  displayYear,
}: StatCardsProps) {
  const seasonLabel = displayYear ?? String(peakYear);
  const epiYearStart = Number(seasonLabel) - 1;
  const stats = [
    {
      label: "Cumulative notified cases",
      value: cumulativeCases.toLocaleString("en-US"),
      hint: `Since ${epiYearStart}-45`,
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
      value: seasonLabel,
      hint: "Epidemiological year",
      icon: CalendarClock,
      tooltip: `Epidemiological year from week 45 of ${epiYearStart} to week 45 of ${seasonLabel}`,
    },
    {
      label: locationLabel,
      value: locationValue,
      hint: locationHint,
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
              <div className="flex items-center gap-1">
                {stat.tooltip && (
                  <span
                    title={stat.tooltip}
                    className="cursor-help"
                  >
                    <HelpCircle className="size-3.5 text-muted-foreground" />
                  </span>
                )}
                <stat.icon
                  className="size-4 shrink-0 text-primary"
                  aria-hidden="true"
                />
              </div>
            </div>
            <span className="flex min-h-[2rem] items-center text-2xl font-bold tracking-tight tabular-nums">
              {loading ? (
                <Loader2 className="size-5 animate-spin text-muted-foreground" />
              ) : (
                stat.value
              )}
            </span>
            <span className="text-xs text-muted-foreground">{stat.hint}</span>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
