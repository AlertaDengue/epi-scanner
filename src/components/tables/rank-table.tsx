type RankRow = { name: string; value: number };

type RankTableProps = {
  rows: RankRow[];
  valueLabel: string;
  format?: (v: number) => string;
  barColor?: string;
};

export function RankTable({
  rows,
  valueLabel,
  format = (v) => v.toLocaleString("en-US"),
  barColor = "var(--color-chart-1)",
}: RankTableProps) {
  const max = Math.max(...rows.map((r) => r.value), 1);

  return (
    <ul className="flex flex-col gap-1.5">
      <li className="flex items-center justify-between px-1 pb-1 text-xs font-medium text-muted-foreground">
        <span>City</span>
        <span>{valueLabel}</span>
      </li>
      {rows.map((row, i) => (
        <li key={row.name} className="flex items-center gap-3">
          <span className="w-4 shrink-0 font-mono text-xs text-muted-foreground">
            {i + 1}
          </span>
          <div className="relative flex-1 overflow-hidden rounded-md bg-muted/60">
            <div
              className="h-7 rounded-md opacity-25"
              style={{
                width: `${(row.value / max) * 100}%`,
                backgroundColor: barColor,
              }}
            />
            <span className="absolute inset-0 flex items-center px-2 text-sm font-medium">
              {row.name}
            </span>
          </div>
          <span className="w-14 shrink-0 text-right text-sm font-semibold tabular-nums">
            {format(row.value)}
          </span>
        </li>
      ))}
    </ul>
  );
}
