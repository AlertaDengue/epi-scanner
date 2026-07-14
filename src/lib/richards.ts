/**
 * Richards growth model - ported from Python
 * j = L - L * (1 + a * exp(b * (t - tj))) ** (-1/a)
 */
export function richards(
  L: number,
  a: number,
  b: number,
  t: number,
  tj: number
): number {
  return L - L * Math.pow(1 + a * Math.exp(b * (t - tj)), -1 / a);
}

/**
 * Vectorized Richards model - returns array of values
 */
export function richardsVectorized(
  L: number,
  a: number,
  b: number,
  tValues: number[],
  tj: number
): number[] {
  return tValues.map((t) => richards(L, a, b, t, tj));
}

/**
 * Calculate epidemiological week start date for a given year.
 * Matches Python's get_ini_end_week function using epiweeks.
 */
export function getIniEndWeek(year: number, endYear?: number): { startDate: string; endDate: string } {
  // Week 1 of previous year starts around Jan 4 (ISO week)
  // Python: Week(year - 1, 1).startdate() gives the Monday of week 1
  const prevYearWeek1 = getWeekStartDate(year - 1, 1);

  // Generate 104 weeks (2 years) of Sundays
  const dates: Date[] = [];
  const start = new Date(prevYearWeek1);
  // Find the first Sunday on or after start
  const dayOfWeek = start.getDay();
  const sundayOffset = dayOfWeek === 0 ? 0 : 7 - dayOfWeek;
  const firstSunday = new Date(start);
  firstSunday.setDate(firstSunday.getDate() + sundayOffset);

  for (let i = 0; i < 104; i++) {
    const d = new Date(firstSunday);
    d.setDate(d.getDate() + i * 7);
    dates.push(d);
  }

  // Filter dates >= year-1 and take 44..96 (52 weeks)
  const filtered = dates.filter((d) => d.getFullYear() >= year - 1);
  const slice = filtered.slice(44, 44 + 52);

  const startDate = formatDate(slice[0]);
  let endDate: string;

  if (endYear !== undefined && endYear !== null) {
    endDate = `${endYear}-11-01`;
  } else {
    endDate = formatDate(slice[slice.length - 1]);
  }

  return { startDate, endDate };
}

function getWeekStartDate(year: number, week: number): string {
  // ISO week: week 1 contains Jan 4
  const jan4 = new Date(year, 0, 4);
  const dayOfWeek = jan4.getDay() || 7; // Convert Sunday=0 to 7
  const monday = new Date(jan4);
  monday.setDate(jan4.getDate() - dayOfWeek + 1);

  // Go to the requested week
  const targetMonday = new Date(monday);
  targetMonday.setDate(monday.getDate() + (week - 1) * 7);

  return formatDate(targetMonday);
}

function formatDate(d: Date): string {
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

/**
 * Get the Sunday-based week number from a date string
 */
export function getWeekNumber(dateStr: string): number {
  const d = new Date(dateStr);
  const startOfYear = new Date(d.getFullYear(), 0, 1);
  const diff = d.getTime() - startOfYear.getTime();
  return Math.ceil(diff / (7 * 24 * 60 * 60 * 1000));
}
