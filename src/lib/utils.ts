import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDateISO(date: Date): string {
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}-${String(date.getUTCDate()).padStart(2, "0")}`;
}

export function cdcEpiweekStart(year: number, week: number): Date {
  const jan4 = new Date(Date.UTC(year, 0, 4));
  const dayOfWeek = jan4.getUTCDay();
  const sundayBefore = new Date(jan4);
  sundayBefore.setUTCDate(jan4.getUTCDate() - dayOfWeek);
  sundayBefore.setUTCDate(sundayBefore.getUTCDate() + (week - 1) * 7);
  return sundayBefore;
}
