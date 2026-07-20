import { Radio } from "lucide-react";
import Image from "next/image";

export function DashboardHeader() {
  return (
    <header className="sticky top-0 z-[1100] border-b border-border bg-primary text-primary-foreground">
      <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4 px-4 py-3 md:px-6">
        <div className="flex items-center gap-3">
          <Image
            src="https://github.com/AlertaDengue/AlertaDengue/blob/main/AlertaDengue/static/img/logo-infodengue.png?raw=true"
            alt="Infodengue"
            width={40}
            height={40}
            className="size-10 rounded-lg"
            priority
          />
          <div className="flex flex-col leading-tight">
            <h1 className="text-lg font-bold tracking-tight text-balance md:text-xl">
              Real-time Epidemic Scanner
            </h1>
            <p className="text-xs text-primary-foreground/70">
              Real-time epidemiology · Infodengue
            </p>
          </div>
        </div>
        <div className="hidden items-center gap-2 rounded-full bg-primary-foreground/15 px-3 py-1.5 text-xs font-medium sm:flex">
          <Radio className="size-3.5 animate-pulse" aria-hidden="true" />
          <span>Live surveillance</span>
        </div>
      </div>
    </header>
  );
}
