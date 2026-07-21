"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { MapPin, Stethoscope, ChevronDown, Calendar, Building2 } from "lucide-react";

type SidebarProps = {
  disease: string;
  state: string;
  city: string;
  cities: { geocode: number; name: string }[];
  topCities: { name_muni: string; transmissao: number; geocode: number }[];
  onDiseaseChange: (v: string) => void;
  onStateChange: (v: string) => void;
  onCityChange: (v: string) => void;
  loading?: boolean;
  epiYear: string;
  onEpiYearChange: (v: string) => void;
  epiYears: number[];
};

const STATE_OPTIONS = [
  { code: "AC", name: "Acre" },
  { code: "AL", name: "Alagoas" },
  { code: "AP", name: "Amapá" },
  { code: "AM", name: "Amazonas" },
  { code: "BA", name: "Bahia" },
  { code: "CE", name: "Ceará" },
  { code: "DF", name: "Distrito Federal" },
  { code: "ES", name: "Espírito Santo" },
  { code: "GO", name: "Goiás" },
  { code: "MA", name: "Maranhão" },
  { code: "MT", name: "Mato Grosso" },
  { code: "MS", name: "Mato Grosso do Sul" },
  { code: "MG", name: "Minas Gerais" },
  { code: "PA", name: "Pará" },
  { code: "PB", name: "Paraíba" },
  { code: "PR", name: "Paraná" },
  { code: "PE", name: "Pernambuco" },
  { code: "PI", name: "Piauí" },
  { code: "RJ", name: "Rio de Janeiro" },
  { code: "RN", name: "Rio Grande do Norte" },
  { code: "RS", name: "Rio Grande do Sul" },
  { code: "RO", name: "Rondônia" },
  { code: "RR", name: "Roraima" },
  { code: "SC", name: "Santa Catarina" },
  { code: "SP", name: "São Paulo" },
  { code: "SE", name: "Sergipe" },
  { code: "TO", name: "Tocantins" },
];

const DISEASE_OPTIONS = [
  { value: "dengue", label: "Dengue" },
  { value: "chikungunya", label: "Chikungunya" },
];

export function DashboardSidebar({
  disease,
  state,
  cities,
  city,
  topCities,
  onDiseaseChange,
  onStateChange,
  onCityChange,
  loading = false,
  epiYear,
  onEpiYearChange,
  epiYears,
}: SidebarProps) {
  const [expanded, setExpanded] = useState(false);
  const visibleCities = expanded ? topCities : topCities.slice(0, 5);
  const selectedCityName = cities.find((c) => String(c.geocode) === city)?.name ?? "";
  const diseaseLabel = DISEASE_OPTIONS.find((d) => d.value === disease)?.label ?? disease;
  const stateLabel = STATE_OPTIONS.find((s) => s.code === state)?.name ?? state;
  return (
    <aside className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-semibold">Parameters</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <label className="flex flex-col gap-1.5">
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              <Stethoscope className="size-3.5" aria-hidden="true" />
              Select disease
            </span>
            <Select value={disease} onValueChange={(v) => v && onDiseaseChange(v)}>
              <SelectTrigger className="w-full">
                <SelectValue>
                  {diseaseLabel}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {DISEASE_OPTIONS.map((d) => (
                  <SelectItem key={d.value} value={d.value}>
                    {d.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </label>

          <label className="flex flex-col gap-1.5">
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              <MapPin className="size-3.5" aria-hidden="true" />
              Select state
            </span>
            <Select value={state} onValueChange={(v) => v && onStateChange(v)}>
              <SelectTrigger className="w-full">
                <SelectValue>
                  {stateLabel}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {STATE_OPTIONS.map((s) => (
                  <SelectItem key={s.code} value={s.code}>
                    {s.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </label>

          <label className="flex flex-col gap-1.5">
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              <Building2 className="size-3.5" aria-hidden="true" />
              Select city
            </span>
            <Select value={city} onValueChange={(v) => v && onCityChange(v)}>
              <SelectTrigger className="w-full">
                <SelectValue>
                  {selectedCityName || "Search city..."}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {cities.map((c) => (
                  <SelectItem key={c.geocode} value={String(c.geocode)}>
                    {c.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </label>

          <label className="flex flex-col gap-1.5">
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              <Calendar className="size-3.5" aria-hidden="true" />
              Epidemic year
            </span>
            <Select value={epiYear} onValueChange={(v) => v && onEpiYearChange(v)}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All</SelectItem>
                {epiYears.map((y) => (
                  <SelectItem key={y} value={String(y)}>
                    {y}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </label>
        </CardContent>
      </Card>

      <Card className="flex-1">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm font-semibold">
            Top 20 most active cities
            {loading && <Spinner className="size-3.5" />}
          </CardTitle>
          <p className="text-xs text-muted-foreground">
            Ranked by weeks of Rt &gt; 1
          </p>
        </CardHeader>
        <CardContent>
          <ul className="flex flex-col gap-1">
            {visibleCities.map((c, i) => {
              const active = String(c.geocode) === city;
              return (
                <li key={c.geocode}>
                  <button
                    type="button"
                    onClick={() => onCityChange(String(c.geocode))}
                    aria-pressed={active}
                    className={`flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors ${
                      active
                        ? "bg-accent text-accent-foreground"
                        : "hover:bg-muted"
                    }`}
                  >
                    <span className="w-5 shrink-0 font-mono text-xs text-muted-foreground">
                      {i + 1}
                    </span>
                    <span className="flex-1 truncate text-sm font-medium">
                      {c.name_muni}
                    </span>
                    <Badge variant="secondary" className="tabular-nums">
                      {c.transmissao}
                    </Badge>
                  </button>
                </li>
              );
            })}
          </ul>
          {topCities.length > 5 && (
            <button
              type="button"
              onClick={() => setExpanded(!expanded)}
              className="mt-2 flex w-full items-center justify-center gap-1 rounded-lg py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              {expanded ? "Show less" : `Show all ${topCities.length}`}
              <ChevronDown className={`size-3.5 transition-transform ${expanded ? "rotate-180" : ""}`} />
            </button>
          )}
        </CardContent>
      </Card>
    </aside>
  );
}
