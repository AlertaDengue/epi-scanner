"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { MapPin, Stethoscope } from "lucide-react";

type SidebarProps = {
  disease: string;
  state: string;
  city: string;
  cities: { geocode: number; name: string }[];
  topCities: { name_muni: string; transmissao: number; geocode: number }[];
  onDiseaseChange: (v: string) => void;
  onStateChange: (v: string) => void;
  onCityChange: (v: string) => void;
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
  city,
  topCities,
  onDiseaseChange,
  onStateChange,
  onCityChange,
}: SidebarProps) {
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
                <SelectValue />
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
                <SelectValue />
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
        </CardContent>
      </Card>

      <Card className="flex-1">
        <CardHeader>
          <CardTitle className="text-sm font-semibold">
            Top 20 most active cities
          </CardTitle>
          <p className="text-xs text-muted-foreground">
            Ranked by weeks of Rt &gt; 1
          </p>
        </CardHeader>
        <CardContent>
          <ul className="flex flex-col gap-1">
            {topCities.map((c, i) => {
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
        </CardContent>
      </Card>
    </aside>
  );
}
