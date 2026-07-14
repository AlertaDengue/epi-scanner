"use client";

import { useState, useEffect } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface SidebarProps {
  disease: string;
  state: string;
  city: string;
  cities: { geocode: number; name: string }[];
  onDiseaseChange: (disease: string) => void;
  onStateChange: (state: string) => void;
  onCityChange: (geocode: string) => void;
}

const STATES = [
  { code: "AC", name: "Acre" }, { code: "AL", name: "Alagoas" },
  { code: "AM", name: "Amazonas" }, { code: "AP", name: "Amapá" },
  { code: "BA", name: "Bahia" }, { code: "CE", name: "Ceará" },
  { code: "DF", name: "Distrito Federal" }, { code: "ES", name: "Espírito Santo" },
  { code: "GO", name: "Goiás" }, { code: "MA", name: "Maranhão" },
  { code: "MG", name: "Minas Gerais" }, { code: "MS", name: "Mato Grosso do Sul" },
  { code: "MT", name: "Mato Grosso" }, { code: "PA", name: "Pará" },
  { code: "PB", name: "Paraíba" }, { code: "PE", name: "Pernambuco" },
  { code: "PI", name: "Piauí" }, { code: "PR", name: "Paraná" },
  { code: "RJ", name: "Rio de Janeiro" }, { code: "RN", name: "Rio Grande do Norte" },
  { code: "RO", name: "Rondônia" }, { code: "RR", name: "Roraima" },
  { code: "RS", name: "Rio Grande do Sul" }, { code: "SC", name: "Santa Catarina" },
  { code: "SE", name: "Sergipe" }, { code: "SP", name: "São Paulo" },
  { code: "TO", name: "Tocantins" },
];

export function Sidebar({
  disease,
  state,
  city,
  cities,
  onDiseaseChange,
  onStateChange,
  onCityChange,
}: SidebarProps) {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Filters</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Disease</label>
            <Select value={disease} onValueChange={(v) => v && onDiseaseChange(v)}>
              <SelectTrigger>
                <SelectValue placeholder="Select disease" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="dengue">Dengue</SelectItem>
                <SelectItem value="chikungunya">Chikungunya</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">State</label>
            <Select value={state} onValueChange={(v) => v && onStateChange(v)}>
              <SelectTrigger>
                <SelectValue placeholder="Select state" />
              </SelectTrigger>
              <SelectContent>
                {STATES.map((s) => (
                  <SelectItem key={s.code} value={s.code}>
                    {s.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">City</label>
            <Select value={city} onValueChange={(v) => v && onCityChange(v)}>
              <SelectTrigger>
                <SelectValue placeholder="Select city" />
              </SelectTrigger>
              <SelectContent>
                {cities.map((c) => (
                  <SelectItem key={c.geocode} value={String(c.geocode)}>
                    {c.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <p className="text-xs text-muted-foreground">
            Parameters can be downloaded from the{" "}
            <a
              href="https://api.mosqlimate.org/docs/datastore/GET/episcanner/"
              target="_blank"
              rel="noopener noreferrer"
              className="underline"
            >
              Mosqlimate API
            </a>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
