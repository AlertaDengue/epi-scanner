"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import { StateMap } from "@/components/maps/state-map";
import { R0Map } from "@/components/maps/r0-map";
import { ModelEvalMap } from "@/components/maps/model-eval-map";
import { TimeSeriesChart } from "@/components/charts/time-series";
import { ModelEvalHist } from "@/components/charts/model-eval-hist";
import { EpidemicCalculator } from "@/components/charts/epidemic-calculator";
import { TopCitiesTable } from "@/components/tables/top-cities-table";
import { TopR0Table } from "@/components/tables/top-r0-table";
import { SIRParamsTable } from "@/components/tables/sir-params-table";
import { ModelEvalTable } from "@/components/tables/model-eval-table";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CURRENT_YEAR, MIN_YEAR } from "@/lib/constants";
import type { GeoJSON } from "@/lib/types";

export default function Dashboard() {
  // State
  const [disease, setDisease] = useState("dengue");
  const [state, setState] = useState("CE");
  const [city, setCity] = useState("");
  const [cities, setCities] = useState<{ geocode: number; name: string }[]>([]);
  const [r0year, setR0Year] = useState(CURRENT_YEAR);
  const [modelEvalYear, setModelEvalYear] = useState(CURRENT_YEAR);
  const [epiYear, setEpiYear] = useState("all");

  // Data
  const [weeksMap, setWeeksMap] = useState<GeoJSON.FeatureCollection | null>(null);
  const [r0Data, setR0Data] = useState<GeoJSON.FeatureCollection | null>(null);
  const [modelEval, setModelEval] = useState<{ rateMap: { code_muni: number; rate: number | null }[]; table: { range: string; count: number; percentage: number }[] } | null>(null);
  const [topCities, setTopCities] = useState<{ name_muni: string; transmissao: number; code_muni: number }[]>([]);
  const [top10Cities, setTop10Cities] = useState<{ name_muni: string; transmissao: number; code_muni: number }[]>([]);
  const [topR0, setTopR0] = useState<{ name: string; geocode: number; R0: number }[]>([]);
  const [timeSeries, setTimeSeries] = useState<{ date: string; casos: number; casos_cum: number }[]>([]);
  const [sirParams, setSirParams] = useState<{ geocode: number; year: number; disease: string; beta: number; gamma: number; R0: number; peak_week: number; ep_pw: number; ep_ini: number; ep_end: number; ep_dur: number; total_cases: number }[]>([]);
  const [cityParams, setCityParams] = useState<{ geocode: number; year: number; disease: string; beta: number; gamma: number; R0: number; peak_week: number; ep_pw: number; ep_ini: number; ep_end: number; ep_dur: number; total_cases: number }[]>([]);
  const [modelEvalRates, setModelEvalRates] = useState<number[]>([]);
  const [cases, setCases] = useState(0);
  const [medianParams, setMedianParams] = useState({
    medianR0: 2,
    medianPeak: 10,
    medianCases: 0,
    minCases: 0,
    maxCases: 1000,
    step: 50,
  });

  // Loading states
  const [loading, setLoading] = useState(true);

  // Fetch cities when disease/state changes
  useEffect(() => {
    const fetchCities = async () => {
      const res = await fetch(
        `/api/cities?disease=${disease}&uf=${state}`
      );
      const data = await res.json();
      setCities(data);
      if (data.length > 0 && !city) {
        // Find city with highest transmissao (first in top cities)
        const topRes = await fetch(
          `/api/top-cities?disease=${disease}&uf=${state}&limit=1`
        );
        const topData = await topRes.json();
        if (topData.length > 0) {
          setCity(String(topData[0].code_muni));
        } else {
          setCity(String(data[0].geocode));
        }
      }
    };
    fetchCities();
  }, [disease, state]);

  // Fetch weeks map
  useEffect(() => {
    const fetchWeeksMap = async () => {
      const res = await fetch(
        `/api/maps/weeks?disease=${disease}&uf=${state}`
      );
      const data = await res.json();
      setWeeksMap(data);

      // Get top 10 cities
      const topRes = await fetch(
        `/api/top-cities?disease=${disease}&uf=${state}&limit=10`
      );
      const topData = await topRes.json();
      setTop10Cities(topData);

      // Get all top cities (20) for sidebar
      const top20Res = await fetch(
        `/api/top-cities?disease=${disease}&uf=${state}&limit=20`
      );
      setTopCities(await top20Res.json());
    };
    fetchWeeksMap();
  }, [disease, state]);

  // Fetch R0 map
  useEffect(() => {
    const fetchR0 = async () => {
      const res = await fetch(
        `/api/maps/r0?disease=${disease}&uf=${state}&year=${r0year}`
      );
      const data = await res.json();
      setR0Data(data.geojson);
      setTopR0(data.topR0);
    };
    fetchR0();
  }, [disease, state, r0year]);

  // Fetch model evaluation
  useEffect(() => {
    const fetchModelEval = async () => {
      const res = await fetch(
        `/api/maps/model-eval?disease=${disease}&uf=${state}&year=${modelEvalYear}`
      );
      const data = await res.json();
      setModelEval(data);
      setModelEvalRates(data.rateMap.map((r: { rate: number | null }) => r.rate).filter((r: number | null) => r !== null));
    };
    fetchModelEval();
  }, [disease, state, modelEvalYear]);

  // Fetch parameters
  useEffect(() => {
    const fetchParams = async () => {
      const res = await fetch(
        `/api/parameters?disease=${disease}&uf=${state}`
      );
      const data = await res.json();
      setSirParams(data);
    };
    fetchParams();
  }, [disease, state]);

  // Fetch time series + model for selected city
  useEffect(() => {
    if (!city) return;
    const fetchTimeSeries = async () => {
      const res = await fetch(
        `/api/timeseries?disease=${disease}&uf=${state}&geocode=${city}&year=${epiYear}`
      );
      const data = await res.json();
      setTimeSeries(data);

      // Calculate total cases for header
      const totalCases = data.reduce(
        (sum: number, d: { casos: number }) => sum + (d.casos || 0),
        0
      );
      setCases(totalCases);

      // Get params for this city
      const cityP = sirParams.filter(
        (p: { geocode: number }) => p.geocode === Number(city)
      );
      setCityParams(cityP);

      // If a specific year is selected, get model data
      if (epiYear !== "all" && cityP.length > 0) {
        const yearParam = cityP.find(
          (p: { year: number }) => p.year === Number(epiYear)
        ) as { total_cases: number; peak_week: number } | undefined;
        if (yearParam) {
          // Compute median params for epidemic calculator
          const pastParams = cityP.filter(
            (p: { year: number }) => p.year < Number(epiYear)
          );
          let medianR0 = 2;
          let medianPeak = 10;
          let medianCases = totalCases;
          if (pastParams.length > 0) {
            const r0Vals = pastParams.map((p: { R0: number }) => p.R0).sort((a: number, b: number) => a - b);
            const peakVals = pastParams.map((p: { peak_week: number }) => p.peak_week).sort((a: number, b: number) => a - b);
            const casesVals = pastParams.map((p: { total_cases: number }) => p.total_cases).sort((a: number, b: number) => a - b);
            medianR0 = r0Vals[Math.floor(r0Vals.length / 2)];
            medianPeak = peakVals[Math.floor(peakVals.length / 2)];
            medianCases = casesVals[Math.floor(casesVals.length / 2)];
          }
          const minCases = 0.85 * totalCases;
          const maxCases = Math.max(1.25 * totalCases, 1.25 * medianCases, yearParam.total_cases || 0);
          const step = Math.max(1, Math.floor((maxCases - minCases) / 20));
          setMedianParams({
            medianR0,
            medianPeak: yearParam.peak_week || medianPeak,
            medianCases: yearParam.total_cases || medianCases,
            minCases,
            maxCases,
            step,
          });
        }
      }
      setLoading(false);
    };
    fetchTimeSeries();
  }, [city, disease, state, epiYear, sirParams]);

  const topCityName = cities.find((c) => String(c.geocode) === city)?.name || "";

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <div className="mb-4 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent mx-auto" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col">
      <Header disease={disease} state={state} cases={cases} year={new Date().getFullYear()} />

      <div className="flex flex-1">
        {/* Sidebar */}
        <div className="w-[25%] min-w-[280px] border-r bg-gray-50 p-4 overflow-y-auto">
          <Sidebar
            disease={disease}
            state={state}
            city={city}
            cities={cities}
            onDiseaseChange={setDisease}
            onStateChange={setState}
            onCityChange={setCity}
          />

          {/* Top 20 Active Cities */}
          <Card className="mt-4">
            <CardHeader>
              <CardTitle className="text-sm">Top 20 most active cities</CardTitle>
            </CardHeader>
            <CardContent>
              {topCities.map((c: { name_muni: string; transmissao: number; code_muni: number }, i: number) => (
                <div key={i} className="flex justify-between py-1 text-xs">
                  <span className="font-medium">{c.name_muni}</span>
                  <span className="text-muted-foreground">{c.transmissao} weeks</span>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* Weeks Map + Top 10 */}
          <div className="grid grid-cols-3 gap-4">
            <div className="col-span-2">
              {weeksMap && (
                <StateMap
                  geojson={weeksMap}
                  title="Number of weeks of Rt > 1 since 2010"
                  colorField="transmissao"
                  legendTitle="Weeks"
                />
              )}
            </div>
            <div className="col-span-1">
              <TopCitiesTable cities={top10Cities} title="Top 10 cities" />
            </div>
          </div>

          {/* R0 Map + Top R0 + Year Slider */}
          <div className="grid grid-cols-3 gap-4">
            <div className="col-span-2">
              {r0Data && <R0Map geojson={r0Data} year={r0year} />}
            </div>
            <div className="col-span-1 space-y-2">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Top 10 R0s</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="mb-4">
                    <label className="text-xs font-medium">Year: {r0year}</label>
                    <Slider
                      value={[r0year]}
                      onValueChange={(v) => setR0Year(Array.isArray(v) ? v[0] : v)}
                      min={MIN_YEAR}
                      max={CURRENT_YEAR}
                      step={1}
                    />
                  </div>
                  <TopR0Table cities={topR0} />
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Model Evaluation */}
          <div className="grid grid-cols-3 gap-4">
            <div className="col-span-2">
              {modelEval && (
                <ModelEvalMap
                  geojson={weeksMap!}
                  rateMap={modelEval.rateMap}
                  year={modelEvalYear}
                />
              )}
            </div>
            <div className="col-span-1 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Year: {modelEvalYear}</CardTitle>
                </CardHeader>
                <CardContent>
                  <Slider
                    value={[modelEvalYear]}
                    onValueChange={(v) => setModelEvalYear(Array.isArray(v) ? v[0] : v)}
                    min={MIN_YEAR}
                    max={CURRENT_YEAR}
                    step={1}
                  />
                </CardContent>
              </Card>
              {modelEvalRates.length > 0 && (
                <ModelEvalHist rates={modelEvalRates} />
              )}
              {modelEval && (
                <ModelEvalTable data={modelEval.table} />
              )}
            </div>
          </div>

          {/* SIR Parameters + Year Selection + Time Series */}
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">
                  SIR Parameters for {disease} Epidemics in {topCityName}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xs text-muted-foreground mb-2">
                  Description available at{" "}
                  <a
                    href="https://api.mosqlimate.org/docs/datastore/GET/episcanner/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline"
                  >
                    Mosqlimate API
                  </a>
                </p>
                {cityParams.length > 0 && (
                  <SIRParamsTable params={cityParams} />
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Parameters</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-4">
                  <div className="w-48">
                    <Select value={epiYear} onValueChange={(v) => v && setEpiYear(v)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select Year" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All</SelectItem>
                        {cityParams.map((p: { year: number }) => (
                          <SelectItem key={p.year} value={String(p.year)}>
                            {p.year}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardContent>
            </Card>

            <TimeSeriesChart
              data={timeSeries}
              title={`${disease} weekly cases in ${topCityName}`}
            />
          </div>

          {/* Epidemic Calculator */}
          <div className="space-y-4">
            <Card>
              <CardContent className="pt-4">
                <h3 className="text-base font-semibold">
                  SIR Parameters for {disease} Epidemics in {topCityName}
                </h3>
                <p className="text-xs text-muted-foreground mt-1">
                  Interactive epidemic calculator. Adjust the sliders to explore
                  different scenarios.
                </p>
              </CardContent>
            </Card>
            <EpidemicCalculator
              disease={disease}
              city={topCityName}
              dataCumulative={timeSeries.map((d: { casos_cum: number }) => d.casos_cum)}
              dates={timeSeries.map((d: { date: string }) => d.date)}
              initialPeakWeek={medianParams.medianPeak}
              initialR0={medianParams.medianR0}
              initialTotalCases={medianParams.medianCases}
              minCases={medianParams.minCases}
              maxCases={medianParams.maxCases}
              step={medianParams.step}
            />
          </div>
        </div>
      </div>

      <Footer />
    </div>
  );
}
