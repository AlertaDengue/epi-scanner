"use client";

import { useState, useEffect, useMemo, type ReactNode } from "react";
import { DashboardHeader } from "@/components/layout/header";
import { DashboardSidebar } from "@/components/layout/sidebar";
import { StatCards } from "@/components/charts/stat-cards";
import { StateMap } from "@/components/maps/state-map";
import { R0Map } from "@/components/maps/r0-map";
import { ModelEvalMap } from "@/components/maps/model-eval-map";
import { TimeSeriesChart } from "@/components/charts/time-series";
import { ModelEvalHist } from "@/components/charts/model-eval-hist";
import { EpidemicCalculator } from "@/components/charts/epidemic-calculator";
import { RankTable } from "@/components/tables/rank-table";
import { SIRParamsTable } from "@/components/tables/sir-params-table";
import { ModelEvalTable } from "@/components/tables/model-eval-table";
import { Spinner } from "@/components/ui/spinner";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent, CardHeader, CardTitle, CardAction } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CURRENT_YEAR, MIN_YEAR } from "@/lib/constants";

function PanelHeader({
  icon,
  title,
  description,
  action,
}: {
  icon: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2.5">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
          {icon}
        </div>
        <h3 className="text-base font-medium">{title}</h3>
        {action && <div className="ml-auto">{action}</div>}
      </div>
      {description && (
        <p className="mt-0.5 text-sm text-muted-foreground text-pretty">
          {description}
        </p>
      )}
    </div>
  );
}

export default function Dashboard() {
  const [disease, setDisease] = useState("dengue");
  const [state, setState] = useState("RJ");
  const [city, setCity] = useState("");
  const [cities, setCities] = useState<{ geocode: number; name: string }[]>([]);
  const [r0year, setR0Year] = useState(CURRENT_YEAR);
  const [modelEvalYear, setModelEvalYear] = useState(CURRENT_YEAR);
  const [epiYear, setEpiYear] = useState("all");

  const [weeksData, setWeeksData] = useState<{ geocode: number; transmissao: number }[]>([]);
  const [r0MapData, setR0MapData] = useState<{ geocode: number; R0: number }[]>([]);
  const [modelEval, setModelEval] = useState<{ rateMap: { geocode: number; rate: number | null }[]; table: { range: string; count: number; percentage: number }[] } | null>(null);
  const [topCities, setTopCities] = useState<{ name_muni: string; transmissao: number; geocode: number }[]>([]);
  const [top10Cities, setTop10Cities] = useState<{ name_muni: string; transmissao: number; geocode: number }[]>([]);
  const [topR0, setTopR0] = useState<{ name: string; geocode: number; R0: number }[]>([]);
  const [timeSeries, setTimeSeries] = useState<{ date: string; casos: number; casos_cum: number }[]>([]);
  const [sirParams, setSirParams] = useState<{ geocode: number; year: number; disease: string; beta: number; gamma: number; R0: number; peak_week: number; ep_pw: string; ep_ini: string | null; ep_end: string | null; ep_dur: number | null; total_cases: number }[]>([]);
  const [cityParams, setCityParams] = useState<{ geocode: number; year: number; disease: string; beta: number; gamma: number; R0: number; peak_week: number; ep_pw: string; ep_ini: string | null; ep_end: string | null; ep_dur: number | null; total_cases: number }[]>([]);
  const [cases, setCases] = useState(0);
  const [medianParams, setMedianParams] = useState({
    medianR0: 2,
    medianPeak: 10,
    medianCases: 0,
    minCases: 0,
    maxCases: 1000,
    step: 50,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/geolocation")
      .then((r) => r.json())
      .then((d) => { if (d?.uf) setState(d.uf); })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const fetchCities = async () => {
      const res = await fetch(`/api/cities?disease=${disease}&uf=${state}`);
      const data = await res.json();
      setCities(data);
      if (data.length > 0 && !city) {
        const topRes = await fetch(`/api/top-cities?disease=${disease}&uf=${state}&limit=1`);
        const topData = await topRes.json();
        if (topData.length > 0) {
          setCity(String(topData[0].geocode));
        } else {
          setCity(String(data[0].geocode));
        }
      }
    };
    fetchCities();
  }, [disease, state]);

  useEffect(() => {
    const fetchWeeksMap = async () => {
      const [weeksRes, top10Res, top20Res] = await Promise.all([
        fetch(`/api/maps/weeks?disease=${disease}&uf=${state}`),
        fetch(`/api/top-cities?disease=${disease}&uf=${state}&limit=10`),
        fetch(`/api/top-cities?disease=${disease}&uf=${state}&limit=20`),
      ]);
      setWeeksData(await weeksRes.json());
      setTop10Cities(await top10Res.json());
      setTopCities(await top20Res.json());
    };
    fetchWeeksMap();
  }, [disease, state]);

  useEffect(() => {
    const fetchR0 = async () => {
      const res = await fetch(`/api/maps/r0?disease=${disease}&uf=${state}&year=${r0year}`);
      const data = await res.json();
      setR0MapData(data.r0Data);
      setTopR0(data.topR0);
    };
    fetchR0();
  }, [disease, state, r0year]);

  useEffect(() => {
    const fetchModelEval = async () => {
      const res = await fetch(`/api/maps/model-eval?disease=${disease}&uf=${state}&year=${modelEvalYear}`);
      setModelEval(await res.json());
    };
    fetchModelEval();
  }, [disease, state, modelEvalYear]);

  useEffect(() => {
    const fetchParams = async () => {
      const res = await fetch(`/api/parameters?disease=${disease}&uf=${state}`);
      setSirParams(await res.json());
    };
    fetchParams();
  }, [disease, state]);

  useEffect(() => {
    if (!city) return;
    const fetchTimeSeries = async () => {
      const res = await fetch(`/api/timeseries?disease=${disease}&uf=${state}&geocode=${city}&year=${epiYear}`);
      const data = await res.json();
      setTimeSeries(data);
      const totalCases = data.reduce((sum: number, d: { casos: number }) => sum + (d.casos || 0), 0);
      setCases(totalCases);
      const cityP = sirParams.filter((p) => p.geocode === Number(city));
      setCityParams(cityP);
      if (epiYear !== "all" && cityP.length > 0) {
        const yearParam = cityP.find((p) => p.year === Number(epiYear));
        if (yearParam) {
          const pastParams = cityP.filter((p) => p.year < Number(epiYear));
          let medianR0 = 2, medianPeak = 10, medianCases = totalCases;
          if (pastParams.length > 0) {
            const r0Vals = pastParams.map((p) => p.R0).sort((a, b) => a - b);
            const peakVals = pastParams.map((p) => p.peak_week).sort((a, b) => a - b);
            const casesVals = pastParams.map((p) => p.total_cases).sort((a, b) => a - b);
            medianR0 = r0Vals[Math.floor(r0Vals.length / 2)];
            medianPeak = peakVals[Math.floor(peakVals.length / 2)];
            medianCases = casesVals[Math.floor(casesVals.length / 2)];
          }
          const minCases = 0.85 * totalCases;
          const maxCases = Math.max(1.25 * totalCases, 1.25 * medianCases, yearParam.total_cases || 0);
          const step = Math.max(1, Math.floor((maxCases - minCases) / 20));
          setMedianParams({ medianR0, medianPeak: yearParam.peak_week || medianPeak, medianCases: yearParam.total_cases || medianCases, minCases, maxCases, step });
        }
      }
      setLoading(false);
    };
    fetchTimeSeries();
  }, [city, disease, state, epiYear, sirParams]);

  const topCityName = cities.find((c) => String(c.geocode) === city)?.name || "";
  const peakYear = useMemo(() => {
    if (cityParams.length === 0) return CURRENT_YEAR;
    const maxR0 = Math.max(...cityParams.map((p) => p.R0));
    const peak = cityParams.find((p) => p.R0 === maxR0);
    return peak?.year ?? CURRENT_YEAR;
  }, [cityParams]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="text-center">
          <Spinner className="size-8" />
          <p className="text-sm text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader />

      <main className="mx-auto grid max-w-[1600px] grid-cols-1 gap-4 px-4 py-4 md:px-6 lg:grid-cols-[340px_minmax(0,1fr)]">
        <DashboardSidebar
          disease={disease}
          state={state}
          city={city}
          cities={cities}
          topCities={topCities}
          onDiseaseChange={setDisease}
          onStateChange={setState}
          onCityChange={setCity}
        />

        <div className="flex min-w-0 flex-col gap-4">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold tracking-tight text-balance">
                {topCityName || "Select a city"}
              </h2>
              <Badge variant="outline">{state}</Badge>
            </div>
            <p className="text-sm text-muted-foreground text-pretty">
              {disease} epidemiological surveillance data for {state}
            </p>
          </div>

          <StatCards
            cumulativeCases={cases}
            topR0={topR0[0]?.R0 ?? 0}
            peakYear={peakYear}
            state={state}
          />

          <Card>
            <CardHeader>
              <PanelHeader
                icon={<span className="text-xs font-mono text-balance">Rt</span>}
                title="Epidemic weeks — Number of weeks of Rt &gt; 1 since 2010"
              />
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1.5fr_1fr]">
                <div className="min-h-[400px]">
                  {weeksData.length > 0 && (
                    <StateMap
                      data={weeksData}
                      title=""
                      uf={state}
                    />
                  )}
                </div>
                <div>
                  <h4 className="mb-3 text-sm font-semibold">Top 10 cities</h4>
                  <RankTable
                    rows={top10Cities.map((c) => ({ name: c.name_muni, value: c.transmissao }))}
                    valueLabel="Weeks"
                    format={(v) => Math.round(v).toString()}
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <PanelHeader
                icon={<span className="text-xs font-mono text-balance">R₀</span>}
                title="Basic reproduction number (R₀) by city"
              />
              <CardAction>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium tabular-nums">{r0year}</span>
                </div>
              </CardAction>
            </CardHeader>
            <CardContent>
              <div className="mb-4">
                <Slider
                  value={[r0year]}
                  onValueChange={(v) => setR0Year(Array.isArray(v) ? v[0] : v)}
                  min={MIN_YEAR}
                  max={CURRENT_YEAR}
                  step={1}
                />
              </div>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1.5fr_1fr]">
                <div className="min-h-[400px]">
                  {r0MapData.length > 0 && (
                    <R0Map data={r0MapData} year={r0year} uf={state} />
                  )}
                </div>
                <div>
                  <h4 className="mb-3 text-sm font-semibold">Top 10 R₀</h4>
                  <RankTable
                    rows={topR0.map((c) => ({ name: c.name, value: c.R0 }))}
                    valueLabel="R₀"
                    format={(v) => v.toFixed(2)}
                    barColor="var(--color-chart-2)"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <Card>
              <CardHeader>
                <PanelHeader
                  icon={<span className="text-xs font-mono text-balance">Ev</span>}
                  title="Model evaluation"
                  description={`Observed vs estimated cases ratio · ${modelEvalYear}`}
                />
                <CardAction>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium tabular-nums">{modelEvalYear}</span>
                  </div>
                </CardAction>
              </CardHeader>
              <CardContent>
                <div className="mb-4">
                  <Slider
                    value={[modelEvalYear]}
                    onValueChange={(v) => setModelEvalYear(Array.isArray(v) ? v[0] : v)}
                    min={MIN_YEAR}
                    max={CURRENT_YEAR}
                    step={1}
                  />
                </div>
                <div className="min-h-[400px]">
                  {modelEval?.rateMap && modelEval.rateMap.length > 0 ? (
                    <ModelEvalMap data={modelEval.rateMap} year={modelEvalYear} uf={state} />
                  ) : (
                    <div className="flex h-[400px] items-center justify-center text-sm text-muted-foreground">
                      No evaluation data available for {modelEvalYear}
                    </div>
                  )}
                </div>
                {(modelEval?.rateMap?.length ?? 0) > 0 && (
                  <ModelEvalHist rates={modelEval!.rateMap.map((r) => r.rate).filter((r): r is number => r !== null)} />
                )}
                {modelEval?.table && modelEval.table.length > 0 && (
                  <ModelEvalTable data={modelEval.table} />
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <PanelHeader
                  icon={<span className="text-xs font-mono text-balance">TS</span>}
                  title="Time Series & Epidemic Calculator"
                  description={`${disease} weekly cases in ${topCityName}`}
                />
              </CardHeader>
              <CardContent>
                <TimeSeriesChart
                  data={timeSeries}
                  title=""
                />
                <div className="mt-4">
                  <p className="mb-3 text-sm font-medium text-muted-foreground">
                    SIR Parameters for {disease} Epidemics in {topCityName}
                  </p>
                  <div className="mb-4">
                    <Select value={epiYear} onValueChange={(v) => v && setEpiYear(v)}>
                      <SelectTrigger className="w-40">
                        <SelectValue placeholder="Select Year" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All</SelectItem>
                        {cityParams.map((p) => (
                          <SelectItem key={p.year} value={String(p.year)}>
                            {p.year}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  {cityParams.length > 0 && (
                    <SIRParamsTable params={cityParams} />
                  )}
                  <div className="mt-4">
                    <p className="mb-3 text-xs text-muted-foreground">
                      Interactive epidemic calculator. Adjust the sliders to explore different scenarios.
                    </p>
                    <EpidemicCalculator
                      disease={disease}
                      city={topCityName}
                      dataCumulative={timeSeries.map((d) => d.casos_cum)}
                      dates={timeSeries.map((d) => d.date)}
                      initialPeakWeek={medianParams.medianPeak}
                      initialR0={medianParams.medianR0}
                      initialTotalCases={medianParams.medianCases}
                      minCases={medianParams.minCases}
                      maxCases={medianParams.maxCases}
                      step={medianParams.step}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <footer className="border-t pt-4 text-xs text-muted-foreground">
            <p>
              Epi Scanner · Infodengue · FGV/EMA ·{' '}
              <a
                href="https://api.mosqlimate.org/docs/datastore/GET/episcanner/"
                target="_blank"
                rel="noopener noreferrer"
                className="underline underline-offset-4 hover:text-foreground"
              >
                Mosqlimate API
              </a>
            </p>
          </footer>
        </div>
      </main>
    </div>
  );
}
