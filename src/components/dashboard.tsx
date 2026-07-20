"use client";

import { useState, useEffect, useMemo, type ReactNode } from "react";
import { basePath } from "@/lib/base-path";
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
import { richards } from "@/lib/richards";

function PanelHeader({
  icon,
  title,
  description,
  action,
  loading = false,
}: {
  icon: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
  loading?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2.5">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
          {loading ? <Spinner className="size-4" /> : icon}
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
  const [r0yearSlider, setR0YearSlider] = useState(CURRENT_YEAR);
  const [modelEvalYear, setModelEvalYear] = useState(CURRENT_YEAR);
  const [modelEvalYearSlider, setModelEvalYearSlider] = useState(CURRENT_YEAR);
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
    medianCases: 1000,
    minCases: 1,
    maxCases: 10000,
    step: 50,
  });
  const [loadingCities, setLoadingCities] = useState(true);
  const [loadingWeeks, setLoadingWeeks] = useState(true);
  const [loadingTopCities, setLoadingTopCities] = useState(true);
  const [loadingR0, setLoadingR0] = useState(true);
  const [loadingModelEval, setLoadingModelEval] = useState(true);
  const [loadingTimeSeries, setLoadingTimeSeries] = useState(true);

  useEffect(() => {
    fetch(basePath("/api/geolocation"))
      .then((r) => r.json())
      .then((d) => { if (d?.uf) setState(d.uf); })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => setR0Year(r0yearSlider), 300);
    return () => clearTimeout(timer);
  }, [r0yearSlider]);

  useEffect(() => {
    const timer = setTimeout(() => setModelEvalYear(modelEvalYearSlider), 300);
    return () => clearTimeout(timer);
  }, [modelEvalYearSlider]);

  useEffect(() => {
    const fetchCities = async () => {
      setLoadingCities(true);
      const res = await fetch(basePath(`/api/cities?disease=${disease}&uf=${state}`));
      const data = await res.json();
      setCities(data);
      if (data.length > 0 && !city) {
        const topRes = await fetch(basePath(`/api/top-cities?disease=${disease}&uf=${state}&limit=1`));
        const topData = await topRes.json();
        if (topData.length > 0) {
          setCity(String(topData[0].geocode));
        } else {
          setCity(String(data[0].geocode));
        }
      }
      setLoadingCities(false);
    };
    fetchCities();
  }, [disease, state]);

  useEffect(() => {
    const fetchWeeksMap = async () => {
      setLoadingWeeks(true);
      const res = await fetch(basePath(`/api/maps/weeks?disease=${disease}&uf=${state}`));
      setWeeksData(await res.json());
      setLoadingWeeks(false);
    };
    fetchWeeksMap();
  }, [disease, state]);

  useEffect(() => {
    const fetchTopCities = async () => {
      setLoadingTopCities(true);
      const [top10Res, top20Res] = await Promise.all([
        fetch(basePath(`/api/top-cities?disease=${disease}&uf=${state}&limit=10`)),
        fetch(basePath(`/api/top-cities?disease=${disease}&uf=${state}&limit=20`)),
      ]);
      setTop10Cities(await top10Res.json());
      setTopCities(await top20Res.json());
      setLoadingTopCities(false);
    };
    fetchTopCities();
  }, [disease, state]);

  useEffect(() => {
    const fetchR0 = async () => {
      setLoadingR0(true);
      const res = await fetch(basePath(`/api/maps/r0?disease=${disease}&uf=${state}&year=${r0year}`));
      const data = await res.json();
      setR0MapData(data.r0Data);
      setTopR0(data.topR0);
      setLoadingR0(false);
    };
    fetchR0();
  }, [disease, state, r0year]);

  useEffect(() => {
    const fetchModelEval = async () => {
      setLoadingModelEval(true);
      const res = await fetch(basePath(`/api/maps/model-eval?disease=${disease}&uf=${state}&year=${modelEvalYear}`));
      setModelEval(await res.json());
      setLoadingModelEval(false);
    };
    fetchModelEval();
  }, [disease, state, modelEvalYear]);

  useEffect(() => {
    const fetchParams = async () => {
      const res = await fetch(basePath(`/api/parameters?disease=${disease}&uf=${state}`));
      setSirParams(await res.json());
    };
    fetchParams();
  }, [disease, state]);

  useEffect(() => {
    if (!city) return;
    const fetchTimeSeries = async () => {
      setLoadingTimeSeries(true);
      const res = await fetch(basePath(`/api/timeseries?disease=${disease}&uf=${state}&geocode=${city}&year=${epiYear}`));
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
          const maxCases = Math.max(2 * totalCases, 2 * medianCases, yearParam.total_cases || 0);
          const step = Math.max(1, Math.floor((maxCases - minCases) / 20));
          setMedianParams({ medianR0, medianPeak: yearParam.peak_week || medianPeak, medianCases: yearParam.total_cases || medianCases, minCases, maxCases, step });
        }
      } else if (totalCases > 0) {
        const minC = 0.85 * totalCases;
        const maxC = Math.max(2 * totalCases, totalCases + 5000);
        const stp = Math.max(1, Math.floor((maxC - minC) / 20));
        setMedianParams({ medianR0: 2, medianPeak: 10, medianCases: totalCases, minCases: minC, maxCases: maxC, step: stp });
      }
      setLoadingTimeSeries(false);
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

  const selectedYearParam = useMemo(() => {
    if (epiYear === "all") return null;
    return cityParams.find((p) => p.year === Number(epiYear)) ?? null;
  }, [cityParams, epiYear]);

  const modelDataForChart = useMemo(() => {
    if (!selectedYearParam || timeSeries.length === 0) return undefined;
    const { R0: r0, peak_week, total_cases } = selectedYearParam;
    if (!r0 || r0 <= 1) return undefined;
    const r = 1 - 1 / r0;
    const gamma = 0.3;
    const b = (r * gamma) / (1 - r);
    const a = b / (gamma + b);
    return timeSeries.map((d, i) => ({
      date: d.date,
      model: richards(total_cases, a, b, i, peak_week),
    }));
  }, [selectedYearParam, timeSeries]);

  const calculatorDates = useMemo(() => timeSeries.map((d) => d.date), [timeSeries]);
  const calculatorCumulative = useMemo(() => timeSeries.map((d) => d.casos_cum), [timeSeries]);

  const initialLoad = timeSeries.length === 0 && loadingTimeSeries;

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader />

      {initialLoad ? (
        <div className="flex min-h-[calc(100vh-60px)] items-center justify-center bg-background">
          <div className="text-center">
            <Spinner className="size-8" />
            <p className="mt-3 text-sm text-muted-foreground">Loading dashboard...</p>
          </div>
        </div>
      ) : (
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
          loading={loadingTopCities}
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
            loading={loadingTimeSeries || loadingR0}
          />

          <Card>
            <CardHeader>
              <PanelHeader
                icon={<span className="text-xs font-mono">Rt</span>}
                title="Epidemic weeks — Number of weeks of Rt &gt; 1 since 2010"
                loading={loadingWeeks}
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
                  <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold">
                    Top 10 cities
                    {loadingTopCities && <Spinner className="size-3.5" />}
                  </h4>
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
                icon={<span className="text-xs font-mono">R₀</span>}
                title="Basic reproduction number (R₀) by city"
                loading={loadingR0}
              />
              <CardAction>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium tabular-nums">{r0yearSlider}</span>
                </div>
              </CardAction>
            </CardHeader>
            <CardContent>
              <div className="mb-4">
                <Slider
                  value={[r0yearSlider]}
                  onValueChange={(v) => setR0YearSlider(Array.isArray(v) ? v[0] : v)}
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
                  <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold">
                    Top 10 R₀
                    {loadingR0 && <Spinner className="size-3.5" />}
                  </h4>
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
                  icon={<span className="text-xs font-mono">Ev</span>}
                  title="Model evaluation"
                  description={`Observed vs estimated cases ratio · ${modelEvalYearSlider}`}
                  loading={loadingModelEval}
                />
                <CardAction>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium tabular-nums">{modelEvalYearSlider}</span>
                  </div>
                </CardAction>
              </CardHeader>
              <CardContent>
                <div className="mb-4">
                  <Slider
                    value={[modelEvalYearSlider]}
                    onValueChange={(v) => setModelEvalYearSlider(Array.isArray(v) ? v[0] : v)}
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
                      No evaluation data available for {modelEvalYearSlider}
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
                  icon={<span className="text-xs font-mono">TS</span>}
                  title="Time Series & Epidemic Calculator"
                  description={`${disease} weekly cases in ${topCityName}`}
                  loading={loadingTimeSeries}
                />
              </CardHeader>
              <CardContent>
                <TimeSeriesChart
                  data={timeSeries}
                  modelData={modelDataForChart}
                  peakWeekDate={selectedYearParam?.ep_pw ?? null}
                  startDate={selectedYearParam?.ep_ini ?? null}
                  endDate={selectedYearParam?.ep_end ?? null}
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
                    <h4 className="mb-3 text-base font-semibold">
                      Interactive epidemic calculator
                    </h4>
                    <div className="mb-3 flex items-start gap-2 rounded-md border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-700">
                      <svg className="mt-0.5 size-3.5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" /></svg>
                      <span>Adjust the sliders to explore different scenarios.</span>
                    </div>
                    <EpidemicCalculator
                      key={`${city}-${epiYear}`}
                      disease={disease}
                      city={topCityName}
                      dataCumulative={calculatorCumulative}
                      dates={calculatorDates}
                      initialPeakWeek={medianParams.medianPeak}
                      initialR0={medianParams.medianR0}
                      initialTotalCases={medianParams.medianCases}
                      minCases={medianParams.minCases}
                      maxCases={medianParams.maxCases}
                      step={medianParams.step}
                      startDate={selectedYearParam?.ep_ini ?? null}
                      endDate={selectedYearParam?.ep_end ?? null}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <footer className="border-t pt-4 text-xs text-muted-foreground">
            <p>
              Epi Scanner · Infodengue · FGV/EMAp ·{' '}
              <a
                href="https://api.mosqlimate.org/docs/datastore/GET/episcanner/"
                target="_blank"
                rel="noopener noreferrer"
                className="underline underline-offset-4 hover:text-foreground"
              >
                Mosqlimate API
              </a>
              {' · '}
              <a
                href="https://royalsocietypublishing.org/rsos/article/12/5/241261/235685/Large-scale-epidemiological-modelling-scanning-for"
                target="_blank"
                rel="noopener noreferrer"
                className="underline underline-offset-4 hover:text-foreground"
              >
                EpiScanner Article
              </a>
            </p>
          </footer>
        </div>
      </main>
      )}
    </div>
  );
}
