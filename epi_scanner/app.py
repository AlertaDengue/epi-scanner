"""
This app analyzes dengue incidence
across space and time in the south of Brasil.

To facilitate customization, we list below key information about the code.
To avoid reloading data from disk the app maintains the following
 cached objects under `q.client`:

- q.client.brmap: Map of Brasil. GeoDataFrame.
- q.client.statemap: map of the currently selected state. GeoDataFrame.
- q.client.weeks_map: map of the state merged with the total number
 of transmission weeks. GeoDataFrame.
- q.client.cities: dictionary of city names by geocode.
- q.client.scanner: EpiScanner object
- q.client.city: geocode of the currently selected city.
- q.client.uf: two-letter code for the currently selected state.
- q.client.disease: name of the currently selected disease.
- q.client.parameters: SIR parameters for every city/year in current state.
"""
import asyncio
import datetime
import os
import warnings
import concurrent.futures
from pathlib import Path
from typing import Optional, Literal
from functools import lru_cache
from threading import Event

import duckdb
import numpy as np
import pandas as pd
import geopandas as gpd
from epi_scanner.elements import cards, charts
from epi_scanner.settings import EPISCANNER_DATA_DIR, EPISCANNER_DUCKDB_DIR, STATES
from epi_scanner.viz import (
    get_ini_end_week,
    load_map,
    markdown_table,
    plot_epidemic_calc_altair,
    plot_model_evaluation_hist_altair,
    plot_model_evaluation_map_altair,
    pars_map_chart,
    plot_series_altair,
    state_map_chart,
    client_weeks_map,
    table_model_evaluation,
    top_n_cities,
    top_n_cities_md,
    top_n_R0_md,
    client_state_map,
)
from h2o_wave import Q, app, copy_expando, data, main, ui  # Noqa F401
from loguru import logger

warnings.filterwarnings("ignore")

DUCKDB_FILE = Path(
    os.path.join(str(EPISCANNER_DUCKDB_DIR), "episcanner.duckdb")
)


@app("/", mode="unicast")
async def serve(q: Q):
    if not q.client.initialized:
        q.client.cities = {}
        q.client.uf = "CE"
        q.client.disease = "dengue"
        q.client.weeks = False
        q.client.event = Event()
        q.client.r0year = datetime.date.today().year

        await create_layout(q)
        await add_sidebar(q)
        await initialize_app(q, disease=q.client.disease, uf=q.client.uf)
        q.client.initialized = True

    if q.args.disease and q.args.disease != q.client.disease:
        if q.client and q.client.set:
            q.client.set()
        q.client.disease = q.args.disease
        await on_update_disease(q, q.client.disease)

    if q.args.state and q.args.state != q.client.state:
        if q.client and q.client.set:
            q.client.set()
        q.client.uf = q.args.state
        await on_update_UF(q, q.client.uf)

    if q.args.city:
        if q.client and q.client.set:
            q.client.set()
        q.client.city = q.args.city
        # await on_update_city(q, q.client.city)

    if q.args.r0year and int(q.args.r0year or 0) != q.client.r0year:
        if q.client and q.client.set:
            q.client.set()
        q.client.r0year = int(q.args.r0year)
        await update_r0map(q, year=q.client.r0year)

    await q.page.save()


async def initialize_app(q: Q, disease: str, uf: str):
    q.page["title"] = ui.header_card(
        box=ui.box("header"),
        title="Real-time Epidemic Scanner",
        subtitle="Real-time epidemiology",
        color="primary",
        image=(
            "https://info.dengue.mat.br/static/"
            "img/info-dengue-logo-multicidades.png"
        ),
    )

    brmap = await q.run(load_map)
    q.client.brmap = brmap

    cards.StateHeader(q)

    year = datetime.date.today().year
    q.page["footer"] = ui.footer_card(
        box="footer",
        caption=(
            f"(c) {year} [Infodengue](https://info.dengue.mat.br). "
            "All rights reserved.\n"
            "Powered by [Mosqlimate](https://mosqlimate.org) & "
            "[EpiGraphHub](https://epigraphhub.org/)"
        ),
    )

    q.page["form"].items[0].dropdown.value = disease
    q.page["form"].items[1].dropdown.value = uf
    q.page["form"].items[2].dropdown.value = q.client.uf

    await on_update_UF(q, uf=uf)
    await on_update_city(q, geocode=0)


async def on_update_disease(
    q: Q,
    disease: str,
    date: datetime.date = datetime.date.today()
):
    uf = q.client.uf

    # fetch client variables
    await client_data_table(q, disease=disease, uf=uf)
    await client_state_map(q, uf=uf)
    await client_cities(q)
    await client_weeks_map(q, q.client.data_table, q.client.statemap)
    await client_parameters(q, disease=disease, uf=uf)

    # update layout
    await update_sidebar(q, disease=disease, uf=uf)
    await update_header(q, disease=disease, date=date)
    await update_weeks_map(q, weeks_map=q.client.weeks_map)
    await update_results(q, parameters=q.client.parameters)
    await update_r0map(q, year=q.client.r0year)


async def on_update_UF(
    q: Q,
    uf: str,
    date: datetime.date = datetime.date.today()
):
    disease = q.client.disease
    await client_data_table(q, disease=disease, uf=uf)
    await client_state_map(q, uf=uf)
    await client_cities(q)
    await client_weeks_map(q, q.client.data_table, q.client.statemap)
    await client_cities(q)
    await client_parameters(q, disease=disease, uf=uf)

    await update_sidebar(q, disease=disease, uf=uf)
    await update_header(q, disease=disease, date=date)
    await update_weeks_map(q, weeks_map=q.client.weeks_map)
    await update_results(q, parameters=q.client.parameters)
    await update_r0map(q, year=q.client.r0year)

    # await update_model_evaluation(q)


async def sum_cases(
    q: Q,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    geocode: Optional[int] = None,
):
    df = q.client.data_table
    df.sort_index(inplace=True)
    if geocode is not None:
        df = df[df.municipio_geocodigo == geocode]
    df = df.loc[start_date:end_date]
    return df.casos.sum()


async def update_weeks_map(q: Q, weeks_map: gpd.GeoDataFrame):
    plot_alt = cards.Vega(q, "plot_alt", "week_map", title="")

    q.page["wtable"] = ui.form_card(
        box="week_table",
        title="",
        items=[
            ui.text("**Top 10 cities**"),
        ],
    )

    if q.client.event.is_set():
        return

    chart = state_map_chart(q, weeks_map)
    plot_alt.update(q, chart)

    q.page["wtable"] = ui.form_card(
        box="week_table",
        title="",
        items=[
            ui.text("**Top 10 cities**"),
            ui.text(top_n_cities_md(top_n_cities(weeks_map, 10)))
        ],
    )


async def update_results(q: Q, parameters: pd.DataFrame):
    report = {}
    for gc, citydf in parameters.groupby("geocode"):
        if len(citydf) < 1:
            continue
        report[
            q.client.cities[gc]
        ] = f"{len(citydf)} epidemic years: {list(sorted(citydf.year))}\n"
    results = sorted(
        report.items(),
        key=lambda x: int(x[1][0:2]),
        reverse=True
    )[:20]
    cards.Results.update(q, results=results)


async def update_r0map(q: Q, year: int = datetime.date.today().year):
    """
    Updates R0 map and table
    """
    end_year = datetime.date.today().year
    chart = pars_map_chart(q.client.weeks_map, q.client.parameters, year)
    cards.Vega(q, "plot_alt_R0", "R0_map", title="", chart=chart)

    q.page["R0table"] = ui.form_card(
        box="R0_table",
        title="",
        items=[
            ui.text("**Top 10 R0s**"),
            ui.slider(
                name="r0year",
                label="Year",
                min=2010,
                max=end_year,
                step=1,
                value=year,
                trigger=True,
            ),
            ui.text(top_n_R0_md(q, year, 10)),
        ],
    )


async def update_model_evaluation(q: Q):
    end_year = datetime.date.today().year
    year = q.client.model_evaluation_year or (datetime.date.today().year - 1)
    fig_alt = await plot_model_evaluation_map_altair(
        q, q.client.statemap, [year], STATES[q.client.uf]
    )
    q.page["map_alt_model_evaluation"] = ui.vega_card(
        box="model_evaluation_map", title="", specification=fig_alt.to_json()
    )
    fig_alt = await plot_model_evaluation_hist_altair(
        q, q.client.statemap, [year], STATES[q.client.uf]
    )
    q.page["hist_alt_model_evaluation"] = ui.vega_card(
        box="model_evaluation_hist", title="", specification=fig_alt.to_json()
    )
    q.page["timeslide_evaluation_model"] = ui.form_card(
        box="model_evaluation_time",
        title="",
        items=[
            ui.slider(
                name="model_evaluation_year",
                label="Year",
                min=2010,
                max=end_year,
                step=1,
                value=year,
                trigger=True,
            ),
        ],
    )
    table = await table_model_evaluation(q, year)
    q.page["table_model_evaluation"] = ui.form_card(
        box="model_evaluation_table",
        items=[ui.text(table)],
    )


async def on_update_city(q: Q, geocode: int):
    """
    Prepares the city visualizations
    Args:
        q:
    """
    # create_analysis_form(q)
    # years = [
    #     ui.choice(name=str(y), label=str(y)) for y
    #     in q.client.parameters[q.client.parameters.geocode == geocode].year
    # ]
    # years.insert(0, ui.choice(name="all", label="All"))
    # q.page["years"].items[0].dropdown.choices = years
    # await update_analysis(q)
    #
    # title = (
    #     f"SIR Parameters for {q.client.disease} Epidemics in "
    #     f"{q.client.cities[geocode]}"
    # )
    #
    # q.page["epidemic_calc_header_"] = ui.form_card(
    #     box="epidemic_calc_header",
    #     title="",
    #     items=[
    #         ui.inline(
    #             items=[
    #                 ui.text_l(
    #                     content=f'<h1 style="font-size:18px;">{title}</h1>'
    #                 ),
    #             ]
    #         ),
    #         ui.text(
    #             content=(
    #                 "The section below displays the cumulative cases for the "
    #                 "selected city in blue, the Richards model in orange, and "
    #                 "the peak week in red. The sliders allow you to adjust "
    #                 "the peak week, reproduction number (R0), and total "
    #                 "number of cases. The orange curve represents just one "
    #                 "possible scenario for the evolution of the epidemic curve."
    #                 "The table shows the parameters estimated for the current "
    #                 "year. If no parameter values are displayed in the table, "
    #                 "it means that the series has not met the necessary "
    #                 "requirements for value estimation."
    #             )
    #         ),
    #     ],
    # )
    #
    # df_pars = q.client.parameters
    # df_pars = df_pars.loc[
    #     (df_pars.geocode == geocode)
    #     & (df_pars.year == int(datetime.date.today().year))
    # ]
    #
    # df_pars_ = pd.DataFrame()
    # df_pars_["pars"] = ["Peak week", "R0", "Total cases"]
    #
    # if df_pars.empty:
    #     df_pars_["values"] = ["---", "---", "---"]
    # else:
    #     df_pars_["values"] = [
    #         int(df_pars["peak_week"].values[0]),
    #         round(df_pars["R0"].values[0], 2),
    #         int(df_pars["total_cases"].values[0]),
    #     ]
    #
    # table = markdown_table(
    #     fields=["Parameter", "Value"],
    #     rows=df_pars_.values.tolist(),
    # )
    #
    # q.page["table_ep_calc_pars"] = ui.form_card(
    #     box="ep_calc_pars_table",
    #     items=[ui.text(table)],
    # )
    #
    # await on_update_ini_epi_calc(q)


async def update_pars(q: Q):
    table = (
        "| Year | Beta | Gamma | R0 | Peak Week |  Start Week |  End Week | Duration | Cumulative cases estimated| Cumulative cases reported | \n"
        # "|     estimated |  estimated    |    estimated   |  estimated  |   estimated   |  estimated  | estimated |  estimated | estimated | \n"
        "| ---- | ---- | ----- | -- | ---- | ---- | ---- | ---- | ---- | ---- | \n"
    )
    for _, res in q.client.parameters[
        q.client.parameters.geocode == int(q.client.city)
    ].iterrows():
        start_date, end_date = get_ini_end_week(int(res["year"]))

        cases = await sum_cases(
            q,
            start_date,
            end_date,
            int(q.client.city),
        )
        table += (
            f"| {int(res['year'])} | {res['beta']:.2f} "
            f"| {res['gamma']:.2f} | {res['R0']:.2f} "
            f"| {int(res['ep_pw'])} | {int(res['ep_ini'])} "
            f"| {int(res['ep_end'])} | {int(res['ep_dur'])} "
            f"| {int(res['total_cases'])} | {cases} | \n"
        )
    q.page["sir_pars"].items[2].text.content = table


async def get_median_pars(q: Q):

    year = int(datetime.datetime.today().year)

    start_date, end_date = get_ini_end_week(year)

    cases = await sum_cases(
        q,
        start_date,
        end_date,
        int(q.client.city),
    )

    cases = float(cases)

    df_pars = q.client.parameters
    df_pars_ = df_pars.loc[
        (df_pars.geocode == int(q.client.city)) & (df_pars.year < year)
    ]
    df_pars_atual = df_pars.loc[
        (df_pars.geocode == int(q.client.city)) & (df_pars.year == year)
    ]

    if df_pars_.empty == True:
        median_R0 = 2
        median_peak = 10
        median_cases = cases
    else:
        median_R0 = round(np.median(df_pars_["R0"].values), 2)
        median_peak = int(np.median(df_pars_["peak_week"].values))
        median_cases = int(np.median(df_pars_["total_cases"].values))

    min_cases = 0.85 * cases

    if df_pars_atual.empty == True:
        max_cases = max(1.25 * cases, 1.25 * median_cases)
    else:
        max_cases = max(
            1.25 * cases,
            1.25 * median_cases,
            df_pars_atual["total_cases"].values[0],
        )

    step = int((max_cases - min_cases) / 20)

    return median_R0, median_peak, median_cases, min_cases, max_cases, step


async def on_update_ini_epi_calc(q: Q):
    (
        median_R0,
        median_peak,
        median_cases,
        min_cases,
        max_cases,
        step,
    ) = await get_median_pars(q)

    q.page["peak_model"] = ui.form_card(
        box="ep_calc_peak",
        title="",
        items=[
            ui.slider(
                name="ep_peak_week",
                label="Peak week",
                min=5,
                max=45,
                step=1,
                value=median_peak,
                trigger=True,
            ),
        ],
    )

    q.page["r0_model"] = ui.form_card(
        box="ep_calc_R0",
        title="",
        items=[
            ui.slider(
                name="ep_R0",
                label="R0",
                min=0.1,
                max=5,
                step=0.05,
                value=median_R0,
                trigger=True,
            ),
        ],
    )

    q.page["total_model"] = ui.form_card(
        box="ep_calc_total",
        title="",
        items=[
            ui.slider(
                name="ep_total",
                label="Total cases",
                min=min_cases,
                max=max_cases,
                step=step,
                value=median_cases,
                trigger=True,
            ),
        ],
    )

    altair_plot = charts.EpidemicCalculator(
        disease=q.client.disease,
        city=q.client.cities[int(q.client.city)],
        df=q.client.data_table,
        gc=int(q.client.city),
        pw=median_peak,
        R0=median_R0,
        total_cases=median_cases,
    )

    q.page["epidemic_calc"] = ui.vega_card(
        box="epi_calc_alt", title="", specification=altair_plot.chart.to_json()
    )

    q.args.ep_peak_week = False
    q.args.ep_R0 = False
    q.args.ep_total = False


async def create_layout(q):
    """
    Creates the main layout of the app
    """
    q.page["meta"] = ui.meta_card(
        box="",
        icon="https://info.dengue.mat.br/static/img/favicon.ico",
        title="Realtime Epi Report",
        theme="default",
        layouts=[
            ui.layout(
                breakpoint="xs",
                width="1200px",
                min_height="100vh",
                zones=[
                    ui.zone("header"),
                    ui.zone(name="footer"),
                    ui.zone(
                        "body",
                        direction=ui.ZoneDirection.ROW,
                        zones=[
                            ui.zone(
                                "sidebar",
                                size="25%",
                                direction=ui.ZoneDirection.COLUMN,
                                zones=[
                                    ui.zone("sidebar_form"),
                                    ui.zone("sidebar_results", size="1605px"),
                                ],
                            ),
                            ui.zone(
                                "content",
                                size="75%",
                                direction=ui.ZoneDirection.COLUMN,
                                zones=[
                                    ui.zone("pre"),
                                    ui.zone(
                                        name="week_zone",
                                        direction=ui.ZoneDirection.ROW,
                                        zones=[
                                            ui.zone("week_map", size="65%"),
                                            ui.zone("week_table", size="35%"),
                                        ],
                                    ),
                                    ui.zone(
                                        name="R0_zone",
                                        direction=ui.ZoneDirection.ROW,
                                        zones=[
                                            ui.zone("R0_map", size="65%"),
                                            ui.zone("R0_table", size="35%"),
                                        ],
                                    ),
                                    ui.zone(
                                        name="model_evaluation",
                                        direction=ui.ZoneDirection.ROW,
                                        zones=[
                                            ui.zone(
                                                "model_evaluation_map",
                                                size="65%",
                                            ),
                                            ui.zone(
                                                name="model_evaluation_column",
                                                size="35%",
                                                direction=ui.ZoneDirection.COLUMN,
                                                zones=[
                                                    ui.zone(
                                                        "model_evaluation_time"
                                                    ),
                                                    ui.zone(
                                                        "model_evaluation_hist"
                                                    ),
                                                    ui.zone(
                                                        "model_evaluation_table"
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    ui.zone(
                                        name="analysis",
                                        direction=ui.ZoneDirection.COLUMN,
                                        zones=[
                                            ui.zone("SIR parameters"),
                                            ui.zone("Year"),
                                            ui.zone(
                                                "SIR curves", size="600px"
                                            ),
                                        ],
                                    ),
                                    ui.zone("epidemic_calc_header"),
                                    ui.zone(
                                        name="epidemic_calc",
                                        direction=ui.ZoneDirection.ROW,
                                        zones=[
                                            ui.zone(
                                                "epi_calc_alt", size="77%"
                                            ),
                                            ui.zone(
                                                name="epic_calc_column",
                                                size="23%",
                                                direction=ui.ZoneDirection.COLUMN,
                                                zones=[
                                                    ui.zone("ep_calc_peak"),
                                                    ui.zone("ep_calc_R0"),
                                                    ui.zone("ep_calc_total"),
                                                    ui.zone(
                                                        "ep_calc_pars_table"
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ],
    )


@lru_cache
def read_data(disease: str, uf: str) -> pd.DataFrame:
    return pd.read_parquet(
        f"{EPISCANNER_DATA_DIR}/{uf}_{disease}.parquet"
    )


@lru_cache
def read_duckdb(disease: str, uf: str) -> pd.DataFrame:
    if DUCKDB_FILE.exists():
        db = duckdb.connect(str(DUCKDB_FILE), read_only=True)
    else:
        raise FileNotFoundError("Duckdb file not found")
    try:
        return db.execute(
            f"SELECT * FROM '{uf.upper()}' "
            f"WHERE disease = '{disease}'"
        ).fetchdf()
    finally:
        db.close()


async def client_data_table(q: Q, disease: Literal["dengue", "chik"], uf: str):
    if disease == "chik":
        disease = "chikungunya"
    df = await q.run(read_data, disease, uf)
    q.client.data_table = df


async def client_parameters(q: Q, disease: Literal["dengue", "chik"], uf: str):
    if disease == "chik":
        disease = "chikungunya"
    df = await q.run(read_duckdb, disease, uf)
    q.client.parameters = df


async def update_analysis(q):
    if (q.client.epi_year is None) or (q.client.epi_year == "all"):
        syear = 2011
        eyear = datetime.date.today().year
    else:
        syear = eyear = q.client.epi_year

    start_date, end_date = get_ini_end_week(int(syear), eyear)

    altair_plot = await plot_series_altair(
        q, int(q.client.city), start_date, end_date
    )
    q.page["ts_plot_alt"] = ui.vega_card(
        box="SIR curves", title="", specification=altair_plot.to_json()
    )
    await update_pars(q)


async def add_sidebar(q):
    state_choices = [
        ui.choice("AC", "Acre"),
        ui.choice("AL", "Alagoas"),
        ui.choice("AM", "Amazonas"),
        ui.choice("AP", "Amapá"),
        ui.choice("BA", "Bahia"),
        ui.choice("CE", "Ceará"),
        ui.choice("DF", "Distrito Federal"),
        ui.choice("ES", "Espírito Santo"),
        ui.choice("GO", "Goiás"),
        ui.choice("MA", "Maranhão"),
        ui.choice("MG", "Minas Gerais"),
        ui.choice("MS", "Mato Grosso do Sul"),
        ui.choice("MT", "Mato Grosso"),
        ui.choice("PA", "Pará"),
        ui.choice("PB", "Paraíba"),
        ui.choice("PE", "Pernambuco"),
        ui.choice("PI", "Piauí"),
        ui.choice("PR", "Paraná"),
        ui.choice("RJ", "Rio de Janeiro"),
        ui.choice("RN", "Rio Grande do Norte"),
        ui.choice("RO", "Rondônia"),
        ui.choice("RR", "Roraima"),
        ui.choice("RS", "Rio Grande do Sul"),
        ui.choice("SC", "Santa Catarina"),
        ui.choice("SE", "Sergipe"),
        ui.choice("SP", "São Paulo"),
        ui.choice("TO", "Tocantins"),
    ]
    q.page["form"] = ui.form_card(
        box="sidebar_form",
        items=[
            ui.dropdown(
                name="disease",
                label="Select disease",
                required=True,
                choices=[
                    ui.choice("dengue", "Dengue"),
                    ui.choice("chik", "Chikungunya"),
                ],
                trigger=True,
            ),
            ui.dropdown(
                name="state",
                label="Select state",
                required=True,
                choices=state_choices,
                trigger=True,
            ),
            ui.dropdown(
                name="city",
                label="Select city",
                required=True,
                choices=[],
                trigger=True,
                visible=False,
                popup="always",
                placeholder="Nothing selected",
            ),
            ui.text(
                "The parameters table can be downloaded in the [Mosqlimate API](https://api.mosqlimate.org/docs/datastore/GET/episcanner/)."
            ),
        ],
    )
    cards.Results(q)


async def client_cities(q: Q):
    geocodes = set(q.client.data_table.municipio_geocodigo.unique())
    for gc in geocodes:
        city_data = q.client.brmap[
            q.client.brmap.code_muni.astype(int) == int(gc)
        ]
        if city_data.empty:
            q.client.cities[int(gc)] = ""
        else:
            q.client.cities[int(gc)] = city_data.name_muni.values[0]


async def update_sidebar(q: Q, disease: str, uf: str):
    geocodes = set(q.client.data_table.municipio_geocodigo.unique())
    q.page["form"].items[0].dropdown.value = disease
    q.page["form"].items[1].dropdown.value = uf

    choices = [ui.choice(str(gc), q.client.cities[gc]) for gc in geocodes]
    q.page["form"].items[2].dropdown.choices = choices
    q.page["form"].items[2].dropdown.visible = True
    q.page["form_city"].items[0].dropdown.choices = choices
    q.page["form_city"].items[0].dropdown.visible = True


async def update_header(q: Q, disease: str, date: datetime.date):
    cases = await sum_cases(q, f"{date.year}-01-01", date.strftime("%Y-%m-%d"))
    disease = "chikungunya" if disease == "chik" else disease
    cards.StateHeader.update(
        q=q,
        disease=disease,
        uf=q.client.uf,
        cases=cases,
        year=date.year,
    )


def create_analysis_form(q):
    q.page["years"] = ui.form_card(
        box="Year",
        title="Parameters",
        items=[
            ui.dropdown(name="epi_year", label="Select Year", required=True),
            ui.button(name="slice_year", label="Update"),
        ],
    )
    title = (
        f"SIR Parameters for {q.client.disease} Epidemics in "
        f"{q.client.cities[int(q.client.city)]}"
    )
    q.page["sir_pars"] = ui.form_card(
        box="SIR parameters",
        title="",
        items=[
            ui.text_l(content=f'<h1 style="font-size:18px;">{title}</h1>'),
            ui.text(
                content="A description of each parameter below is available in the [Mosqlimate API](https://api.mosqlimate.org/docs/datastore/GET/episcanner/)."
            ),
            ui.text(name="sirp_table", content=""),
        ],
    )
