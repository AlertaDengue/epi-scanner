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
import os
import datetime
import warnings
from typing import List
from pathlib import Path
from typing import Optional
import pandas as pd
import duckdb
from epi_scanner.settings import (
    EPISCANNER_DATA_DIR,
    EPISCANNER_DUCKDB_DIR,
    STATES
)
from epi_scanner.viz import (
    load_map,
    plot_series_altair,
    plot_state_map_altair,
    plot_pars_map_altair,
    plot_diff_map_altair,
    t_weeks,
    top_n_cities,
    top_n_R0,
    top_n_diff,
    update_state_map,
)
from h2o_wave import Q, app, copy_expando, data, main, ui  # Noqa F401
from loguru import logger

warnings.filterwarnings("ignore")

DATA_TABLE = None
DUCKDB_FILE = Path(os.path.join(
    str(EPISCANNER_DUCKDB_DIR), "episcanner.duckdb")
)


async def initialize_app(q: Q):
    """
    Set up UI elements
    """
    create_layout(q)
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
    await q.page.save()
    # Setup some client side variables
    q.client.cities = {}
    q.client.loaded = False
    q.client.uf = "SC"
    q.client.disease = "dengue"
    q.client.weeks = False
    await load_map(q)

    await q.page.save()

    q.page["state_header"] = ui.markdown_card(
        box="pre", title=f"", content=f"## Epidemiological Report for {q.client.disease}"
    )
    add_sidebar(q)
    q.page["analysis_header"] = ui.markdown_card(
        box="analysis", title="City-level Analysis", content=""
    )
    year = datetime.date.today().year
    q.page["footer"] = ui.footer_card(
        box="footer",
        caption=(
            f"(c) {year} [Infodengue](https://info.dengue.mat.br). "
            "All rights reserved.\n"
            "Powered by [Mosqlimate](https://mosqlimate.org) & [EpiGraphHub](https://epigraphhub.org/)"
        )
    )
    q.page["form"].items[0].dropdown.value = q.client.disease


@app("/", mode="unicast")
async def serve(q: Q):
    copy_expando(
        q.args, q.client
    )  # Maintain updated copies of q.args in q.client
    if not q.client.initialized:
        await initialize_app(q)
        q.client.initialized = True
    await q.page.save()
    if q.args.disease:
        q.page["form"].items[0].dropdown.value = q.client.disease
        await on_update_disease(q)
        await q.page.save()
    if q.args.state:
        await on_update_UF(q)
        await q.page.save()
    if q.args.city:
        q.page["non-existent"].items = []
        await on_update_city(q)
        await q.page.save()
    if q.args.r0year:
        await update_r0map(q)
        await q.page.save()
    if q.args.diffyear:
        await update_diff(q)
        await q.page.save()
    if "slice_year" in q.args:
        await update_analysis(q)
        await q.page.save()


async def update_sum_cases(
    q: Q,
    start_date: str = '2024-01-01',
    end_date: str = '2024-12-31',
    city: Optional[int] = None
):
    df = q.client.data_table
    df.sort_index(inplace=True)
    if city is not None:
        df = df[df.municipio_geocodigo == city]
    df = df.loc[start_date:end_date]
    return df.casos.sum()


async def update_weeks(q: Q):
    if (not q.client.weeks) and (q.client.data_table is not None):
        await t_weeks(q)
        logger.info("plot weeks")
        fig_alt = await plot_state_map_altair(
            q, q.client.weeks_map, column="transmissao"
        )
        await q.page.save()
        q.page["plot_alt"] = ui.vega_card(
            box="week_map",
            title="",
            specification=fig_alt.to_json(),
        )
        ttext = await top_n_cities(q, 10)
        q.page["wtable"] = ui.form_card(
            box="week_table",
            title="",
            items=[ui.text('**Top 10 cities**'), ui.text(ttext)]
        )


async def update_r0map(q: Q):
    """
    Updates R0 map and table
    """
    end_year = datetime.date.today().year
    year = q.client.r0year or datetime.date.today().year
    fig_alt = await plot_pars_map_altair(
        q, q.client.weeks_map, [year], STATES[q.client.uf]
    )
    await q.page.save()
    q.page["plot_alt_R0"] = ui.vega_card(
        box="R0_map",
        title="",
        specification=fig_alt.to_json(),
    )
    ttext = await top_n_R0(q, year, 10)
    q.page["R0table"] = ui.form_card(
        box="R0_table",
        title="",
        items=[
            ui.text('**Top 10 R0s**'),
            ui.slider(
                name="r0year",
                label="Year",
                min=2010,
                max=end_year,
                step=1,
                value=year,
                trigger=True,
            ),
            ui.text(ttext),
        ],
    )
    await q.page.save()

async def update_diff(q:Q):
    end_year = datetime.date.today().year
    year = q.client.diffyear or (datetime.date.today().year-1)
    fig_alt = await plot_diff_map_altair(
        q, q.client.statemap, [year],STATES[q.client.uf]
    )
    await q.page.save()
    q.page["plot_alt_diff"] = ui.vega_card(
        box = "diff_map",
        title = f"",
        specification = fig_alt.to_json()
    )
    ttext = await top_n_diff(q, year, 10)
    q.page["difftable"] = ui.form_card(
        box="diff_table",
        title="",
        items=[
            ui.text("**Top 10 Differences**"),
            ui.slider(
                name="diffyear",
                label="Year",
                min=2010,
                max=end_year,
                step=1,
                value=year,
                trigger=True,
            ),
            ui.text(ttext),
        ],
    )
    await q.page.save()

async def on_update_disease(q: Q):
    q.client.disease = q.args.disease
    await q.page.save()
    await on_update_UF(q)
    if q.client.city is not None:
        await on_update_city(q)


async def on_update_UF(q: Q):
    logger.info(
        f"\nclient.uf: {q.client.uf}"
        f"\nargs.state: {q.args.state}"
        f"\nargs.city: {q.args.city}"
    )
    if q.args.state is not None:
        q.client.uf = q.args.state
    await load_table(q)

    today_date = datetime.date.today()

    sum_cases = await update_sum_cases(
        q,
        f'{today_date.year}-01-01',
        today_date.strftime('%Y-%m-%d'),
        None
    )

    q.page["state_header"].content = f"## Epidemiological Report for {q.client.disease}\n ## {STATES[q.client.uf]}\nCumulative notified cases since Jan {datetime.date.today().year}: {sum_cases}"

    await q.page.save()
    await update_state_map(q)
    q.client.weeks = False
    await update_weeks(q)
    await q.page.save()

    if DUCKDB_FILE.exists():
        db = duckdb.connect(str(DUCKDB_FILE), read_only=True)
    else:
        raise FileNotFoundError("Duckdb file not found")

    try:
        q.client.parameters = db.execute(
            f"SELECT * FROM '{q.client.uf.upper()}' "
            f"WHERE disease = '{q.client.disease}'"
        ).fetchdf()
    finally:
        db.close()

    dump_results(q)
    await update_r0map(q)
    await update_diff(q)
    await q.page.save()


async def on_update_city(q: Q):
    """
    Prepares the city visualizations
    Args:
        q:
    """
    logger.info(
        f"\nclient.uf: {q.client.uf}",
        f"\nargs.state: {q.args.state}",
        f"\nclient.city: {q.client.city}",
        f"\nargs.city: {q.args.city}",
    )
    if (q.client.city != q.args.city) and (q.args.city is not None):
        q.client.city = q.args.city
    q.page[
        "analysis_header"
    ].content = f"## {q.client.cities[int(q.client.city)]}"
    create_analysis_form(q)
    years = [
        ui.choice(name=str(y), label=str(y))
        for y in q.client.parameters[
            q.client.parameters.geocode == int(q.client.city)
        ].year
    ]
    years.insert(0, ui.choice(name='all', label='All'))
    q.page["years"].items[0].dropdown.choices = years
    await update_analysis(q)
    await q.page.save()


async def update_pars(q: Q):
    table = (
        "| Year | Beta | Gamma | R0 | Peak Week |  Start Week |  End Week | Duration | Cumulative cases estimated| Cumulative cases reported | \n"
        # "|     estimated |  estimated    |    estimated   |  estimated  |   estimated   |  estimated  | estimated |  estimated | estimated | \n"
        "| ---- | ---- | ----- | -- | ---- | ---- | ---- | ---- | ---- | ---- | \n"
    )
    for _, res in q.client.parameters[
        q.client.parameters.geocode == int(q.client.city)
    ].iterrows():
        sum_cases = await update_sum_cases(
            q,
            f"{int(res['year'])-1}-11-01",
            f"{int(res['year'])}-11-01",
            int(q.client.city)
        )
        table += (
            f"| {int(res['year'])} | {res['beta']:.2f} "
            f"| {res['gamma']:.2f} | {res['R0']:.2f} "
            f"| {int(res['ep_pw'])} | {int(res['ep_ini'])} "
            f"| {int(res['ep_end'])} | {int(res['ep_dur'])} "
            f"| {int(res['total_cases'])} | {sum_cases} | \n"
        )
    q.page["sir_pars"].items[2].text.content = table
    await q.page.save()


def create_layout(q):
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
                                        ]
                                    ),
                                    ui.zone(
                                        name="diff",
                                        direction=ui.ZoneDirection.ROW,
                                        zones=[
                                            ui.zone("diff_map", size="65%"),
                                            ui.zone("diff_table", size="35%"),
                                        ]
                                    ),
                                    ui.zone(
                                        name="analysis",
                                        direction=ui.ZoneDirection.COLUMN,
                                        zones=[
                                            ui.zone("SIR parameters"),
                                            ui.zone("Year"),
                                            ui.zone("SIR curves", size="100%"),
                                        ]
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ],
    )


def df_to_table_rows(df: pd.DataFrame) -> List[ui.TableRow]:
    return [
        ui.table_row(name=str(r[0]), cells=[str(r[0]), r[1]])
        for r in df.itertuples(index=False)
    ]


async def load_table(q: Q):
    global DATA_TABLE
    UF = q.client.uf
    disease = q.client.disease

    if disease == "chik":
        disease = "chikungunya"

    if os.path.exists(f"{EPISCANNER_DATA_DIR}/{UF}_{disease}.parquet"):
        logger.info("loading data...")
        DATA_TABLE = pd.read_parquet(
            f"{EPISCANNER_DATA_DIR}/{UF}_{disease}.parquet"
        )
        q.client.data_table = DATA_TABLE
        q.client.loaded = True
        for gc in DATA_TABLE.municipio_geocodigo.unique():
            try:  # FIXME: this is a hack to deal with missing cities in the map
                city_name = q.client.brmap[
                    q.client.brmap.code_muni.astype(int) == int(gc)
                ].name_muni.values
                q.client.cities[int(gc)] = '' if not city_name.any(
                ) else city_name[0]
            except IndexError:
                pass  # If city is missing in the map, ignore it
        choices = [
            ui.choice(str(gc), q.client.cities[gc])
            for gc in DATA_TABLE.municipio_geocodigo.unique()
        ]
        q.page["form"].items[2].dropdown.choices = choices
        q.page["form"].items[2].dropdown.visible = True

    await q.page.save()


async def update_analysis(q):
    if (q.client.epi_year is None) or (q.client.epi_year == 'all'):
        syear = 2011
        eyear = datetime.date.today().year
    else:
        syear = eyear = q.client.epi_year
    altair_plot = await plot_series_altair(
        q, int(q.client.city), f"{int(syear)-1}-11-01", f"{eyear}-11-01"
    )
    q.page["ts_plot_alt"] = ui.vega_card(
        box="SIR curves",
        title='',
        specification=altair_plot.to_json()
    )
    await q.page.save()
    await update_pars(q)


def dump_results(q):
    """
    Dump top 20 cities to markdown list
    Args:
        q:
    """
    results = "**Top 20 most active cities** \n\n"
    cities = q.client.parameters.groupby("geocode")
    report = {}
    for gc, citydf in cities:
        if len(citydf) < 1:
            continue
        report[
            q.client.cities[gc]
        ] = f"{len(citydf)} epidemic years: {list(sorted(citydf.year))}\n"
    for n, linha in sorted(report.items(), key=lambda x: x[1], reverse=True)[:20]:
        results += f"**{n}** :{linha}\n"
    q.page["results"].content = results


def add_sidebar(q):
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
        box="sidebar",
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
                placeholder="Nothing selected"
            ),
            ui.text(
                'The parameters table can be downloaded in the [Mosqlimate API](https://api.mosqlimate.org/docs/datastore/GET/episcanner/).')
        ],
    )
    q.page["results"] = ui.markdown_card(
        box="sidebar", title="", content="",
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
        title='',
        items=[ui.text_l(content=f'<h1 style="font-size:18px;">{title}</h1>'),
               ui.text(
                   content='A description of each parameter below is available in the [Mosqlimate API](https://api.mosqlimate.org/docs/datastore/GET/episcanner/).'),
               ui.text(name="sirp_table", content="")],
    )
