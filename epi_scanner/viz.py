import asyncio
import datetime
import os
import uuid
from pathlib import Path
from functools import lru_cache

import altair as alt
import geopandas as gpd
import gpdvega  # NOQA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from altair import datum
from epi_scanner.settings import EPISCANNER_DATA_DIR
from epiweeks import Week
from h2o_wave import Q
from plotly import io as pio
from plotly.subplots import make_subplots


def get_ini_end_week(year: int, eyear=None):
    """
    Returns the start and end dates used in the optimization process in the
    'episcanner-downloader' repository.

    Parameters:
    year (int): The year for which the start and end dates are retrieved.

    Returns:
    Tuple[datetime, datetime]: The start and end dates of the specified year.
    """
    ini_week = Week(year - 1, 1).startdate()

    dates = pd.date_range(start=ini_week, periods=104, freq="W-SUN")

    dates_ = dates[dates.year >= year - 1][44: 44 + 52]

    ini_date = dates_[0].strftime("%Y-%m-%d")

    if eyear is None:
        end_date = dates_[-1].strftime("%Y-%m-%d")
    else:
        end_date = f"{eyear}-11-01"

    return ini_date, end_date


def load_map() -> gpd.GeoDataFrame:
    file_gpkg = Path(f"{EPISCANNER_DATA_DIR}/muni_br.gpkg")
    return gpd.read_file(file_gpkg, driver="GPKG")


def weeks_map_df(
    data_table: pd.DataFrame,
    statemap: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    weeks = data_table.groupby(by="municipio_geocodigo").sum(
        numeric_only=True
    )[["transmissao"]]
    return statemap.merge(weeks, left_on="code_muni", right_index=True)


async def client_state_map(q: Q, uf: str):
    q.client.statemap = q.client.brmap[q.client.brmap.abbrev_state == uf]


async def client_weeks_map(
    q: Q,
    data_table: pd.DataFrame,
    statemap: gpd.GeoDataFrame
):
    q.client.weeks_map = weeks_map_df(data_table, statemap)


async def client_rate_map(
    q: Q,
    years: list,
    statemap: gpd.GeoDataFrame,
    data_table: pd.DataFrame,
    pars: pd.DataFrame,
):
    q.client.rate_map = rate_map(
        years=years, statemap=statemap, data_table=data_table, pars=pars
    )


async def client_city(q: Q, geocode: int):
    q.client.city = geocode


def state_map_chart(
    q: Q,
    weeks_map: gpd.GeoDataFrame,
) -> alt.Chart:
    if q.client.event.is_set():
        return

    spec = (
        alt.Chart(weeks_map)
        .mark_geoshape()
        .encode(
            color=alt.Color(
                "transmissao:Q",
                sort="ascending",
                scale=alt.Scale(
                    scheme="bluepurple"
                ),  # , domain = [-0.999125,41.548309]),
                legend=alt.Legend(
                    title="Weeks",
                    orient="bottom",
                    tickCount=10,
                ),
            ),
            tooltip=["name_muni", "transmissao:Q"],
        )
        .properties(
            title={
                "text":
                "Number of weeks of Rt > 1 since 2010",
                "fontSize": 15,
                "anchor": "start"
            },
            width=500,
            height=400,
        )
    )
    return spec


def rate_map(
    years: list,
    statemap: gpd.GeoDataFrame,
    data_table: pd.DataFrame,
    pars: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Merge map with rate between cases and estimated cases
    Args:
        years: list of one or more years to be selected
        statemap: map to be merged
        data: data tables
        pars: parameters table

    Returns:

    """
    year = years[0]
    df = data_table[data_table["SE"].isin(
        range((year - 1) * 100 + 45, year * 100 + 45)
    )]
    casos = (
        df[["municipio_geocodigo", "casos"]]
        .groupby("municipio_geocodigo")
        .sum()
        .rename(columns={"casos": "observed_cases"})
    )

    map_rate = statemap.merge(
        casos, left_on="code_muni", right_index=True, how="outer"
    ).merge(
        pars[pars.year.astype(int).isin(years)],
        left_on="code_muni",
        right_on="geocode",
        how="outer",
    )[
        ["name_muni", "geometry", "observed_cases", "total_cases"]
    ]

    map_rate["rate"] = map_rate["observed_cases"] / map_rate["total_cases"]
    return map_rate


def pars_map_chart(
    themap: gpd.GeoDataFrame,
    parameters: pd.DataFrame,
    year: int,
) -> alt.Chart:
    map_pars = themap.merge(
        parameters[parameters.year.astype(int).isin([year])],
        left_on="code_muni",
        right_on="geocode",
        how="outer",
    ).fillna(0)[["geometry", "year", "name_muni", "R0"]]

    return alt.Chart(
        data=map_pars,
        padding={"left": 0, "top": 0, "right": 0, "bottom": 0},
    ).mark_geoshape().encode(
        color=alt.Color(
            "R0:Q",
            sort="ascending",
            scale=alt.Scale(
                scheme="bluepurple",
                domainMin=0,
            ),
            legend=alt.Legend(
                title="R0",
                orient="bottom",
                tickCount=10,
            ),
        ),
        tooltip=["name_muni", "R0:Q"],
    ).properties(
        title={
            "text": f"R0 by city in {year}",
            "fontSize": 15,
            "anchor": "start",
        },
        width=500,
        height=400,
    )


async def model_evaluation_chart(
    rate_map: gpd.GeoDataFrame,
    year: int
) -> alt.Chart:
    bins = [1 / 2, 95 / 100, 1.05, 2]
    color_list = ["#006aea", "#00b4ca", "#48d085", "#dc7080", "#cb2b2b"]

    legend_table = pd.DataFrame(
        [
            [0, 1, 0, bins[0], "a"],
            [1, 2, bins[0], bins[1], "b"],
            [2, 3, bins[1], bins[2], "c"],
            [3, 4, bins[2], bins[3], "d"],
            [4, 5, bins[3], bins[3] + 1, "e"],
        ],
        columns=["x1", "x2", "v1", "v2", "color"],
    )

    chart = alt.Chart(data=rate_map).mark_geoshape(
        fillOpacity=0.5, fill="grey", stroke="#000", strokeOpacity=0.5
    ) + alt.Chart(data=rate_map,).mark_geoshape(
        stroke="#000", strokeOpacity=0.5
    ).encode(
        color=alt.Color(
            "rate:Q",
            sort="ascending",
            scale=alt.Scale(type="threshold", domain=bins, range=color_list),
            legend=None,
        ),
        tooltip=["name_muni", "rate:Q"],
    ).properties(
        title={
            "text": f"Observed Cases/Estimated Cases by city in {year}",
            "fontSize": 15,
            "anchor": "start",
        },
        width=500,
        height=400,
    )

    legend = (
        alt.Chart(legend_table, height=70)
        .mark_rect()
        .encode(
            x=alt.X("x1", scale=alt.Scale(), axis=None),
            x2="x2",
            y=datum(0, scale=alt.Scale(), axis=None),
            y2=datum(1),
            color=alt.Color(
                "color", legend=None, scale=alt.Scale(range=color_list)
            ),
        )
        + alt.Chart(legend_table)
        .mark_text(size=10)
        .encode(x="x1", text="v1", y=datum(-0.8))
        + alt.Chart(legend_table)
        .mark_rule(opacity=0.6)
        .encode(x="x1", y=datum(1), y2=datum(-0.4))
        + alt.Chart()
        .mark_rule()
        .encode(x=datum(0), x2=datum(1.95), y=datum(1.1))
        + alt.Chart()
        .mark_rule()
        .encode(x=datum(2.05), x2=datum(2.95), y=datum(1.1))
        + alt.Chart()
        .mark_rule()
        .encode(x=datum(3.05), x2=datum(5), y=datum(1.1))
        + alt.Chart()
        .mark_text(size=10, text="Overestimated")
        .encode(x=datum(1), y=datum(1.5))
        + alt.Chart()
        .mark_text(size=10, text="Good")
        .encode(x=datum(2.5), y=datum(1.5))
        + alt.Chart()
        .mark_text(size=10, text="Underestimated")
        .encode(x=datum(4), y=datum(1.5))
        + alt.Chart()
        .mark_text(size=12, text="Model Evaluation")
        .encode(x=datum(2.5), y=datum(2.5))
        + alt.Chart()
        .mark_text(size=11, text="Observed Cases/Estimated Cases")
        .encode(x=datum(2.5), y=datum(-1.6))
        + alt.Chart()
        .mark_text(
            text="* Cities in gray did not have an",
            size=12,
            color="grey",
            align="left",
            baseline="bottom",
            dx=20,
        )
        .encode(x=datum(5), y=datum(2.5))
        + alt.Chart()
        .mark_text(
            text="epidemic detected.",
            size=12,
            color="grey",
            align="left",
            baseline="bottom",
            dx=30,
        )
        .encode(x=datum(5), y=datum(1.5))
    )

    return chart & legend


async def model_evaluation_hist_chart(rate_map: gpd.GeoDataFrame) -> alt.Chart:
    bins = [1 / 2, 95 / 100, 1.05, 2]
    color_list = ["#006aea", "#00b4ca", "#48d085", "#dc7080", "#cb2b2b"]
    return (
        alt.Chart(rate_map, width=250)
        .mark_bar()
        .encode(
            x=alt.X(
                "rate:Q",
                title="Observed Cases/Estimated Cases",
                bin=alt.Bin(step=0.05, extent=[0, 3]),
                axis=alt.Axis(values=bins),
            ),
            y=alt.Y(
                "count()",
                title="Count of cities",
                scale=alt.Scale(type="sqrt"),
            ),
            color=alt.Color(
                "rate:Q",
                sort="ascending",
                scale=alt.Scale(
                    type="threshold", domain=bins, range=color_list
                ),
                legend=None,
            ),
        )
    )


def top_cities(weeks_map: gpd.GeoDataFrame) -> pd.DataFrame:
    weeks_map["transmissao"] = weeks_map.transmissao.astype(int)
    return weeks_map.sort_values("transmissao", ascending=False)[
        ["name_muni", "transmissao", "code_muni"]
    ]


def top_n_cities_md(df: pd.DataFrame) -> str:
    return markdown_table(
        fields=["Names", "Epi Weeks"],
        rows=df[["name_muni", "transmissao"]].values.tolist(),
    )


@lru_cache(maxsize=None)
def top_n_R0_md(q: Q, year: int, n: int) -> str:
    pars = q.client.parameters
    df = (
        pars[pars.year == year]
        .sort_values("R0", ascending=False)[["geocode", "R0"]]
        .head(n)
    )
    df["name"] = [q.client.cities[gc] for gc in df.geocode]
    return markdown_table(
        fields=["Names", "R0"],
        rows=df[["name", "R0"]].round(decimals=2).values.tolist(),
    )


def table_model_evaluation_md(q: Q, year: int) -> str:
    bins = [0, 0.5, 0.95, 1.05, 2, np.inf]
    df = q.client.data_table[
        q.client.data_table["SE"].isin(
            range((year - 1) * 100 + 45, year * 100 + 45)
        )
    ]
    cases = (
        df[["municipio_nome", "municipio_geocodigo", "casos"]]
        .groupby(["municipio_nome", "municipio_geocodigo"])
        .sum()
    )
    cases = cases.rename(columns={"casos": "observed_cases"}).reset_index()

    map_rate = cases.merge(
        q.client.parameters[q.client.parameters.year.astype(int) == year],
        left_on="municipio_geocodigo",
        right_on="geocode",
        how="outer",
    )[["municipio_nome", "observed_cases", "total_cases"]]

    map_rate["rate"] = map_rate["observed_cases"] / map_rate["total_cases"]

    groupby_rate = (
        map_rate[["municipio_nome"]]
        .groupby(pd.cut(map_rate["rate"], bins=bins))
        .count()
        .reset_index()
    )
    groupby_rate["perc"] = np.round(
        groupby_rate["municipio_nome"]
        / groupby_rate.municipio_nome.sum()
        * 100,
        2,
    )

    groupby_rate["text"] = (
        groupby_rate.municipio_nome.astype(str)
        + " ("
        + groupby_rate.perc.astype(str)
        + "%)"
    )

    return markdown_table(
        fields=["Range", "Range Counts(%)"],
        rows=groupby_rate[["rate", "text"]].values.tolist(),
    )


def markdown_table(fields: list[str], rows: list) -> str:
    """
    Create markdown table
    Args:
        fields: list of column names
        rows: List of rows in Markdown format

    Returns: Markdown table
    """

    def row(values):
        return f"| {' | '.join([str(x) for x in values])} |"

    return "\n".join(
        [
            row(fields),
            row("-" * len(fields)),
            "\n".join([row(r) for r in rows]),
        ]
    )


@np.vectorize
def richards(L, a, b, t, tj):
    """
    Richards model
    """
    j = L - L * (1 + a * np.exp(b * (t - tj))) ** (-1 / a)
    return j


async def plot_series_altair(q: Q, gc: int, start_date: str, end_date: str):

    if (
        "epi_year" in q.client
        and (q.client.epi_year is not None)
        and (q.client.epi_year != "all")
    ):

        title = (
            f"{q.client.disease.capitalize()} weekly cases "
            f"in {q.client.epi_year} for {q.client.cities[int(q.client.city)]}"
        )

    else:
        title = (
            f"{q.client.disease.capitalize()} weekly cases "
            f"in {q.client.cities[int(q.client.city)]}"
        )

    df = q.client.data_table
    dfcity = df[df.municipio_geocodigo == gc].loc[start_date:end_date]
    dfcity.sort_index(inplace=True)
    dfcity["casos_cum"] = dfcity.casos.cumsum()
    sirp = np.array([])
    if (
        "epi_year" in q.client
        and (q.client.epi_year is not None)
        and (q.client.epi_year != "all")
    ):
        sirp = q.client.parameters[
            (q.client.parameters.geocode == gc)
            & (q.client.parameters.year == int(q.client.epi_year))
        ][["total_cases", "beta", "gamma", "peak_week"]].values
        if sirp.any():
            a = 1 - (sirp[0, 2] / sirp[0, 1])
            L, b, tj = sirp[0, 0], sirp[0, 1] - sirp[0, 2], sirp[0, 3]
            dfcity["Model fit"] = richards(
                L, a, b, np.arange(len(dfcity.index)), tj
            )

        dfcity["model_legend"] = "Model"

        vertical_line = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "x": [dfcity.index[int(round(tj, 0))]],
                        "label": ["Peak week"],
                    }
                )
            )
            .mark_rule(size=2)
            .encode(
                x="x:T",
                color=alt.Color(
                    "label:N",
                    scale=alt.Scale(
                        domain=["Peak week"], range=["orange"]
                    ),  # Define specific color for 'Threshold'
                    legend=alt.Legend(title=" ", orient="left", offset=-130),
                ),
            )
        )

    ch1 = (
        alt.Chart(
            dfcity.reset_index(),
            width=750,
            height=200,
            title={"text": f"{title}", "fontSize": 15, "anchor": "start"},
        )
        .mark_area(
            opacity=0.3,
            interpolate="step-after",
        )
        .encode(
            x=alt.X("data_iniSE:T", axis=alt.Axis(title="")),
            y=alt.Y(
                "casos:Q", axis=alt.Axis(title="New Cases", titleFontSize=12)
            ),
            tooltip=["data_iniSE:T", "casos:Q"],
        )
    )

    ch2 = (
        alt.Chart(
            dfcity.reset_index(),
            width=750,
            height=200,
        )
        .mark_area(
            opacity=0.3,
            interpolate="step-after",
        )
        .encode(
            x=alt.X(
                "data_iniSE:T", axis=alt.Axis(title="Date", titleFontSize=12)
            ),
            y=alt.Y(
                "casos_cum:Q",
                axis=alt.Axis(title="Cumulative Cases", titleFontSize=12),
            ),
            tooltip=["data_iniSE:T", "casos_cum:Q", "Model fit:Q"],
        )
    )
    if "epi_year" in q.client and sirp.any():
        model = (
            alt.Chart(dfcity.reset_index())
            .mark_line()
            .encode(
                x=alt.X("data_iniSE:T", axis=alt.Axis(title="Date")),
                y=alt.Y("Model fit:Q", axis=alt.Axis(title="")),
                color=alt.Color(
                    "model_legend:N",
                    scale=alt.Scale(domain=["Model"], range=["red"]),
                    legend=alt.Legend(
                        title="",
                        orient="left",  # Position legend on the left
                        direction="vertical",  # Arrange items vertically
                        titleAnchor="middle",  # Center-align the title within the legend
                        offset=-130,
                    ),
                ),
            )
        )
        spec = alt.vconcat(
            ch1 + vertical_line,
            (ch2 + vertical_line + model).resolve_scale(color="independent"),
        ).resolve_scale(
            color="independent"
        )  # Keep color scales independent for separate legends)  # leaving this off for now
    else:
        spec = alt.vconcat(ch1, ch2)
    return spec


async def plot_epidemic_calc_altair(
    q: Q, gc: int, pw: int, R0: float, total_cases: int
):

    SCALE = alt.Scale(
        domain=["Data", "Model", "Peak week"],  # Adjust categories
        range=["#1f77b4", "#ff7f0e", "red"],
    )

    eyear = datetime.date.today().year

    start_date, end_date = get_ini_end_week(year=eyear)

    title = (
        f"{q.client.disease.capitalize()} weekly cases "
        f"in {eyear} for {q.client.cities[int(q.client.city)]}"
    )

    df = q.client.data_table
    dfcity = df[df.municipio_geocodigo == gc].loc[start_date:end_date]
    dfcity.sort_index(inplace=True)
    dfcity["casos_cum"] = dfcity.casos.cumsum()
    dfcity = dfcity.reset_index().loc[:, ["data_iniSE", "casos_cum"]]

    R = 1 - 1 / R0
    gamma = 0.3
    b = R * gamma / (1 - R)
    a = b / (gamma + b)

    dfcity2 = pd.DataFrame()
    dfcity2["data_iniSE"] = pd.date_range(
        start=dfcity.data_iniSE.values[0], periods=52, freq="W-SUN"
    )
    dfcity2["model"] = richards(total_cases, a, b, np.arange(52), pw)

    dfcity_end = dfcity.merge(
        dfcity2, left_on="data_iniSE", right_on="data_iniSE", how="outer"
    )

    df1 = dfcity_end.copy()
    df1["legend"] = "Data"

    df2 = dfcity_end.copy()
    df2["legend"] = "Model"

    # Create the first chart (Area for Cumulative Cases)
    ch1 = (
        alt.Chart(df1, width=650, height=350)
        .mark_area(
            opacity=0.3,
            interpolate="step-after",
            color="#1f77b4",
        )
        .encode(
            x=alt.X(
                "data_iniSE:T", axis=alt.Axis(title="Date", titleFontSize=12)
            ),
            y=alt.Y(
                "casos_cum:Q",
                axis=alt.Axis(title="Cumulative Cases", titleFontSize=12),
            ),
            color=alt.Color("legend:N", title=" ", scale=SCALE),
            tooltip=["data_iniSE:T", "casos_cum:Q", "model:Q"],
        )
    )

    # Create the second chart (Line for Model Prediction)
    ch2 = (
        alt.Chart(df2, width=650, height=350)
        .mark_line(color="red")
        .encode(
            x=alt.X(
                "data_iniSE:T", axis=alt.Axis(title="Date", titleFontSize=12)
            ),
            y=alt.Y(
                "model:Q",
                axis=alt.Axis(title="Cumulative Cases", titleFontSize=12),
            ),
            color=alt.Color("legend:N", title=" ", scale=SCALE),
            tooltip=["data_iniSE:T", "casos_cum:Q", "model:Q"],
        )
    )

    ch2_points = (
        alt.Chart(df2, width=650, height=350)
        .mark_point(size=60, filled=True, color="red")
        .encode(
            x=alt.X(
                "data_iniSE:T", axis=alt.Axis(title="Date", titleFontSize=12)
            ),
            y=alt.Y(
                "model:Q",
                axis=alt.Axis(title="Cumulative Cases", titleFontSize=12),
            ),
            color=alt.Color(
                "legend:N", title=" ", scale=SCALE
            ),  # Assign color based on legend,
            tooltip=["data_iniSE:T", "casos_cum:Q", "model:Q"],
        )
        .properties(title=title)
    )

    vertical_line = (
        alt.Chart(
            pd.DataFrame(
                {
                    "data_iniSE": [df2.data_iniSE[int(round(pw, 0))]],
                    "label": ["Peak week"],
                }
            )
        )
        .mark_rule(size=2, color="orange")
        .encode(
            x=alt.X(
                "data_iniSE:T", axis=alt.Axis(title="Date", titleFontSize=12)
            ),
            color=alt.Color(
                "label:N",
                scale=SCALE,
                legend=alt.Legend(title=" ", orient="left", offset=-130),
            ),
        )
    )

    spec = ch1 + ch2 + ch2_points + vertical_line

    return spec


async def plot_series_px(q: Q, gc: int, start_date: str, end_date: str):
    """
    Plot timeseries between two dates of city with Plotly
    Args:
        q:
        gc:
        start_date:
        end_date:
    """
    df = q.client.data_table
    dfcity = df[df.municipio_geocodigo == gc].loc[start_date:end_date]
    dfcity.sort_index(inplace=True)
    dfcity["casos_cum"] = dfcity.casos.cumsum()
    spl = make_subplots(rows=2, cols=1)
    spl.add_trace(go.Bar(x=dfcity.index, y=dfcity.casos), row=1, col=1)
    spl.add_trace(go.Bar(x=dfcity.index, y=dfcity.casos_cum), row=2, col=1)
    html = pio.to_html(spl, validate=False, include_plotlyjs=True)
    q.page["ts_plot_px"].content = html
    await q.page.save()
