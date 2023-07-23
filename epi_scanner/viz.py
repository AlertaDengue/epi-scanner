import os
import uuid
from pathlib import Path

import altair as alt
from altair import datum
import geopandas as gpd
import gpdvega  # NOQA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from epi_scanner.settings import EPISCANNER_DATA_DIR
from h2o_wave import Q
from plotly import io as pio
from plotly.subplots import make_subplots


async def load_map(q: Q):
    file_gpkg = Path(f"{EPISCANNER_DATA_DIR}/muni_br.gpkg")

    brmap = gpd.read_file(file_gpkg, driver="GPKG")
    q.client.brmap = brmap


async def update_state_map(q: Q):
    statemap = q.client.brmap[q.client.brmap.abbrev_state == q.client.uf]
    q.client.statemap = statemap


async def t_weeks(q: Q):
    """
    Merge weeks table with map
    """
    weeks = q.client.data_table.groupby(by="municipio_geocodigo").sum(
        numeric_only=True
    )[["transmissao"]]
    wmap = q.client.statemap.merge(
        weeks, left_on="code_muni", right_index=True
    )
    q.client.weeks_map = wmap
    q.client.weeks = True
    await q.page.save()


async def plot_state_map(q, themap: gpd.GeoDataFrame, column=None):
    ax = themap.plot(
        column=column,
        legend=True,
        scheme="natural_breaks",
        legend_kwds={
            "loc": "lower center",
            "ncols": 5,
            "fontsize": "x-small",
        },  # {'bbox_to_anchor': (1.15, 1)}
    )
    ax.set_axis_off()
    image_path = await get_mpl_img(q)
    return image_path


async def plot_state_map_altair(q: Q, themap: gpd.GeoDataFrame, column=None):
    spec = (
        alt.Chart(themap)
        .mark_geoshape()
        .encode(
            color=alt.Color(
                f"{column}:Q",
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
            tooltip=["name_muni", column + ":Q"],
        )
        .properties(width=500, height=400)
    )
    return spec


async def get_mpl_img(q):
    """
    saves current matplotlib figures to a temporary file
     and uploads it to the site.
    Args:
        q: App

    Returns: path in the site.
    """
    image_filename = f"{str(uuid.uuid4())}.png"
    plt.savefig(image_filename)
    # Upload
    (image_path,) = await q.site.upload([image_filename])
    # Clean up
    os.remove(image_filename)
    return image_path


def get_year_map(years: list, themap: gpd.GeoDataFrame, pars: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Merge map with parameters for a given year
    Args:
        years: list of one or more years to be selected
        themap: map to be merged
        pars: parameters table

    Returns:

    """
    map_pars = themap.merge(
        pars[pars.year.astype(int).isin(years)], left_on="code_muni", right_on="geocode",
        how="outer"
    )
    return map_pars.fillna(0)


async def plot_pars_map(
    q, themap: gpd.GeoDataFrame, year: int, state: str, column="R0"
):
    map_pars = get_year_map([year], q.client.weeks_map, q.client.parameters)
    ax = themap.plot(alpha=0.3)
    if len(map_pars) == 0:
        pass
    else:
        map_pars.plot(
            ax=ax,
            column=column,
            legend=True,
            scheme="User_Defined",
            classification_kwds=dict(bins=[1.5, 2, 2.3, 2.7]),
            legend_kwds={"loc": "lower center", "ncols": 5},
        )
    ax.set_title(f"{state} {year}")
    ax.set_axis_off()
    image_path = await get_mpl_img(q)
    return image_path

async def plot_pars_map_altair(q, themap: gpd.GeoDataFrame, years: list, state: str, column="R0"):
    map_pars = get_year_map(years, themap, q.client.parameters) # q.client.weeks_map
    # slider = alt.binding_range(min=2010, max=2022, step=1)
    # select_year = alt.selection_point(name='year', fields=['year'],
    #                                   bind=slider, value={'year': 2010})
    spec = (
        alt.Chart(
            data=map_pars[["geometry","year", "name_muni", "R0"]],
            padding={"left": 0, "top": 0, "right": 0, "bottom": 0},
        )
        .mark_geoshape()
        .encode(
            color=alt.Color(
                f"{column}:Q",
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
            tooltip=["name_muni", column + ":Q"],
        )#.add_params(select_year)#.transform_filter(select_year)
        .properties(width=500, height=400)

    )
    return spec

async def top_n_cities(q: Q, n: int):
    wmap = q.client.weeks_map
    wmap["transmissao"] = wmap.transmissao.astype(int)
    df = wmap.sort_values("transmissao", ascending=False)[
        ["name_muni", "transmissao"]
    ].head(n)
    return make_markdown_table(
        fields=["Names", "Epi Weeks"], rows=df.values.tolist()
    )


async def top_n_R0(q: Q, year: int, n: int):
    pars = q.client.parameters
    table = (
        pars[pars.year == year]
        .sort_values("R0", ascending=False)[["geocode", "R0"]]
        .head(n)
    )
    table["name"] = [q.client.cities[gc] for gc in table.geocode]
    return make_markdown_table(
        fields=["Names", "R0"],
        rows=table[["name", "R0"]].round(decimals=2).values.tolist(),
    )


def make_markdown_row(values):
    return f"| {' | '.join([str(x) for x in values])} |"


def make_markdown_table(fields, rows):
    """
    Create markdown table
    Args:
        fields: list of column names
        rows: List of rows in Markdown format

    Returns: Markdown table
    """
    return "\n".join(
        [
            make_markdown_row(fields),
            make_markdown_row("-" * len(fields)),
            "\n".join([make_markdown_row(row) for row in rows]),
        ]
    )


async def plot_series(q: Q, gc: int, start_date: str, end_date: str):
    """
    Plot timeseries between two dates of city
    Args:
        q:
        gc: geocode of the city
        start_date:
        end_date:

    Returns:
    image path
    """
    df = q.client.data_table
    dfcity = df[df.municipio_geocodigo == gc].loc[start_date:end_date]
    dfcity.sort_index(inplace=True)
    dfcity["casos_cum"] = dfcity.casos.cumsum()
    fig, [ax1, ax2] = plt.subplots(2, 1)
    dfcity.casos.plot.area(ax=ax1, label="Cases", grid=True, alpha=0.4)
    ax1.legend()
    dfcity.casos_cum.plot.area(
        ax=ax2, label="Cumulative cases", grid=True, alpha=0.4
    )
    ax2.legend()
    image_path = await get_mpl_img(q)
    return image_path

@np.vectorize
def richards(L,a,b,t,tj):
    """
    Richards model
    """
    j=L-L*(1+a*np.exp(b*(t-tj)))**(-1/a)
    return j

async def plot_series_altair(q: Q, gc: int, start_date: str, end_date: str):
    df = q.client.data_table
    dfcity = df[df.municipio_geocodigo == gc].loc[start_date:end_date]
    dfcity.sort_index(inplace=True)
    dfcity["casos_cum"] = dfcity.casos.cumsum()
    # if q.client.r0year is not None:
    #     sirp = q.client.parameters[(q.client.parameters.geocode == gc) & (q.client.parameters.year == q.client.r0year)][["total_cases", "beta", "gamma","peak_week"]].values
    #     a = (sirp[0,1]+sirp[0,2])/sirp[0,1]
    #     L,b,tj = sirp[0,0], sirp[0,1]*a, sirp[0,3]
    #     dfcity["richards"] = richards(L,a,b,range(len(dfcity.index)),tj)
    ch1 = (
        alt.Chart(
            dfcity.reset_index(),
            width=750,
            height=200,
                  ).mark_area(
            opacity=0.3,
            interpolate="step-after",
        ).encode(
            x=alt.X("data_iniSE:T", axis=alt.Axis(title="Date")),
            y=alt.Y("casos:Q", axis=alt.Axis(title="Cases")),
            tooltip=["data_iniSE:T", "casos:Q"]
        )
    )

    ch2 = (
        alt.Chart(
            dfcity.reset_index(),
            width=750,
            height=200,
                    ).mark_area(
            opacity=0.3,
            interpolate="step-after",
        ).encode(
            x=alt.X("data_iniSE:T", axis=alt.Axis(title="Date")),
            y=alt.Y("casos_cum:Q", axis=alt.Axis(title="Cumulative Cases")),
            tooltip=["data_iniSE:T", "casos_cum:Q"]
    )
    )
    if q.client.r0year is not None:
        # model = (
        #     alt.Chart(dfcity.reset_index()).mark_line(color="red").encode(
        #     x=alt.X("data_iniSE:T", axis=alt.Axis(title="Date")),
        #     y=alt.Y("richards:Q", axis=alt.Axis(title="Richards model"))
        # )
        # )
        spec = alt.vconcat(ch1, ch2)# +model) leaving this off for now
    else:
        spec = alt.vconcat(ch1, ch2)
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
    # fig = px.bar(dfcity.casos)
    # fig2 = px.bar(dfcity.casos_cum)
    # spl.add_trace(fig.data[0], row=1, col=1)
    # spl.add_trace(fig2.data[0], row=2, col=1)
    # buffer = io.StringIO()
    # spl.write_html(
    #     buffer, include_plotlyjs="cdn", validate=False, full_html=False
    # )
    # html = buffer.getvalue()
    html = pio.to_html(spl, validate=False, include_plotlyjs="cdn")
    # print(html)
    q.page["ts_plot_px"].content = html
    await q.page.save()
