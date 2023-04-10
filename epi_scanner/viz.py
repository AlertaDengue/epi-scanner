import io
import os
import uuid
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from epi_scanner.settings import CTNR_EPISCANNER_DATA_DIR
from h2o_wave import Q
from plotly.subplots import make_subplots


async def load_map(q: Q):
    file_gpkg = Path(f"{CTNR_EPISCANNER_DATA_DIR}/muni_br.gpkg")

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


async def plot_state_map(
    q, themap: gpd.GeoDataFrame, uf: str = "SC", column=None
):
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


def get_year_map(year: int, themap: gpd.GeoDataFrame, pars: pd.DataFrame):
    map_pars = themap.merge(
        pars[pars.year == year], left_on="code_muni", right_on="geocode"
    )
    return map_pars


async def plot_pars_map(
    q, themap: gpd.GeoDataFrame, year: int, state: str, column="R0"
):
    map_pars = get_year_map(year, q.client.weeks_map, q.client.parameters)
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


async def plot_series_px(q: Q, gc: int, start_date: str, end_date: str):
    df = q.client.data_table
    dfcity = df[df.municipio_geocodigo == gc].loc[start_date:end_date]
    dfcity.sort_index(inplace=True)
    dfcity["casos_cum"] = dfcity.casos.cumsum()
    spl = make_subplots(rows=2, cols=1)
    fig = px.bar(dfcity.casos)
    fig2 = px.bar(dfcity.casos_cum)
    spl.add_trace(fig.data[0], row=1, col=1)
    spl.add_trace(fig2.data[0], row=2, col=1)
    buffer = io.StringIO()
    spl.write_html(
        buffer, include_plotlyjs="cdn", validate=False, full_html=False
    )
    html = buffer.getvalue()
    # html = pio.to_html(spl, validate=False, include_plotlyjs='cdn')
    # print(html)
    q.page["ts_plot_px"].content = html
    await q.page.save()
