import pandas as pd
from h2o_wave import Q
import geopandas as gpd
import uuid
import os
import matplotlib.pyplot as plt


async def load_map(q: Q):
    brmap = gpd.read_file('data/muni_br.gpkg', driver='GPKG')
    q.client.brmap = brmap


async def update_state_map(q: Q):
    statemap = q.client.brmap[q.client.brmap.abbrev_state == q.client.uf]
    q.client.statemap = statemap


async def t_weeks(q: Q):
    """
    Merge weeks table with map
    """
    weeks = q.client.data_table.groupby(by='municipio_geocodigo').sum(numeric_only=True)[['transmissao']]
    wmap = q.client.statemap.merge(weeks, left_on='code_muni', right_index=True)
    q.client.weeks_map = wmap
    q.client.weeks = True
    await q.page.save()


async def plot_state_map(q, themap: gpd.GeoDataFrame, uf: str = 'SC', column=None):
    ax = themap.plot(column=column, legend=True,
                     scheme='natural_breaks',
                     legend_kwds={'loc': 'lower left'})
    ax.set_axis_off()
    image_filename = f'{str(uuid.uuid4())}.png'
    plt.savefig(image_filename)
    # Upload
    image_path, = await q.site.upload([image_filename])
    # Clean up
    os.remove(image_filename)
    return image_path


async def top_n_cities(q: Q, n: int):
    wmap = q.client.weeks_map
    df = wmap.sort_values('transmissao', ascending=False)[['name_muni', 'transmissao']].head(n)
    return make_markdown_table(fields=['Names', 'Weeks'],
                               rows=df.values.tolist()
                               )


def make_markdown_row(values):
    return f"| {' | '.join([str(x) for x in values])} |"


def make_markdown_table(fields, rows):
    """
    Create markdown table
    Args:
        fields: list of column names
        rows: List of rows i markdown format

    Returns: Markdown table
    """
    return '\n'.join([
        make_markdown_row(fields),
        make_markdown_row('-' * len(fields)),
        '\n'.join([make_markdown_row(row) for row in rows]),
    ])


async def plot_series(q: Q, gc: int, start_date: str, end_date: str):
    df = q.client.data_table
    dfcity = df[df.municipio_geocodigo == gc].loc[start_date:end_date]
    dfcity.sort_index(inplace=True)
    dfcity['casos_cum'] = dfcity.casos.cumsum()
    fig, [ax1, ax2] = plt.subplots(2, 1)
    dfcity.casos.plot.area(ax=ax1, label='Cases', grid=True, alpha=0.4)
    ax1.legend()
    dfcity.casos_cum.plot.area(ax=ax2, label='Cumulative cases', grid=True, alpha=0.4)
    ax2.legend()
    image_filename = f'{str(uuid.uuid4())}.png'
    plt.savefig(image_filename)
    # Upload
    image_path, = await q.site.upload([image_filename])
    # Clean up
    os.remove(image_filename)
    return image_path
