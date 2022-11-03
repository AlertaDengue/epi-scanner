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
    return make_markdown_table(fields=df.columns.tolist(),
                               rows=df.values.tolist()
                               )


def make_markdown_row(values):
    return f"| {' | '.join([str(x) for x in values])} |"


def make_markdown_table(fields, rows):
    return '\n'.join([
        make_markdown_row(fields),
        make_markdown_row('-' * len(fields)),
        '\n'.join([make_markdown_row(row) for row in rows]),
    ])
