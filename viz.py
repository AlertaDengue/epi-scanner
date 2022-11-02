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
    weeks = q.client.data_table.groupby(by='municipio_geocodigo').sum(numeric_only=True)[['transmissao']]
    wmap = q.client.statemap.merge(weeks, left_on='code_muni', right_index=True)
    q.client.weeks_map = wmap


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
