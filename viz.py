import pandas as pd
import geopandas as gpd
import uuid
import os
import matplotlib.pyplot as plt


def load_map():
    brmap = gpd.read_file('muni_br.gpkg', driver='GPKG')
    return brmap


async def plot_state_map(q, brmap: gpd.GeoDataFrame, uf: str = 'SC', column=None):
    statemap = brmap[brmap.abbrev_state == uf]
    ax = statemap.plot(column=column, legend=True,
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
