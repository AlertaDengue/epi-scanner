import time
import os
from typing import List

import pandas as pd
import psutil
from h2o_wave import ui, data, Q, app, main, copy_expando
from loguru import logger
from viz import load_map, plot_state_map

DATA_TABLE = None
BRMAP = None


async def initialize_app(q: Q):
    """
    Set up UI elements
    """
    global BRMAP
    q.page['title'] = ui.header_card(
        box=ui.box('header'),
        title='Real-time Epidemic Scanner',
        subtitle='Real-time epidemiology',
        # image='/assets/info-dengue-logo.png',
        icon='health'
    )
    await q.page.save()
    BRMAP = load_map()
    add_stats_cards(q)
    await q.page.save()
    UF = q.client.uf
    if len(BRMAP):
        fig = await plot_state_map(q, BRMAP, UF)
    q.page['plot'] = ui.frame_card(box='content', title=f'Map of {UF}', content=f'![plot]({fig})')
    add_sidebar(q)
    q.page['footer'] = ui.footer_card(box='footer', caption='(c) 2022 Infodengue. All rights reserved.')


@app('/', mode='multicast')
async def serve(q: Q):
    copy_expando(q.args, q.client)
    create_layout(q)
    await initialize_app(q)
    await q.page.save()
    q.client.cities = {}
    q.client.loaded = ''
    q.client.uf = 'SC'
    while True:
        if q.args.state:
            await update_UF(q)
        if q.args.city:
            await update_city(q)
        # logger.info(f'UF: {UF}, state: {q.args.state}, city:{q.client.city}')
        await load_table(q)
        # await update_stats(q)
        await q.page.save()


async def update_UF(q: Q):
    logger.info(f'UF: {UF}, state: {q.args.state}, city:{q.client.city}')
    q.client.uf = q.args.state


async def update_city(q: Q):
    logger.info(f'UF: {q.client.uf}, state: {q.args.state}, city:{q.client.city}')
    q.client.city = q.args.city


def create_layout(q):
    q.page['meta'] = ui.meta_card(box='', layouts=[
        ui.layout(
            breakpoint='xs',
            width='800px',
            zones=[
                ui.zone('header'),
                ui.zone('body', direction=ui.ZoneDirection.ROW,
                        zones=[
                            ui.zone('sidebar',
                                    size='25%',
                                    # direction=ui.ZoneDirection.COLUMN,
                                    # align='start',
                                    # justify='around'
                                    ),
                            ui.zone('content',
                                    size='75%',
                                    # direction=ui.ZoneDirection.ROW,
                                    # align='end',
                                    # justify='around',
                                    ),
                        ]),
                ui.zone('footer'),
            ]
        )
    ])


def df_to_table_rows(df: pd.DataFrame) -> List[ui.TableRow]:
    return [ui.table_row(name=str(r[0]), cells=[str(r[0]), r[1]]) for r in df.itertuples(index=False)]


async def load_table(q: Q):
    global DATA_TABLE
    UF = q.client.uf
    if DATA_TABLE is None and os.path.exists(f"{UF}_dengue.parquet"):
        logger.info("loading data...")
        DATA_TABLE = pd.read_parquet(f"{UF}_dengue.parquet")
        q.client.loaded = True
        for gc in DATA_TABLE.municipio_geocodigo.unique():
            q.client.cities[gc] = BRMAP[BRMAP.code_muni.astype(int) == int(gc)].name_muni.values[0]
        choices = [ui.choice(str(gc), q.client.cities[gc]) for gc in DATA_TABLE.municipio_geocodigo.unique()]
        # q.page['form'].items[1].dropdown.enabled = True
        q.page['form'].items[1].dropdown.choices = choices
        q.page['form'].items[1].dropdown.visible = True
        q.page['form'].items[1].dropdown.value = int(gc)

    await q.page.save()


async def update_analysis(q):
    pass


async def update_stats(q: Q):
    cpu_usage = psutil.cpu_percent(interval=1)
    q.page['cpu'].data.usage = cpu_usage
    q.page['cpu'].plot_data[-1] = [time.ctime(), cpu_usage]

    mem_usage = psutil.virtual_memory().percent
    q.page['memory'].data.usage = mem_usage
    q.page['memory'].plot_data[-1] = [time.ctime(), mem_usage]

    await q.page.save()


def add_sidebar(q):
    state_choices = [
        ui.choice('SC', 'Santa Catarina'),
        ui.choice('RS', 'Rio Grande do Sul')
    ]
    q.page['form'] = ui.form_card(box='sidebar', items=[
        ui.dropdown(name='state', label='Select state', value='RS', required=True,
                    choices=state_choices, trigger=True),
        ui.dropdown(name='city', label='Select city', required=True,
                    choices=[], trigger=True, visible=False)
    ])


def add_stats_cards(q):
    q.page['cpu'] = ui.small_series_stat_card(
        box='content',
        title='CPU',
        value='={{usage}}%',
        data=dict(usage=0.0),
        plot_data=data('tick usage', -15),
        plot_category='tick',
        plot_value='usage',
        plot_zero_value=0,
        plot_color='$red',
    )

    q.page['memory'] = ui.small_series_stat_card(
        box='content',
        title='Memory',
        value='={{usage}}%',
        data=dict(usage=0.0),
        plot_data=data('tick usage', -15),
        plot_category='tick',
        plot_value='usage',
        plot_zero_value=0,
        plot_color='$blue',
    )
