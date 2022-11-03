import time
import os
from typing import List

import pandas as pd
import psutil
from h2o_wave import ui, data, Q, app, main, copy_expando
from loguru import logger
from viz import (load_map, plot_state_map, t_weeks,
                 update_state_map, top_n_cities, plot_series)

DATA_TABLE = None
STATES = {'SC': 'Santa Catarina',
          'PR': 'Paraná',
          'RS': 'Rio Grande do Sul'
          }


async def initialize_app(q: Q):
    """
    Set up UI elements
    """
    create_layout(q)
    q.page['title'] = ui.header_card(
        box=ui.box('header'),
        title='Real-time Epidemic Scanner',
        subtitle='Real-time epidemiology',
        # image='assets/info-dengue-logo.png',
        icon='health'
    )
    await q.page.save()
    # Setup some client side variables
    q.client.cities = {}
    q.client.loaded = False
    q.client.uf = 'SC'
    q.client.weeks = False
    await load_map(q)

    await q.page.save()
    UF = q.client.uf = 'SC'
    await update_state_map(q)
    fig = await plot_state_map(q, q.client.statemap, UF)
    q.page['state_header'] = ui.markdown_card(box='pre', title='Epi Report', content=f'## {STATES[q.client.uf]}')
    q.page['plot'] = ui.markdown_card(box='content', title=f'Map of {UF}', content=f'![plot]({fig})')
    add_sidebar(q)
    q.page['analysis_header'] = ui.markdown_card(box='analysis', title='City-level Analysis', content='')
    q.page['footer'] = ui.footer_card(box='footer', caption='(c) 2022 Infodengue. All rights reserved.')


@app('/', mode='multicast')
async def serve(q: Q):
    copy_expando(q.args, q.client)
    if not q.client.initialized:
        await initialize_app(q)
        q.client.initialized = True
    await q.page.save()
    # await update_weeks(q)
    # while True:
    if q.args.state:
        await on_update_UF(q)
        q.page['state_header'].content = f"## {STATES[q.client.uf]}"
        await q.page.save()
    if q.args.city:
        q.page['non-existent'].items = []
        await on_update_city(q)
        await q.page.save()


async def update_weeks(q):
    if (not q.client.weeks) and (q.client.data_table is not None):
        await t_weeks(q)
        logger.info('plot weeks')
        fig = await plot_state_map(q, q.client.weeks_map, q.client.uf, column='transmissao')
        await q.page.save()
        q.page['plot'] = ui.markdown_card(box='week_zone',
                                          title=f'Number of weeks of Rt > 1 over the last s10 years',
                                          content=f'![plot]({fig})')
        ttext = await top_n_cities(q, 10)
        q.page['wtable'] = ui.form_card(box='week_zone',
                                        items=[
                                            ui.text(ttext)
                                        ]
                                        )
        await q.page.save()


async def on_update_UF(q: Q):
    logger.info(f'client.uf: {q.client.uf}, args.state: {q.args.state}, args.city:{q.args.city}')
    uf = q.args.state
    if uf != q.client.uf:
        q.client.uf = q.args.state
        await load_table(q)
        await q.page.save()
        await update_state_map(q)
        q.client.weeks = False
        await update_weeks(q)

    await q.page.save()


async def on_update_city(q: Q):
    logger.info(
        f'client.uf: {q.client.uf}, args.state: {q.args.state}, client.city:{q.client.city}, args.city: {q.args.city}')
    if q.client.city != q.args.city:
        q.client.city = q.args.city
    # print(q.client.cities)
    q.page['analysis_header'].content = f"## {q.client.cities[int(q.client.city)]}"
    create_analysis_form(q)
    await update_analysis(q)
    await q.page.save()


def create_layout(q):
    q.page['meta'] = ui.meta_card(box='', theme='default', layouts=[
        ui.layout(
            breakpoint='xl',
            width='1200px',
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
                                    direction=ui.ZoneDirection.COLUMN,
                                    # align='end',
                                    # justify='around',
                                    zones=[
                                        ui.zone("pre"),
                                        ui.zone(name='week_zone', direction=ui.ZoneDirection.ROW),
                                        ui.zone("analysis")
                                    ]
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
    if os.path.exists(f"data/{UF}_dengue.parquet"):
        logger.info("loading data...")
        DATA_TABLE = pd.read_parquet(f"data/{UF}_dengue.parquet")
        q.client.data_table = DATA_TABLE
        q.client.loaded = True
        for gc in DATA_TABLE.municipio_geocodigo.unique():
            q.client.cities[gc] = q.client.brmap[q.client.brmap.code_muni.astype(int) == int(gc)].name_muni.values[0]
        choices = [ui.choice(str(gc), q.client.cities[gc]) for gc in DATA_TABLE.municipio_geocodigo.unique()]
        # q.page['form'].items[1].dropdown.enabled = True
        q.page['form'].items[1].dropdown.choices = choices
        q.page['form'].items[1].dropdown.visible = True
        q.page['form'].items[1].dropdown.value = int(gc)

    await q.page.save()


async def update_analysis(q):
    img = await plot_series(q, int(q.client.city), q.client.start_date, q.client.end_date)
    q.page['ts_plot'] = ui.markdown_card(box='analysis',
                                         title=f'Weekly Cases',
                                         content=f'![plot]({img})')
    q.page.save()


def add_sidebar(q):
    state_choices = [
        ui.choice('PR', 'Paraná'),
        ui.choice('SC', 'Santa Catarina'),
        ui.choice('RS', 'Rio Grande do Sul')
    ]
    q.page['form'] = ui.form_card(box='sidebar', items=[
        ui.dropdown(name='state', label='Select state', value='', required=True,
                    choices=state_choices, trigger=True),
        ui.dropdown(name='city', label='Select city', required=True,
                    choices=[], trigger=True, visible=False)
    ])


def create_analysis_form(q):
    q.page['dates'] = ui.form_card(box='analysis', items=[
        ui.date_picker(name='start_date', label='Start Date', value='2020-01-01'),
        ui.date_picker(name='end_date', label='End Date ', value='2022-11-3'),
    ])
