import time
import os
from typing import List

import pandas as pd
import psutil
from h2o_wave import ui, data, Q, app, main, copy_expando
from loguru import logger
from viz import load_map, plot_state_map, t_weeks, update_state_map

DATA_TABLE = None


async def initialize_app(q: Q):
    """
    Set up UI elements
    """
    q.page['title'] = ui.header_card(
        box=ui.box('header'),
        title='Real-time Epidemic Scanner',
        subtitle='Real-time epidemiology',
        # image='assets/info-dengue-logo.png',
        icon='health'
    )
    await q.page.save()
    await load_map(q)
    # add_stats_cards(q)
    await q.page.save()
    UF = q.client.uf = 'SC'
    await update_state_map(q)
    fig = await plot_state_map(q, q.client.statemap, UF)
    q.page['plot'] = ui.markdown_card(box='content', title=f'Map of {UF}', content=f'![plot]({fig})')
    add_sidebar(q)
    q.page['footer'] = ui.footer_card(box='footer', caption='(c) 2022 Infodengue. All rights reserved.')


@app('/scanner', mode='multicast')
async def serve(q: Q):
    copy_expando(q.args, q.client)
    create_layout(q)
    await initialize_app(q)
    await q.page.save()
    q.client.cities = {}
    q.client.loaded = ''
    q.client.uf = 'SC'
    q.client.weeks = False
    while True:
        if q.args.state:
            await update_UF(q)
        if q.args.city:
            await update_city(q)
        # logger.info(f'UF: {q.args}, state: {q.args.state}, city:{q.client.city}')
        await load_table(q)
        if (not q.client.weeks) and (q.client.data_table is not None):
            await t_weeks(q)
            fig = await plot_state_map(q, q.client.weeks_map, q.client.uf, column='transmissao')
            q.page['plotweeks'] = ui.markdown_card(box='content',
                                                   title=f'Number of weeks of $R_t>1$ in the last 10 years',
                                                   content=f'![plot]({fig})')

        await q.page.save()


async def update_UF(q: Q):
    logger.info(f'UF: {q.client.uf}, state: {q.args.state}, city:{q.args.city}')
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
                                    direction=ui.ZoneDirection.ROW,
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
    if DATA_TABLE is None and os.path.exists(f"data/{UF}_dengue.parquet"):
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
    pass


def add_sidebar(q):
    state_choices = [
        ui.choice('SC', 'Santa Catarina'),
        ui.choice('RS', 'Rio Grande do Sul')
    ]
    q.page['form'] = ui.form_card(box='sidebar', items=[
        ui.dropdown(name='state', label='Select state', value='RS', required=True,
                    choices=state_choices, trigger=False),
        ui.dropdown(name='city', label='Select city', required=True,
                    choices=[], trigger=False, visible=False)
    ])
