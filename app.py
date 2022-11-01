import time

import psutil
from h2o_wave import ui, data, Q, app, main
from loguru import logger


@app('/', mode='unicast')
async def serve(q: Q):
    create_layout(q)
    q.page['title'] = ui.header_card(
        box=ui.box('header'),
        title='Real-time Epidemic Scanner',
        subtitle='Real-time epidemiology',
        # image='/assets/info-dengue-logo.png',
        icon='health'
    )
    add_stats_cards(q)
    add_sidebar(q)
    q.page['footer'] = ui.footer_card(box='footer', caption='(c) 2022 Infodengue. All rights reserved.')
    while True:
        try:
            await update_stats(q)
        except Exception as e:
            logger.debug(f"Exception while updating: {e}")


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
        ui.choice('SC', 'Santa Catarina')
    ]
    q.page['form'] = ui.form_card(box='sidebar', items = [
        ui.dropdown(name='State', label='Select state', value='SC', required=True,
                    choices=state_choices, trigger=True),
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
