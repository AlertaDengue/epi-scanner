import time
import psutil
from loguru import logger
from h2o_wave import site, ui, data, Q, main, app


@app('/monitor', mode='broadcast')
async def serve(q: Q):
    create_layout(q)
    q.page['title'] = ui.header_card(
        box=ui.box('header'),
        title='My Realtime Dashboard',
        subtitle='Real-time stats',
        # image='https://wave.h2o.ai/img/h2o-logo.svg',
        icon='health'
    )
    add_stats_cards(q)
    q.page['footer'] = ui.footer_card(box='footer', caption='(c) 2022 The GRAPH Network. All rights reserved.')
    while True:
        try:
            await update_stats(q)
        except Exception as e:
            logger.debug(f"Exception while updating: {e}")

def create_layout(q):
    q.page['meta'] = ui.meta_card(box='', layouts=[
        ui.layout(
            breakpoint='xs',
            width='500px',
            zones=[
                ui.zone('header'),
                ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[
                    ui.zone('content', 
                            size='75%', 
                            direction=ui.ZoneDirection.ROW,
                            align='center',
                            justify='around',
                            ),
                    ui.zone('sidebar', size='25%'),
                ]),
                ui.zone('footer'),
            ]
        )
    ])


async def update_stats(q: Q):
    cpu_usage = psutil.cpu_percent(interval=1)
    q.page['cpu'].data.usage = cpu_usage
    q.page['cpu'].plot_data[-1] = [time.ctime(), cpu_usage]

    mem_usage = psutil.virtual_memory().percent
    q.page['memory'].data.usage = mem_usage
    q.page['memory'].plot_data[-1] = [time.ctime(), mem_usage]

    await q.page.save()

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
