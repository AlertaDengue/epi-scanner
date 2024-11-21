import os
import uuid
from pathlib import Path

import altair as alt
from altair import datum
import geopandas as gpd
import gpdvega  # NOQA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from epi_scanner.settings import EPISCANNER_DATA_DIR
from h2o_wave import Q
from plotly import io as pio
from plotly.subplots import make_subplots


async def load_map(q: Q):
    file_gpkg = Path(f"{EPISCANNER_DATA_DIR}/muni_br.gpkg")

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


async def plot_state_map(q, themap: gpd.GeoDataFrame, column=None):
    ax = themap.plot(
        column=column,
        legend=True,
        scheme="natural_breaks",
        legend_kwds={
            "loc": "lower center",
            "ncols": 5,
            "fontsize": "x-small",
        },
    )
    ax.set_axis_off()
    image_path = await get_mpl_img(q)
    return image_path


async def plot_state_map_altair(q: Q, themap: gpd.GeoDataFrame, column=None, 
                                title = 'Number of weeks of Rt > 1 since 2010'):
    spec = (
        alt.Chart(themap)
        .mark_geoshape()
        .encode(
            color=alt.Color(
                f"{column}:Q",
                sort="ascending",
                scale=alt.Scale(
                    scheme="bluepurple"
                ),  # , domain = [-0.999125,41.548309]),
                legend=alt.Legend(
                    title="Weeks",
                    orient="bottom",
                    tickCount=10,
                ),
            ),
            tooltip=["name_muni", column + ":Q"],
        )
        .properties(title={
        "text": f"{title}",
        "fontSize": 15,
        "anchor": "start"}, width=500, height=400)
    )
    return spec


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


def get_year_map(
    years: list, themap: gpd.GeoDataFrame, pars: pd.DataFrame
) -> gpd.GeoDataFrame:
    """
    Merge map with parameters for a given year
    Args:
        years: list of one or more years to be selected
        themap: map to be merged
        pars: parameters table

    Returns:

    """
    map_pars = themap.merge(
        pars[pars.year.astype(int).isin(years)],
        left_on="code_muni",
        right_on="geocode",
        how="outer",
    )
    return map_pars.fillna(0)

def get_rate_map(
    years:list, statemap:gpd.GeoDataFrame, data:pd.DataFrame, pars:pd.DataFrame, 
) -> gpd.GeoDataFrame:
    """
    Merge map with rate between cases and estimated cases
    Args:
        years: list of one or more years to be selected (???)
        statemap: map to be merged
        data: data tables
        pars: parameters table

    Returns:

    """
    year = years[0]
    df = data[data["SE"].isin(range((year-1)*100+45,year*100+45))] 
    casos = df[['municipio_geocodigo','casos']].groupby('municipio_geocodigo').sum().rename(columns={'casos':'observed_cases'})

    map_rate = statemap.merge(casos,
               left_on="code_muni",
               right_index=True,
               how="outer"
        ).merge(pars[pars.year.astype(int).isin(years)],
                left_on="code_muni",
                right_on="geocode",
                how="outer",
        )[['name_muni','geometry','observed_cases','total_cases']]

    map_rate['rate'] = map_rate['observed_cases']/map_rate['total_cases']
    return map_rate


async def plot_pars_map(
    q, themap: gpd.GeoDataFrame, year: int, state: str, column="R0"
):
    map_pars = get_year_map([year], q.client.weeks_map, q.client.parameters)
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


async def plot_pars_map_altair(
    q, themap: gpd.GeoDataFrame, years: list, state: str, column="R0", title = 'R0 by city in'
):
    map_pars = get_year_map(years, themap, q.client.parameters)[
        ["geometry", "year", "name_muni", "R0"]
    ]
    spec = (
        alt.Chart(
            data=map_pars,
            padding={"left": 0, "top": 0, "right": 0, "bottom": 0},
        )
        .mark_geoshape()
        .encode(
            color=alt.Color(
                f"{column}:Q",
                sort="ascending",
                scale=alt.Scale(
                    scheme="bluepurple",
                    domainMin=0,
                ),
                legend=alt.Legend(
                    title="R0",
                    orient="bottom",
                    tickCount=10,
                ),
            ),
            tooltip=["name_muni", column + ":Q"],
        )
        .properties(title={
        "text": f"{title} {years[0]}",
        "fontSize": 15,
        "anchor": "start"}, width=500, height=400)
    )
    return spec

async def plot_model_evaluation_map_altair(
        q, statemap: gpd.GeoDataFrame, years: list, state: str, column="rate", title = "Observed Cases/Estimated Cases by city in", 
        bins = [1/2,95/100,1.05,2], color_list = ['#006aea', '#00b4ca', '#48d085', '#dc7080', '#cb2b2b']
):
    map_rate = get_rate_map(years, statemap, q.client.data_table, q.client.parameters)
    legend_table = pd.DataFrame(
        [
            [0,1,0,bins[0],'a'],
            [1,2,bins[0],bins[1],'b'],
            [2,3,bins[1],bins[2],'c'],
            [3,4,bins[2],bins[3],'d'],
            [4,5,bins[3],bins[3]+1,'e']
        ],
        columns=['x1','x2','v1','v2','color'])

    map = (
        alt.Chart(
            data=map_rate,
        ).mark_geoshape(
            fillOpacity = 0.5,
            fill = 'grey',
            stroke='#000', 
            strokeOpacity=0.5
        )+alt.Chart(
            data=map_rate,
        ).mark_geoshape(
            stroke='#000', 
            strokeOpacity=0.5
        ).encode(
            color=alt.Color(
                f"{column}:Q",
                sort="ascending",
                scale=alt.Scale(
                    type='threshold',
                    domain = bins,
                    range= color_list
                ),
                legend= None
            ),
            tooltip=["name_muni", column + ":Q"],
        ).properties(
            title={
                "text": f"{title} {years[0]}",
                "fontSize": 15,
                "anchor": "start"}, 
            width=500,
            height=400
        )
    )

    legend = (
        alt.Chart(legend_table,height=70).mark_rect().encode(
            x=alt.X('x1',scale=alt.Scale(),axis=None),x2='x2',
            y=alt.datum(0,scale=alt.Scale(),axis=None),y2=alt.datum(1),
            color=alt.Color('color',legend=None, scale=alt.Scale(range= color_list))
        )+alt.Chart(legend_table).mark_text(size=10).encode(
            x='x1',text='v1',y=alt.datum(-.8)
        )+alt.Chart(legend_table).mark_rule(opacity=0.6).encode(
            x='x1',y=alt.datum(1),y2=alt.datum(-0.4)
        )+alt.Chart().mark_rule().encode(x=alt.datum(0),x2=alt.datum(1.95),y=alt.datum(1.1)
        )+alt.Chart().mark_rule().encode(x=alt.datum(2.05),x2=alt.datum(2.95),y=alt.datum(1.1)
        )+alt.Chart().mark_rule().encode(x=alt.datum(3.05),x2=alt.datum(5),y=alt.datum(1.1)
        )+alt.Chart().mark_text(size=10).encode(x=alt.datum(1),y=alt.datum(1.5),text=alt.datum('Underestimated')
        )+alt.Chart().mark_text(size=10).encode(x=alt.datum(2.5),y=alt.datum(1.5),text=alt.datum('Good')
        )+alt.Chart().mark_text(size=10).encode(x=alt.datum(4),y=alt.datum(1.5),text=alt.datum('Overestimated')
        )+alt.Chart().mark_text(size=12,fontWeight='bold').encode(x=alt.datum(2.5),y=alt.datum(2.5),text=alt.datum('Model Evaluation')
        )+alt.Chart().mark_text(size=11,fontWeight='bold').encode(x=alt.datum(2.5),y=alt.datum(-1.6),text=alt.datum("Observed Cases/Estimated Cases"))
        )+alt.Chart().mark_text(
            text="* Cities in gray didn't\nhave an epidemic.",size=12,color='grey', align='left', baseline='bottom'
        ).encode(x=alt.datum(6),y=alt.datum(2.5))
    

    spec = map & legend
    return spec

async def plot_model_evaluation_hist_altair(
        q, statemap: gpd.GeoDataFrame, years: list, state: str, column="rate",
        bins = [1/2,95/100,1.05,2], color_list = ['#006aea', '#00b4ca', '#48d085', '#dc7080', '#cb2b2b']
):
    map_rate = get_rate_map(years, statemap, q.client.data_table, q.client.parameters)

    hist = (alt.Chart(map_rate, width = 250).mark_bar().encode(
        x=alt.X(
            'rate:Q',
            title ="Observed Cases/Estimated Cases",
            bin=alt.Bin(step=0.05,extent=[0,3]), 
            axis=alt.Axis(values=bins)),
        y=alt.Y(
            'count()',
            title='Count of cities', 
            scale=alt.Scale(type='sqrt')),
        color=alt.Color(
            f"{column}:Q",
            sort="ascending",
            scale=alt.Scale(
                type='threshold',
                domain = bins,
                range= color_list
            ),
        legend= None
        ))
    )
    return hist



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

async def table_model_evaluation(q:Q, year: int, bins = [0, 0.5, 0.95, 1.05, 2, np.inf]):
    df = q.client.data_table[q.client.data_table["SE"].isin(range((year-1)*100+45,year*100+45))] 
    cases = df[['municipio_nome','municipio_geocodigo','casos']].groupby(['municipio_nome','municipio_geocodigo']).sum()
    cases = cases.rename(columns={'casos':'observed_cases'}).reset_index()

    map_rate = cases.merge(
        q.client.parameters[q.client.parameters.year.astype(int).isin([year])],
        left_on="municipio_geocodigo",
        right_on="geocode",
        how="outer",
    )[['municipio_nome','observed_cases','total_cases']]

    map_rate['rate'] = map_rate['observed_cases']/map_rate['total_cases']

    groupby_rate = map_rate[['municipio_nome']].groupby(pd.cut(map_rate['rate'], bins=bins)).count().reset_index()
    groupby_rate['perc'] = np.round(groupby_rate['municipio_nome']/groupby_rate.municipio_nome.sum()*100,2)

    groupby_rate['text'] = groupby_rate.municipio_nome.astype(str) + "(" + groupby_rate.perc.astype(str)+'%)' 

    table = make_markdown_table(
        fields=["Range", "Range Counts(%)"],
        rows=groupby_rate[["rate", "text"]].values.tolist(),
    )

    return table

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


@np.vectorize
def richards(L, a, b, t, tj):
    """
    Richards model
    """
    j = L - L * (1 + a * np.exp(b * (t - tj))) ** (-1 / a)
    return j


async def plot_series_altair(q: Q, gc: int, start_date: str, end_date: str):

    if 'epi_year' in q.client and (q.client.epi_year is not None) and (q.client.epi_year != 'all'):

        title = (
            f"{q.client.disease.capitalize()} weekly cases "
            f"in {q.client.epi_year} for {q.client.cities[int(q.client.city)]}"
        )

    else:
        title = (
            f"{q.client.disease.capitalize()} weekly cases "
            f"in {q.client.cities[int(q.client.city)]}"
        ) 

    df = q.client.data_table
    dfcity = df[df.municipio_geocodigo == gc].loc[start_date:end_date]
    dfcity.sort_index(inplace=True)
    dfcity["casos_cum"] = dfcity.casos.cumsum()
    sirp = np.array([])
    if 'epi_year' in q.client and (q.client.epi_year is not None) and (q.client.epi_year != 'all'):
        sirp = q.client.parameters[
            (q.client.parameters.geocode == gc)
            & (q.client.parameters.year == int(q.client.epi_year))
        ][["total_cases", "beta", "gamma", "peak_week"]].values
        if sirp.any():
            a = 1 - (sirp[0, 2] / sirp[0, 1])
            L, b, tj = sirp[0, 0], sirp[0, 1]-sirp[0, 2], sirp[0, 3]
            dfcity["Model fit"] = richards(
                L, a, b, np.arange(len(dfcity.index)), tj
            )

        dfcity['model_legend'] = 'Model'

        vertical_line = alt.Chart(pd.DataFrame({'x': [dfcity.index[int(round(tj,0))]],
                                                'label':['Peak week']})).mark_rule(size=2).encode(
        x='x:T', 
        color = alt.Color('label:N', scale = alt.Scale(domain=['Peak week'], range=['orange']),  # Define specific color for 'Threshold'
        legend=alt.Legend(title=" ", orient = 'left', offset=-130)
        ))

        
    ch1 = (
        alt.Chart( 
            dfcity.reset_index(),
            width=750,
            height=200,
            title={
            "text": f"{title}",
            "fontSize": 15,
            "anchor": "start"},
        )
        .mark_area(
            opacity=0.3,
            interpolate="step-after",
        )
        
        .encode(
            x=alt.X("data_iniSE:T", axis=alt.Axis(title="")),
            y=alt.Y("casos:Q", axis=alt.Axis(title="New Cases", titleFontSize = 12)),
            tooltip=["data_iniSE:T", "casos:Q"],
        )
    )

    ch2 = (
        alt.Chart(
            dfcity.reset_index(),
            width=750,
            height=200,
        )
        .mark_area(
            opacity=0.3,
            interpolate="step-after",
        )
        .encode(
            x=alt.X("data_iniSE:T", axis=alt.Axis(title="Date", titleFontSize = 12)),
            y=alt.Y("casos_cum:Q", axis=alt.Axis(title="Cumulative Cases", titleFontSize=12)),
            tooltip=["data_iniSE:T", "casos_cum:Q", "Model fit:Q"],
        )
    )
    if 'epi_year' in q.client and sirp.any():
        model = (
            alt.Chart(dfcity.reset_index())
            .mark_line()
            .encode(
                x=alt.X("data_iniSE:T", axis=alt.Axis(title="Date")),
                y=alt.Y("Model fit:Q", axis=alt.Axis(title="")),
                color=alt.Color('model_legend:N',
                                scale = alt.Scale(domain=['Model'], range=['red']), 
                                legend=alt.Legend(title="", orient="left",          # Position legend on the left
                direction="vertical",    # Arrange items vertically
                titleAnchor="middle",    # Center-align the title within the legend
                offset=-130))
            )
        )
        spec = alt.vconcat(ch1+vertical_line, (ch2 + vertical_line +model).resolve_scale(
    color='independent')).resolve_scale(
    color='independent')  # Keep color scales independent for separate legends)  # leaving this off for now
    else:
        spec = alt.vconcat(ch1, ch2)
    return spec


async def plot_series_px(q: Q, gc: int, start_date: str, end_date: str):
    """
    Plot timeseries between two dates of city with Plotly
    Args:
        q:
        gc:
        start_date:
        end_date:
    """
    df = q.client.data_table
    dfcity = df[df.municipio_geocodigo == gc].loc[start_date:end_date]
    dfcity.sort_index(inplace=True)
    dfcity["casos_cum"] = dfcity.casos.cumsum()
    spl = make_subplots(rows=2, cols=1)
    spl.add_trace(go.Bar(x=dfcity.index, y=dfcity.casos), row=1, col=1)
    spl.add_trace(go.Bar(x=dfcity.index, y=dfcity.casos_cum), row=2, col=1)
    html = pio.to_html(spl, validate=False, include_plotlyjs=True)
    q.page["ts_plot_px"].content = html
    await q.page.save()
