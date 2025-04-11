from typing import Optional

import altair as alt
from h2o_wave import Q, ui

from epi_scanner.elements import Card, Chart
from epi_scanner.settings import STATES


class Vega(Card):
    element_id: str
    box: str
    title: str
    chart: alt.Chart = None

    def __init__(
        self,
        q: Q,
        element_id: str,
        box: str,
        title: str,
        chart: Optional[Chart] = None
    ):
        self.element_id = element_id
        self.box = box
        self.title = title
        self.chart = chart

        if not chart:
            q.page[element_id] = ui.image_card(
                box=box,
                title="",
                path="https://api.mosqlimate.org/static/img/loading-dots.gif",
                type="gif",
            )
        else:
            if isinstance(chart, alt.Chart):
                data = chart.to_json()
            else:
                data = chart.chart.to_json()

            q.page[element_id] = ui.vega_card(
                box=box, title=title, specification=data
            )

    def update(self, q: Q, chart: Chart):
        if isinstance(chart, alt.Chart):
            data = chart.to_json()
        else:
            data = chart.chart.to_json()

        q.page[self.element_id] = ui.vega_card(
            box=self.box, title=self.title, specification=data
        )


class Markdown(Card):
    def __init__(
        self, q: Q, element_id: str, box: str, title: str, content: str
    ):
        q.page[element_id] = ui.markdown_card(
            box=box, title=title, content=content
        )


class StateHeader(Markdown):
    def __init__(self, q: Q):
        text = f"## Epidemiological Report for {q.client.disease.title()}"
        super().__init__(
            q=q, element_id="state_header", box="pre", title="", content=text
        )

    @staticmethod
    def update(q: Q, disease: str, uf: str, cases: int, year: int):
        text = (
            f"## Epidemiological Report for {disease.title()}\n ## "
            f"{STATES[uf]}\nCumulative notified cases since "
            f"Jan {year}: {cases}"
        )
        q.page["state_header"].content = text


class Results(Markdown):
    def __init__(self, q: Q):
        text = "**Top 20 most active cities** \n\n"
        super().__init__(
            q=q,
            element_id="results",
            box="sidebar_results",
            title="",
            content=text,
        )

    @staticmethod
    def update(q: Q, results: list[tuple[[str, str]]]):
        text = "**Top 20 most active cities** \n\n"
        for result in results:
            city, line = result
            text += f"**{city}**: {line}\n"
        q.page["results"].content = text
