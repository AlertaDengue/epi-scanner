from epi_scanner.elements import Card, Chart
from epi_scanner.settings import STATES
from h2o_wave import Q, ui


class Vega(Card):
    def __init__(
        self, q: Q, element_id: str, box: str, title: str, chart: Chart
    ):
        q.page[element_id] = ui.vega_card(
            box=box, title=title, specification=chart.chart.to_json()
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
            text += f"**{city}** :{line}\n"
        q.page["results"].content = text
