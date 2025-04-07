from epi_scanner.elements import Card, Chart

from h2o_wave import Q, ui


class Vega(Card):
    def __init__(
        self,
        q: Q,
        element_id: str,
        box: str,
        title: str,
        chart: Chart
    ):
        q.page[element_id] = ui.vega_card(
            box=box,
            title=title,
            specification=chart.chart.to_json()
        )


class Markdown(Card):
    def __init__(
        self,
        q: Q,
        element_id: str,
        box: str,
        title: str,
        content: str
    ):
        q.page[element_id] = ui.markdown_card(
            box=box,
            title=title,
            content=content
        )


class Results(Markdown):
    def __init__(self, q: Q):
        results = "**Top 20 most active cities** \n\n"
        super().__init__(
            q=q,
            element_id="results",
            box="sidebar_results",
            title="",
            content=results
        )

    @staticmethod
    def update(q: Q):
        results = "**Top 20 most active cities** \n\n"
        cities = q.client.parameters.groupby("geocode")
        report = {}
        for gc, citydf in cities:
            if len(citydf) < 1:
                continue
            report[
                q.client.cities[gc]
            ] = f"{len(citydf)} epidemic years: {list(sorted(citydf.year))}\n"
        for n, linha in sorted(
            report.items(), key=lambda x: int(x[1][0:2]), reverse=True
        )[:20]:
            results += f"**{n}** :{linha}\n"
        q.page["results"].content = results
