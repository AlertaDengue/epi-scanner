from abc import ABC, abstractmethod
from typing import TypeVar

import altair as alt
from h2o_wave import Q, ui

ChartT = TypeVar("ChartT", bound="Chart")
ElementT = TypeVar("ElementT", bound="Element")
CardT = TypeVar("CardT", bound="Card")


class Element(ABC):
    @staticmethod
    def get(q: Q, element_id: str):
        if element_id not in q.page:
            raise ValueError(f"{element_id} does not exist")
        return q.page[element_id]


class Chart(Element, ABC):
    chart: alt.Chart

    @abstractmethod
    async def update(self, *args, **kwargs) -> ChartT:
        ...


class Card(Element, ABC):
    ...
