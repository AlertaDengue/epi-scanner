import datetime
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
from epi_scanner.elements import Chart
from epi_scanner.viz import get_ini_end_week, richards


class EpidemicCalculator(Chart):
    def __init__(
        self,
        disease: str,
        city: str,
        year: int = datetime.date.today().year,
        #
        df: Optional[pd.DataFrame] = None,
        gc: Optional[int] = None,
        pw: Optional[int] = None,
        R0: Optional[float] = None,
        total_cases: Optional[int] = None,
    ):
        self.chart = self.do_chart(
            disease=disease,
            city=city,
            year=year,
            df=df,
            gc=gc,
            pw=pw,
            R0=R0,
            total_cases=total_cases,
        )

    @staticmethod
    def do_chart(
        disease: str,
        city: str,
        year: int,
        df: Optional[pd.DataFrame] = None,
        gc: Optional[int] = None,
        pw: Optional[int] = None,
        R0: Optional[float] = None,
        total_cases: Optional[int] = None,
    ) -> alt.Chart:
        start_date, end_date = get_ini_end_week(year=year)

        if df is not None:
            df = df[df.municipio_geocodigo == gc].loc[start_date:end_date]
            df.sort_index(inplace=True)
            df["casos_cum"] = df.casos.cumsum()
            df = df.reset_index().loc[:, ["data_iniSE", "casos_cum"]]

            r = 1 - 1 / R0
            gamma = 0.3
            b = r * gamma / (1 - r)
            a = b / (gamma + b)

            dfcity2 = pd.DataFrame()
            dfcity2["data_iniSE"] = pd.date_range(
                start=df.data_iniSE.values[0], periods=52, freq="W-SUN"
            )
            dfcity2["model"] = richards(total_cases, a, b, np.arange(52), pw)
            df = df.merge(
                dfcity2,
                left_on="data_iniSE",
                right_on="data_iniSE",
                how="outer",
            )
        else:
            dtypes = {
                "data_iniSE": "datetime64[ns]",
                "casos_cum": "int64",
                "model": "float64",
            }
            df = pd.DataFrame(columns=dtypes.keys()).astype(dtypes)

        df1 = df.copy()
        df1["legend"] = "Data"

        df2 = df.copy()
        df2["legend"] = "Model"

        scale = alt.Scale(
            domain=["Data", "Model", "Peak week"],
            range=["#1f77b4", "#ff7f0e", "red"],
        )

        title = f"{disease.capitalize()} weekly cases in {year} for {city}"

        ch1 = (
            alt.Chart(df1, width=650, height=350)
            .mark_area(
                opacity=0.3,
                interpolate="step-after",
                color="#1f77b4",
            )
            .encode(
                x=alt.X(
                    "data_iniSE:T",
                    axis=alt.Axis(title="Date", titleFontSize=12),
                ),
                y=alt.Y(
                    "casos_cum:Q",
                    axis=alt.Axis(title="Cumulative Cases", titleFontSize=12),
                ),
                color=alt.Color("legend:N", title=" ", scale=scale),
                tooltip=["data_iniSE:T", "casos_cum:Q", "model:Q"],
            )
        )

        ch2 = (
            alt.Chart(df2, width=650, height=350)
            .mark_line(color="red")
            .encode(
                x=alt.X(
                    "data_iniSE:T",
                    axis=alt.Axis(title="Date", titleFontSize=12),
                ),
                y=alt.Y(
                    "model:Q",
                    axis=alt.Axis(title="Cumulative Cases", titleFontSize=12),
                ),
                color=alt.Color("legend:N", title=" ", scale=scale),
                tooltip=["data_iniSE:T", "casos_cum:Q", "model:Q"],
            )
        )

        ch2_points = (
            alt.Chart(df2, width=650, height=350)
            .mark_point(size=60, filled=True, color="red")
            .encode(
                x=alt.X(
                    "data_iniSE:T",
                    axis=alt.Axis(title="Date", titleFontSize=12),
                ),
                y=alt.Y(
                    "model:Q",
                    axis=alt.Axis(title="Cumulative Cases", titleFontSize=12),
                ),
                color=alt.Color(
                    "legend:N", title=" ", scale=scale
                ),  # Assign color based on legend,
                tooltip=["data_iniSE:T", "casos_cum:Q", "model:Q"],
            )
            .properties(title=title)
        )

        if pw:
            vline_date = df2.data_iniSE[int(round(pw, 0))]
        else:
            vline_date = None

        vertical_line = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "data_iniSE": [vline_date],
                        "label": ["Peak week"],
                    }
                )
            )
            .mark_rule(size=2, color="orange")
            .encode(
                x=alt.X(
                    "data_iniSE:T",
                    axis=alt.Axis(title="Date", titleFontSize=12),
                ),
                color=alt.Color(
                    "label:N",
                    scale=scale,
                    legend=alt.Legend(title=" ", orient="left", offset=-130),
                ),
            )
        )

        return ch1 + ch2 + ch2_points + vertical_line

    async def update(self, *args, **kwargs):
        return self
