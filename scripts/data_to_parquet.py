#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from epi_scanner.settings import (
    DATA_HOST_DIR,
    UFs_dict,
    get_disease_suffix,
    make_connection,
)

connection = make_connection()


def get_alerta_table(
    municipio_geocodigo: Optional[str] = None,
    state_abbv: Optional[str] = None,
    disease: str = "dengue",
) -> pd:
    """
    Pulls the data from a single city, cities from a state or all cities from
    the InfoDengue database.

    Parameters
    ----------
        city: geocode (one city) or None (all)
        state_abbv: abbreviation codes of the federative units of Brazil
        disease: name of disease {'dengue', 'chik', 'zika'}
    Returns
    -------
        df: Pandas dataframe
    """

    # Need the name of the state to query DengueGlobal table

    if state_abbv in UFs_dict:
        state_name = UFs_dict.get(state_abbv)

    table_suffix = ""
    if disease != "dengue":
        table_suffix = get_disease_suffix(disease)

    if municipio_geocodigo is None:
        query = f"""
            SELECT historico.*
            FROM "Municipio"."Historico_alerta{table_suffix}" historico
            JOIN "Dengue_global"."Municipio" municipio
            ON historico.municipio_geocodigo=municipio.geocodigo
            WHERE municipio.uf=\'{state_name}\'
            ORDER BY "data_iniSE" DESC ;"""

    else:
        query = f"""
            SELECT *
            FROM "Municipio"."Historico_alerta{table_suffix}"
            WHERE municipio_geocodigo={municipio_geocodigo}
            ORDER BY "data_iniSE" DESC ;"""

    df = pd.read_sql_query(query, connection, index_col="id")

    connection.dispose()

    df.data_iniSE = pd.to_datetime(df.data_iniSE)

    df.set_index("data_iniSE", inplace=True)

    return df


def data_to_parquet(
    state_abbv: str,
    disease: str = "dengue",
) -> Path:
    """
    Create the parquet files for each disease state within the data directory.

    Parameters
    ----------
        state_abbv: abbreviated codes of the federative units of Brazil
        disease: name of disease {'dengue', 'chik', 'zika'}
    Returns
    -------
        pathlib: Path to the parquet file with the name of the disease by state
    """

    CID10 = {"dengue": "A90", "chikungunya": "A92.0", "zika": "A928"}

    if disease not in CID10.keys():
        raise Exception(
            f"The diseases available are: {[k for k in CID10.keys()]}"
        )

    df = get_alerta_table(
        state_abbv=state_abbv,
        disease=disease,
    )

    data_path = Path(f"{DATA_HOST_DIR}")
    data_path.mkdir(parents=True, exist_ok=True)

    parquet_fname = f"{data_path}/{state_abbv}_{disease}.parquet"

    return df.to_parquet(parquet_fname)


# receive arguments
parser = argparse.ArgumentParser()
parser.add_argument("state_abbv", help="state abbreviation codes")
parser.add_argument("disease", help="disease name")

args = parser.parse_args()

data_to_parquet(args.state_abbv, args.disease)

print("The parquet file was created in the data directory!")
