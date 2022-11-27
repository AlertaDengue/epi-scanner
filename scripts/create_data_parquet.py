#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)


PSQL_DB = os.getenv("PSQL_DB")
PSQL_DBF = os.getenv("PSQL_DBF")
PSQL_USER = os.getenv("PSQL_USER")
PSQL_HOST = os.getenv("PSQL_HOST")
PSQL_PASSWORD = os.getenv("PSQL_PASSWORD")
PSQL_PORT = os.getenv("PSQL_PORT")

PSQL_URI = (
    "postgresql://"
    f"{PSQL_USER}:{PSQL_PASSWORD}@{PSQL_HOST}:{PSQL_PORT}/{PSQL_DB}"
)


def make_connection():
    """
    Returns:
        db_engine: URI with driver connection.
    """
    try:
        connection = create_engine(PSQL_URI)
    except ConnectionError as e:
        raise e
    return connection


def get_disease_suffix(disease: str, empty_for_dengue: bool = True):
    """
    :param disease:
    :return: suffix to table name
    """
    return (
        ("" if empty_for_dengue else "_dengue")
        if disease == "dengue"
        else "_chik"
        if disease == "chikungunya"
        else "_zika"
        if disease == "zika"
        else ""
    )


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

    connection = make_connection()

    states_dict = {
        "RJ": "Rio de Janeiro",
        "ES": "Espírito Santo",
        "PR": "Paraná",
        "CE": "Ceará",
        "MA": "Maranhão",
        "MG": "Minas Gerais",
        "SC": "Santa Catarina",
        "PE": "Pernambuco",
        "PB": "Paraíba",
        "SE": "Sergipe",
        "SP": "São Paulo",
        "RS": "Rio Grande do Sul",
    }

    # Need the name of the state to query DengueGlobal table
    if state_abbv in states_dict:
        state_name = states_dict.get(state_abbv)

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

    data_path = Path("epi_scanner/data")
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
