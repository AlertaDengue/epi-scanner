import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

# This variable is used by container application modules
ROOT_DIR = Path(__file__).resolve(strict=True).parent.parent

# Episcanner/
APPS_DIR = ROOT_DIR / "epi_scanner"

# Stores the file path to the data directory.
# epi_scanner/data
EPISCANNER_DATA_DIR = APPS_DIR / "data"


def make_connection():
    """
    Returns:
        db_engine: URI with driver connection.
    """
    PSQL_DB = os.getenv("PSQL_DB")
    PSQL_USER = os.getenv("PSQL_USER")
    PSQL_HOST = os.getenv("PSQL_HOST")
    PSQL_PASSWORD = os.getenv("PSQL_PASSWORD")
    PSQL_PORT = os.getenv("PSQL_PORT")

    PSQL_URI = (
        "postgresql://"
        f"{PSQL_USER}:{PSQL_PASSWORD}@{PSQL_HOST}:{PSQL_PORT}/{PSQL_DB}"
    )

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


STATES = {
    "AC": "Acre",
    "AL": "Alagoas",
    "AM": "Amazonas",
    "AP": "Amapá",
    "BA": "Bahia",
    "CE": "Ceará",
    "DF": "Distrito Federal",
    "ES": "Espírito Santo",
    "GO": "Goiás",
    "MA": "Maranhão",
    "MG": "Minas Gerais",
    "MS": "Mato Grosso do Sul",
    "MT": "Mato Grosso",
    "PA": "Pará",
    "PB": "Paraíba",
    "PE": "Pernambuco",
    "PI": "Piauí",
    "PR": "Paraná",
    "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte",
    "RO": "Rondônia",
    "RR": "Roraima",
    "RS": "Rio Grande do Sul",
    "SC": "Santa Catarina",
    "SE": "Sergipe",
    "SP": "São Paulo",
    "TO": "Tocantins",
}
