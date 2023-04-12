from pathlib import Path
from typing import Optional

import pandas as pd

# Local
from epi_scanner.settings import (  # EPISCANNER_DATA_DIR,
    STATES,
    get_disease_suffix,
    make_connection,
)

# from tqdm import tqdm


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

    if state_abbv in STATES:
        state_name = STATES.get(state_abbv)

    table_suffix = ""
    if disease != "dengue":
        table_suffix = get_disease_suffix(disease)

    # Need the name of the state to query DengueGlobal table

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

    # print(state_abbv, ">>>", df.data_iniSE.max(), df.data_iniSE.min())

    connection.dispose()

    df.data_iniSE = pd.to_datetime(df.data_iniSE)

    df.set_index("data_iniSE", inplace=True)

    return df


def data_to_parquet(
    state_abbv: Optional[str] = None,
    disease: str = "dengue",
    output_dir: Optional[str] = None,
) -> Path:
    """
    Create the parquet files for each disease state within the data directory.

    Parameters
    ----------
        state_abbv: abbreviated codes of the federative units of Brazil
        disease: name of disease {'dengue', 'chik', 'zika'}
        output_dir: directory where the parquet file will be saved
    Returns
    -------
        pathlib: Path to the parquet file with the name of the disease by state
    """

    CID10 = {"dengue": "A90", "chikungunya": "A92.0", "zika": "A928"}

    if disease not in CID10.keys():
        raise ValueError(
            f"""
            Invalid disease name: {disease}.
            Available diseases: {list(CID10.keys())}
            """
        )

    if state_abbv is None:
        print(
            "Saving the parquet files for all states in the data directory..."
        )

        for state in STATES.keys():
            pq_fname = f"{state}_{disease}.parquet"

            if output_dir:
                output_dir = Path(output_dir)
                pq_fname_path = output_dir / pq_fname

                if not output_dir.exists():
                    raise FileNotFoundError(
                        f"""
                        Output directory not found: {output_dir}.
                        Please create it before running this function.
                        """
                    )

            else:
                pq_fname_path = Path(pq_fname)
                print(
                    f"Saving {pq_fname} in the root directory."
                )

            get_alerta_table(state_abbv=state, disease=disease).to_parquet(
                pq_fname_path
            )

        return pq_fname_path

    else:
        pq_fname = f"{state_abbv}_{disease}.parquet"

        if output_dir:
            output_dir = Path(output_dir)
            pq_fname_path = output_dir / pq_fname

            if not output_dir.exists():
                raise FileNotFoundError(
                    f"""
                    Output directory not found: {output_dir}.
                    Please create it before running this function.
                    """
                )

        else:
            pq_fname_path = Path(pq_fname)
            print(
                f"Saving the {pq_fname_path} in the root directory."
            )

        get_alerta_table(state_abbv=state_abbv, disease=disease).to_parquet(
            pq_fname_path
        )

        print(
            f"The parquet file was successfully created in: {pq_fname_path}"
        )
        return pq_fname_path
