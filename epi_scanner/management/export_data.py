from pathlib import Path
from typing import Optional

import pandas as pd
from epi_scanner.management.fetch_data import data_to_parquet
from epi_scanner.model.scanner import EpiScanner
from epi_scanner.settings import EPISCANNER_DATA_DIR, STATES


def export_data_to_dir(state: str, output_dir: Optional[str] = None) -> None:
    """
    Export data for a single state to Parquet and CSV files.

    Parameters
    ----------
    state : str
        The abbreviation of the state to export.
    output_dir : str, optional
        The directory where the output files will be saved.
        If not provided, the default directory set in
        `EPISCANNER_DATA_DIR` will be used.

    Raises
    ------
    FileNotFoundError
        If the output directory does not exist or the input Parquet file
        for the specified state does not exist.
    """
    try:
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = EPISCANNER_DATA_DIR
        if not output_dir.exists():
            raise FileNotFoundError(
                f"Error: {output_dir} directory does not exist."
            )

        parquet_file = output_dir / f"{state}_dengue.parquet"
        if not parquet_file.exists():
            data_to_parquet(state, output_dir=output_dir)
        df = pd.read_parquet(parquet_file)
        model = EpiScanner(202306, df)
        csv_file = output_dir / f"curves_{state}.csv.gz"
        model.to_csv(csv_file)

    except FileNotFoundError as e:
        print(e)
        exit()


if __name__ == "__main__":
    output_dir = "/tmp/epi_scanner/data"
    states = list(STATES.keys())
    for state in states:
        export_data_to_dir(state, output_dir)
