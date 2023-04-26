import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
from epi_scanner.management.fetch_data import data_to_parquet
from epi_scanner.model.scanner import EpiScanner
from epi_scanner.settings import EPISCANNER_DATA_DIR, STATES


def export_data_to_dir(
    state: str, diseases: List[str], output_dir: Optional[str] = None
) -> None:
    """
    Export data for a single state to Parquet and CSV files.

    Parameters
    ----------
    state : str
        The abbreviation of the state to export.
    diseases : list of str
        The diseases to export data for.
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
        if output_dir is not None:
            output_dir = Path(output_dir)
        else:
            output_dir = EPISCANNER_DATA_DIR

        if not output_dir.exists():
            raise FileNotFoundError(
                f"Error: {output_dir} directory does not exist."
            )

        if state not in STATES:
            raise ValueError(
                f"""Error: Invalid state abbreviation {state}.
                    Valid states are {STATES}.
                """
            )

        for disease in diseases:
            parquet_file = output_dir / f"{state}_{disease}.parquet"

            if not parquet_file.exists():
                # fetch data and save it to Parquet file
                data_to_parquet(state, output_dir=output_dir, disease=disease)

            if not parquet_file.exists():
                raise FileNotFoundError(
                    f"Error: Failed to save data to {parquet_file}."
                )

            df = pd.read_parquet(parquet_file)
            if df.empty:
                raise pd.errors.EmptyDataError(
                    f"Error: {parquet_file} is empty."
                )

            SE = int(df.SE.max()) % 100
            scanner = EpiScanner(last_week=SE, data=df)
            csv_file = output_dir / f"curves_{state}_{disease}.csv.gz"
            for gc in df.municipio_geocodigo.unique():
                scanner.scan(gc)
            scanner.to_csv(csv_file)

    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export data for a single state to Parquet and CSV files."
    )
    parser.add_argument(
        "state", type=str, help="The abbreviation of the state to export."
    )
    parser.add_argument(
        "diseases",
        nargs="+",
        type=str,
        help="The diseases to export data for.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="The directory where the output files will be saved.",
    )

    args = parser.parse_args()

    export_data_to_dir(args.state, args.diseases, output_dir=args.output_dir)
