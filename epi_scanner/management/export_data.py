from pathlib import Path
from typing import Optional

from epi_scanner.management.fetch_data import data_to_parquet
from epi_scanner.settings import EPISCANNER_DATA_DIR, STATES


def export_data(output_dir: Optional[str] = None) -> Path:
    """
    Export data for all states to Parquet files.

    Parameters
    ----------
    output_dir : str, optional
        The directory where the Parquet files will be saved.
        If not provided, the default directory set in
        `EPISCANNER_DATA_DIR` will be used.

    Returns
    -------
    Path
        A `Path` object representing the directory where
        the Parquet files are saved.

    Raises
    ------
    FileNotFoundError
        If the output directory does not exist.

    Notes
    -----
    This function exports the data for all states to Parquet files
    and saves them in the specified output directory. If the output
    directory is not provided, the default directory set in the
    `EPISCANNER_DATA_DIR` constant is used.

    Example
    -------
    >>> export_data(output_dir="/tmp/epi_scanner/data")
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
    except FileNotFoundError as e:
        print(e)
        exit()

    for state in STATES:
        data_to_parquet(state, output_dir=output_dir)

    return output_dir


if __name__ == "__main__":
    # HOST_EPISCANNER_DATA_DIR
    export_data()
