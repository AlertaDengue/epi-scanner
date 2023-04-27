from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from epi_scanner.management.fetch_data import data_to_parquet, get_alerta_table
from epi_scanner.model.scanner import EpiScanner


def test_get_alerta_table():
    # Test case with municipality code argument
    df = get_alerta_table(municipio_geocodigo="5300108")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 5300108 in df["municipio_geocodigo"].values

    # Test case with state abbreviation argument
    df = get_alerta_table(state_abbv="DF")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.SE.min() == 201001

    # Test case with no arguments
    # df = get_alerta_table()
    # assert isinstance(df, pd.DataFrame)
    # assert len(df) > 0


# Define a fixture to create a temporary directory for testing
@pytest.fixture(scope="module")
def tmp_data_dir():
    """Create a temporary directory for testing."""
    test_dir = Path("/tmp/epi_scanner/data")
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    # shutil.rmtree(test_dir)


# Define a fixture to create test data for the data_to_parquet() function
@pytest.fixture
def uf_data(tmp_data_dir):
    uf = "AC"
    disease = "dengue"
    data_file = Path(tmp_data_dir) / f"{uf}_dengue.parquet"
    return uf, disease, pd.read_parquet(data_file)


def test_data_to_parquet(uf_data, tmp_data_dir):
    """Test if parquet file is created for a specific state and disease."""
    state_abbv, disease, df = uf_data

    # Call the function to create the parquet file
    parquet_fname = data_to_parquet(
        state_abbv=state_abbv, disease=disease, output_dir=tmp_data_dir
    )

    # Create a path to the parquet file
    parquet_fpath = tmp_data_dir / f"{parquet_fname}"

    # Check if the parquet file exists
    # assert str(parquet_fpath) == '/tmp/epi_scanner/data/AC_dengue.parquet'
    assert parquet_fpath.is_file(), f"{parquet_fpath} does not exist"

    # Load the data from the parquet file and check if it's a Pandas DataFrame
    # df = pd.read_parquet(parquet_fpath)
    assert isinstance(df, pd.DataFrame), "Data is not a Pandas DataFrame"

    # Check if the parquet file was saved in the correct directory
    expected_pqfile = f"/tmp/epi_scanner/data/{state_abbv}_dengue.parquet"
    assert (
        str(parquet_fname) == expected_pqfile
    ), f"{parquet_fname} does not match expected path"


def test_save_to_csv_gz_file(uf_data, tmp_data_dir):
    state_abbv, disease, data_table = uf_data

    # Initialize EpiScanner model with data table and a specific date
    model = EpiScanner(202306, data_table)

    # Set filename for CSV output
    fname = f"curves_{state_abbv}_{disease}"

    # Define the full file path
    fname_path = tmp_data_dir / f"{fname}.csv.gz"

    # Save model output to CSV file in gzip format
    model.to_csv(str(fname_path))

    # Check if the filename and path match the expected values
    assert str(fname_path) == "/tmp/epi_scanner/data/curves_AC_dengue.csv.gz"
    assert fname_path.is_file(), f"{fname_path} does not exist"


def test_open_csv_gz_file():
    # Define the path to the CSV file using pathlib
    filepath = Path("/tmp/epi_scanner/data/curves_AC_dengue.csv.gz")

    # Check if the file exists
    if not filepath.exists():
        pytest.fail(f"File does not exist: {filepath}")

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(
        filepath, usecols=lambda x: x != "Unnamed: 0", encoding="utf-8"
    )
    # df = pd.read_csv(filepath, encoding="utf-8")

    # Check if the DataFrame has the expected number of rows and columns
    expected_shape = (312, 8)  # e.g., 100 rows, 8 columns
    assert df.shape == expected_shape

    expected_from_df = {
        "geocode": np.int64,
        "year": np.int64,
        "peak_week": np.float64,
        "beta": np.float64,
        "gamma": np.float64,
        "R0": np.float64,
        "total_cases": np.float64,
        "alpha": np.float64,
    }

    expected_columns = list(expected_from_df.keys())

    # Check if the DataFrame contains the expected column names
    assert list(df.columns) == expected_columns

    # Check if the DataFrame contains the expected data type for each column
    for col, dtype in expected_from_df.items():
        assert df[col].dtype == dtype
