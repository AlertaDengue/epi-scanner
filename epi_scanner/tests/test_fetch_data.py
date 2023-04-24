# import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from epi_scanner.management.fetch_data import data_to_parquet, get_alerta_table
from epi_scanner.model.scanner import EpiScanner


#
def test_get_alerta_table():

    # # Test case with municipality code argument
    df = get_alerta_table(municipio_geocodigo="5300108")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 5300108 in df["municipio_geocodigo"].values

    # Test case with state abbreviation argument
    df = get_alerta_table(state_abbv="DF")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.SE.min() == 201001


#
@pytest.fixture(scope="module")
def tmp_data_dir():
    """Create a temporary directory for testing."""
    test_dir = Path("/tmp/epi_scanner/data")
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    # shutil.rmtree(test_dir)


def test_data_to_parquet(tmp_data_dir):
    """Test if parquet file is created for a specific state and disease."""
    state_abbv = "AC"
    disease = "dengue"

    # Call the function to create the parquet file
    parquet_fname = data_to_parquet(
        state_abbv=state_abbv, disease=disease, output_dir=tmp_data_dir
    )

    # Create a path to the parquet file
    parquet_fpath = tmp_data_dir / f"{parquet_fname}"

    # Check if the parquet file exists
    assert parquet_fpath.is_file(), f"{parquet_fpath} does not exist"

    # Load the data from the parquet file and check if it's a Pandas DataFrame
    df = pd.read_parquet(parquet_fpath)
    assert isinstance(df, pd.DataFrame), "Data is not a Pandas DataFrame"

    # Check if the parquet file was saved in the correct directory
    expected_pqfile = f"/tmp/epi_scanner/data/{state_abbv}_dengue.parquet"
    assert (
        str(parquet_fname) == expected_pqfile
    ), f"{parquet_fname} does not match expected path"


#
@pytest.fixture
def uf_data(tmp_data_dir):
    uf = "AC"
    data_file = Path(tmp_data_dir) / f"{uf}_dengue.parquet"
    return uf, pd.read_parquet(data_file)


def test_save_to_csv_gz_file(uf_data, tmp_data_dir):
    uf, data_table = uf_data

    model = EpiScanner(202306, data_table)
    fname = f"curves_{uf}"
    fname_path = tmp_data_dir / f"{fname}.csv.gz"
    model.to_csv(str(fname_path))

    # Print statements
    # print(f"File path: {fname_path}")
    # print(f"File exists: {fname_path.is_file()}")
    # print(data_table)

    assert fname_path.is_file(), f"{fname_path} does not exist"

    assert str(fname_path) == f"/tmp/epi_scanner/data/curves_{uf}.csv.gz"


def test_open_csv_gz_file():
    # Define the path to the CSV file using pathlib
    filepath = Path("/tmp/epi_scanner/data/curves_RJ.csv.gz")

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
