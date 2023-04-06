import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from epi_scanner.management.fetch_data import data_to_parquet, get_alerta_table
from epi_scanner.settings import CTNR_EPISCANNER_DATA_DIR


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
@pytest.mark.skipif(reason="Create a real data directory")
def temp_dir():
    """Create a temporary directory for testing."""
    test_dir = Path("./temp_test_dir/data")
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    shutil.rmtree(test_dir)


def test_data_to_parquet(temp_dir):
    """Test if parquet file is created for a specific state and disease."""
    state_abbv = "DF"
    disease = "dengue"

    # Call the function to create the parquet file
    file_path = data_to_parquet(state_abbv=state_abbv, disease=disease)

    # Check if the parquet file exists
    assert os.path.exists(file_path), f"{file_path} does not exist"

    # Load the data from the parquet file and check if it's a Pandas DataFrame
    df = pd.read_parquet(file_path)
    assert isinstance(df, pd.DataFrame), "Data is not a Pandas DataFrame"

    # Check if the parquet file was saved in the correct directory
    expected_path = CTNR_EPISCANNER_DATA_DIR / f"{state_abbv}_{disease}.parquet"
    assert Path(file_path) == expected_path, f"{file_path} does not match expected path"
