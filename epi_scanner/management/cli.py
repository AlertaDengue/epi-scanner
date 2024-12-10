import argparse

from epi_scanner.management.fetch_data import data_to_parquet
from epi_scanner.settings import EPISCANNER_DATA_DIR


def Command():
    """ """
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "disease",
        help="disease name. Options: ['dengue', 'zika', 'chikungunya']",
    )
    parser.add_argument(
        "uf",
        nargs="?",
        default=None,
        help="state abbreviation codes. None to all UFs",
    )

    args = parser.parse_args()

    return data_to_parquet(args.uf, args.disease, EPISCANNER_DATA_DIR)


if __name__ == "__main__":
    Command()
