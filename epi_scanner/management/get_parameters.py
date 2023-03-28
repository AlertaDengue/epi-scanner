import argparse

from epi_scanner.management.fetch_data import data_to_parquet


def Command():
    """ """
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("state_abbv", help="state abbreviation codes")
    parser.add_argument("disease", help="disease name")

    args = parser.parse_args()

    return data_to_parquet(args.state_abbv, args.disease)


if __name__ == "__main__":
    Command()
