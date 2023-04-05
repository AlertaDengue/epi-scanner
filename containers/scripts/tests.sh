#!/usr/bin/env bash

current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${current_dir}; cd ..

# Set the environment variable
export CTNR_EPISCANNER_DATA_DIR="/opt/services/epi_scanner/data"
# Run the test

exec pytest -vv tests/test_fetch_data.py
