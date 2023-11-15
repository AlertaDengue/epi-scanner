#!/usr/bin/env bash

set -ex

if [[ $ENV == "prod" ]]; then
  export POETRY_INSTALL_ARGS="--only main"
fi

poetry config virtualenvs.create true
poetry install $POETRY_INSTALL_ARGS
