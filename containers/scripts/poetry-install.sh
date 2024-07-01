#!/usr/bin/env bash

set -ex

if [[ $ENV == "prod" ]]; then
  export POETRY_INSTALL_ARGS="--only main"
fi

poetry config virtualenvs.create false
poetry config installer.max-workers 10
poetry install $POETRY_INSTALL_ARGS --no-interaction --no-ansi
