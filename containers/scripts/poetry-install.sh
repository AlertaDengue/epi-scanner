#!/usr/bin/env bash

set -ex

poetry config virtualenvs.create false
poetry config installer.max-workers 10
poetry install --only main --no-interaction --no-ansi
