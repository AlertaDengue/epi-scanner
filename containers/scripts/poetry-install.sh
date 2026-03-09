#!/usr/bin/env bash
set -euo pipefail

POETRY_HOME="/home/epiuser/.local/share/pypoetry"
export PATH="/home/epiuser/.local/bin:${POETRY_HOME}/bin:${PATH}"

curl -sSL https://install.python-poetry.org | python -

cd "${PROJECT_ROOT}"

poetry config virtualenvs.create true --local
poetry config virtualenvs.in-project true --local

poetry env use "${ENV_PREFIX}/bin/python"
poetry check --lock

if [[ "${ENV:-}" == "prod" ]]; then
  poetry install --only main --no-interaction --no-ansi --no-root
else
  poetry install --with dev --no-interaction --no-ansi --no-root
fi

test -x "${PROJECT_ROOT}/.venv/bin/python"
test -x "${PROJECT_ROOT}/.venv/bin/pip"
