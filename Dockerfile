FROM python:3.10-slim

RUN pip install -U pip poetry h20-wave

WORKDIR /app

# Copy data, app and dependencies
COPY ./app.py ./app.py
COPY ./pyproject.toml ./pyproject.toml
COPY ./poetry.lock ./poetry.lock

# Install dependencies
RUN poetry install

ENTRYPOINT [ "wave", "run", "app.py" ]