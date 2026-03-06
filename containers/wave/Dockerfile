FROM condaforge/mambaforge:latest

LABEL maintainer="Sandro Loch <es.loch@gmail.com>"

SHELL ["/bin/bash", "-c"]
USER root

ARG HOST_UID=1000
ARG HOST_GID=1000
ARG ENV=dev
ARG PROJECT_ROOT=/opt/services/epi-scanner

ENV DEBIAN_FRONTEND=noninteractive
ENV ENV_NAME=episcanner
ENV ENV_PREFIX=/opt/conda/envs/${ENV_NAME}
ENV ENV=${ENV}
ENV PROJECT_ROOT=${PROJECT_ROOT}
ENV PYTHONPATH=${PROJECT_ROOT}

RUN apt-get -qq update --yes \
  && apt-get -qq install --yes --no-install-recommends \
  build-essential ca-certificates curl sudo tini \
  && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid ${HOST_GID} epiuser \
  && useradd --uid ${HOST_UID} --gid ${HOST_GID} -ms /bin/bash epiuser \
  && echo "epiuser ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/epiuser \
  && chmod 0440 /etc/sudoers.d/epiuser \
  && mkdir -p /opt/services/logs ${PROJECT_ROOT} /tmp/conda \
  && chown -R ${HOST_UID}:${HOST_GID} /opt/services ${PROJECT_ROOT} /tmp

# Ensure the target prefix is writable while creating the env
RUN mkdir -p /opt/conda/envs \
  && chown -R ${HOST_UID}:${HOST_GID} /opt/conda/envs

COPY --chown=${HOST_UID}:${HOST_GID} ./conda/ /tmp/conda/

# Create conda env at a fixed prefix (as root or with ensured permissions)
RUN conda env create -p "${ENV_PREFIX}" --file /tmp/conda/env-base.yaml \
  && conda clean --all -y \
  && rm -rf /opt/conda/pkgs /tmp/conda

RUN "${ENV_PREFIX}/bin/python" -V

# Now switch to epiuser for Poetry/venv work
USER epiuser
WORKDIR ${PROJECT_ROOT}

ENV POETRY_HOME=/home/epiuser/.local/share/pypoetry
ENV PATH=/home/epiuser/.local/bin:${POETRY_HOME}/bin:${ENV_PREFIX}/bin:${PATH}

COPY --chown=${HOST_UID}:${HOST_GID} pyproject.toml poetry.lock README.md ${PROJECT_ROOT}/
COPY --chown=${HOST_UID}:${HOST_GID} ./containers/scripts/poetry-install.sh \
  ${PROJECT_ROOT}/poetry-install.sh

# deps (no-root)
RUN bash ${PROJECT_ROOT}/poetry-install.sh

ENV PATH="${PROJECT_ROOT}/.venv/bin:${PATH}"

# now copy code
COPY --chown=${HOST_UID}:${HOST_GID} ./epi_scanner ${PROJECT_ROOT}/epi_scanner

# install the project itself into the same .venv
RUN /home/epiuser/.local/share/pypoetry/bin/poetry install --no-interaction --no-ansi

COPY --chown=${HOST_UID}:${HOST_GID} ./containers/scripts/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh \
  && echo "source /opt/entrypoint.sh" > ~/.bashrc

ENTRYPOINT ["tini", "--", "/opt/entrypoint.sh"]