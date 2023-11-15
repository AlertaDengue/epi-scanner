# ref: https://github.com/mamba-org/micromamba-docker/blob/main/Dockerfile

FROM condaforge/mambaforge:latest

LABEL maintainer="Sandro Loch <es.loch@gmail.com>"
LABEL org.opencontainers.image.title="EpiGraphHub"
LABEL org.opencontainers.image.authors="EpiGraphHub Team"
LABEL org.opencontainers.image.source="https://github.com/thegraphnetwork/wave-docker-template"
LABEL org.opencontainers.image.version="latest"
LABEL org.opencontainers.image.description="EpiGraphHub"
LABEL org.thegraphnetwork.epigraphhub.version="latest"

# it is the default, but using it here to have it explicitly
USER root

SHELL ["/bin/bash", "-c"]
# Use bash in Dockerfile RUN commands and make sure bashrc is sourced when
# executing commands with /bin/bash -c
# Needed to have the micromamba activate command configured etc.

ARG HOST_UID
ARG HOST_GID
ARG SERVICES
ARG ENV
ARG PROJECT_ROOT
ENV ENV_NAME=episcanner
ENV DEBIAN_FRONTEND=noninteractive

# Change shell to bash
SHELL ["/bin/bash", "-c"]

RUN apt-get -qq update --yes \
  && apt-get -qq install --yes --no-install-recommends \
  build-essential \
  ca-certificates sudo curl\
  && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid ${HOST_GID} epiuser \
  && useradd --uid ${HOST_UID} --gid ${HOST_GID} -ms /bin/bash epiuser \
  && echo "epiuser ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/epiuser \
  && chmod 0440 /etc/sudoers.d/epiuser \
  && mkdir -p /opt/conda /opt/services/logs ${PROJECT_ROOT} \
  && chmod -R a+rwx /opt/conda /opt/services/ /tmp \
  && export ENV_NAME="$ENV_NAME" \
  && chown  ${HOST_UID}:${HOST_GID} /opt/services/ /tmp

ENV PATH "$PATH:/home/epiuser/.local/bin"
ENV PYTHONPATH='${PROJECT_ROOT}'
ENV PATH /opt/conda/envs/$ENV_NAME/bin:$PATH

USER epiuser

WORKDIR ${PROJECT_ROOT}/

COPY --chown=${HOST_UID}:${HOST_GID} ./conda/ /tmp/conda

# Use environment to update the env episcanner
RUN conda env create -n $ENV_NAME --file /tmp/conda/env-base.yaml \
  && conda clean --all \
  && find /opt/conda/ -type f,l -name '*.a' -delete \
  && find /opt/conda/ -type f,l -name '*.pyc' -delete \
  && find /opt/conda/ -type f,l -name '*.js.map' -delete \
  && rm -rf /opt/conda/pkgs

COPY --chown=${HOST_UID}:${HOST_GID} ./epi_scanner ${PROJECT_ROOT}/epi_scanner

COPY --chown=${HOST_UID}:${HOST_GID} poetry.lock pyproject.toml README.md ${PROJECT_ROOT}/

COPY --chown=${HOST_UID}:${HOST_GID} ./containers/scripts/poetry-install.sh /tmp/poetry-install.sh

RUN bash /tmp/poetry-install.sh

COPY --chown=${HOST_UID}:${HOST_GID} ./containers/scripts/tests.sh ${PROJECT_ROOT}/tests.sh
COPY --chown=${HOST_UID}:${HOST_GID} ./containers/scripts/entrypoint.sh /opt/entrypoint.sh

RUN chmod +x /opt/entrypoint.sh \
  && echo "source /opt/entrypoint.sh" > ~/.bashrc

ENTRYPOINT ["tini", "--", "/opt/entrypoint.sh"]
