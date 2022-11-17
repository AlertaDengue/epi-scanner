FROM condaforge/mambaforge

ARG HOST_UID
ARG HOST_GID
ARG SERVICES
ARG ENV
ENV ENV_NAME=episcanner
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq update --yes \
  && apt-get -qq install --yes --no-install-recommends \
  build-essential \
  ca-certificates sudo \
  && rm -rf /var/lib/apt/lists/*

# Create epiuser user
RUN addgroup --gid ${HOST_GID} epiuser \
  && useradd --uid ${HOST_UID} --gid ${HOST_GID} -ms /bin/bash epiuser \
  && echo "epiuser ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/epiuser \
  && chmod 0440 /etc/sudoers.d/epiuser \
  && export ENV_NAME="$ENV_NAME" \
  && mkdir -p /opt/conda /opt/services/logs /opt/services/epi_scanner \
  && chmod -R a+rwx /opt/conda /opt/services/epi_scanner/ /tmp \
  && chown epiuser:epiuser /opt/services/ /tmp \
  && echo 'source /opt/conda/bin/activate "$ENV_NAME" && exec "$@"' > /activate.sh \
  && echo 'source activate "$ENV_NAME"' >  /home/epiuser/.bashrc \
  && chmod +x /activate.sh

ENV PATH "$PATH:/home/epiuser/.local/bin"
ENV PATH /opt/conda/envs/$ENV_NAME/bin:$PATH

USER epiuser

## Copy code to the container with permissions to user
COPY --chown=epiuser:epiuser pyproject.toml poetry.lock  /opt/services/
COPY --chown=epiuser:epiuser conda/dev.yaml /tmp/environment.yaml
COPY --chown=epiuser:epiuser docker/scripts/poetry-install.sh /tmp/poetry-install.sh

COPY --chown=epiuser:epiuser ./epi_scanner /opt/services/epi_scanner

# Change shell to bash
SHELL ["/bin/bash", "-c"]

# Use environment to update the env episcanner
RUN mamba env create -n episcanner --file /tmp/environment.yaml --force \
  && cd /opt/services/ \
  && . /tmp/poetry-install.sh \
  && conda clean --all \
  && find /opt/conda/ -type f,l -name '*.a' -delete \
  && find /opt/conda/ -type f,l -name '*.pyc' -delete \
  && find /opt/conda/ -type f,l -name '*.js.map' -delete \
  && rm -rf /opt/conda/pkgs

WORKDIR /opt/services/epi_scanner

ENTRYPOINT ["bash", "/activate.sh"]
