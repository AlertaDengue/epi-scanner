SHELL:=/bin/bash

include .env

# options: dev, prod
ENV:=$(ENV)
HOST_UID:=$(HOST_UID)
HOST_GID:=$(HOST_GID)
SERVICE:=
SERVICES:=


# PREPARE ENVIRONMENT
.PHONY:prepare-env
prepare-env:
	envsubst < env.tpl > .env

# -----------

# LINTING CODE
.PHONY: lint
lint: ## formatting linter with poetry
	pre-commit install
	pre-commit run --all-files

# -----------

# DOCKER
DOCKER=docker-compose \
	--env-file .env \
	--project-name infodengue-$(ENV) \
	--file docker/docker-compose.yaml


.PHONY:docker-build
docker-build:
	$(DOCKER) build ${SERVICES}

.PHONY:docker-start
docker-start:
	$(DOCKER) up -d ${SERVICES}

.PHONY:docker-logs-follow
docker-logs-follow:
	$(DOCKER) logs --follow --tail 300 ${SERVICES}

.PHONY:docker-stop
docker-stop:
	$(DOCKER) down -v --remove-orphans

.PHONY: docker-wait
docker-wait:
	ENV=${ENV} timeout 90 ./docker/scripts/healthcheck.sh ${SERVICE}

.PHONY: docker-wait-all
docker-wait-all:
	$(MAKE) docker-wait ENV=${ENV} SERVICE="wave-app"

# -----------

# Python
.PHONY: clean
clean: ## clean all artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr .idea/
	rm -fr */.eggs
	rm -fr db
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '*.ipynb_checkpoints' -exec rm -rf {} +
	find . -name '*.pytest_cache' -exec rm -rf {} +
