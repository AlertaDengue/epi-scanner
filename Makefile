SHELL:=/bin/bash

include .env

# options: dev, prod
ENV:=$(ENV)
HOST_UID:=$(HOST_UID)
HOST_GID:=$(HOST_GID)
SERVICE:=
SERVICES:=

DOCKER=docker-compose \
	--env-file .env \
	--project-name infodengue-$(ENV) \
	--file docker/docker-compose.yaml


# PREPARE ENVIRONMENT
.PHONY:prepare-env
prepare-env:
	envsubst < env.tpl > .env

# DOCKER
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

# Run with Dockerfile

# # this line will set the build args from env file
DECONARGS:=$(shell echo "$$(for i in `cat .env`; do out+="--build-arg $$i " ; done; echo $$out;out="")")
GEN_ARGS:=$(eval BARGS=$(DECONARGS))
IMAGETAG:=wave-app:latest


.PHONY:build-dockerfile
build-dockerfile:
	docker build -f docker/Dockerfile -t $(IMAGETAG) $(BARGS) .

.PHONY:run-dockerfile
run-dockerfile:
	@echo "Running docker build..."
	docker run -it --env-file .env $(IMAGETAG) bash -c "wave run app.py"


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
