SHELL:=/bin/bash

include .env

# options: dev, prod
ENV:=$(ENV)
HOST_UID:=$(HOST_UID)
HOST_GID:=$(HOST_GID)
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
