# options: dev, prod
ENV:=$(ENV)
HOST_UID:=$(HOST_UID)
HOST_GID:=$(HOST_GID)
SERVICE:=
SERVICES:=
CONSOLE:=bash
CMD:=
ARGS:=
TIMEOUT:=60
STATE_ABBV:=
DISEASE:=


#  APP ON CI
# .PHONY: app-wait
# app-wait:
# 	timeout ${TIMEOUT} ./scripts/ci/status_code_check.sh


# https://github.com/containers/podman-compose/issues/491#issuecomment-1289944841
CONTAINER_APP=docker-compose \
	--env-file=.env \
	--project-name episcanner \
	--file containers/docker-compose.yaml


# CONTAINER_APP

.ONESHELL:
.PHONY:containers-pull
containers-pull:
	set -e
	$(CONTAINER_APP) pull ${SERVICES}

.ONESHELL:
.PHONY:containers-build
containers-build: containers-pull
	set -e
	$(CONTAINER_APP) build ${SERVICES}

.PHONY:containers-start
containers-start:
	set -ex
	$(CONTAINER_APP) up --remove-orphans -d ${SERVICES}

.PHONY:containers-stop
containers-stop:
	set -ex
	$(CONTAINER_APP) stop ${SERVICES}

.PHONY:containers-restart
containers-restart: containers-stop containers-start

.PHONY:containers-logs
containers-logs:
	$(CONTAINER_APP) logs ${ARGS} ${SERVICES}

.PHONY:containers-logs-follow
containers-logs-follow:
	$(CONTAINER_APP) logs --follow ${ARGS} ${SERVICES}

.PHONY: containers-wait
containers-wait:
	timeout ${TIMEOUT} ./containers/scripts/healthcheck.sh ${SERVICE}

.PHONY:containers-exec
containers-exec:
	set -e
	$(CONTAINER_APP) exec ${ARGS} ${SERVICE} ${CMD}

.PHONY:containers-console
containers-console:
	set -e
	$(MAKE) containers-exec ARGS="${ARGS}" SERVICE=${SERVICE} CMD="${CONSOLE}"

.PHONY:containers-run-console
containers-run-console:
	$(CONTAINER_APP) run --rm ${ARGS} ${SERVICE} ${CONSOLE}

.PHONY:containers-down
containers-down:
	$(CONTAINER_APP) down --volumes --remove-orphans

# https://github.com/containers/podman/issues/5114#issuecomment-779406347
.PHONY:containers-reset-storage
containers-reset-storage:
	rm -rf ~/.local/share/containers/

.PHONY:create-dotenv
create-dotenv:
	touch .env
	echo -n "HOST_UID=`id -u`\nHOST_GID=`id -g`" > .env

# Fetches data for a specified state and disease and exports it to the data directory.
.PHONY:fetch-data
fetch-data:
	python epi_scanner/management/get_parameters.py ${STATE_ABBV} ${DISEASE}

# Export all fetched data to parquet files for all states.
.PHONY:fetch-all-data
fetch-all-data:
	python epi_scanner/management/export_data.py

#  Run pytest for all tests in epi_scanner/tests inside the container
.PHONY:test-fetch-data
test-fetch-data:
	$(CONTAINER_APP) exec -T wave-app /bin/bash -c 'pytest -vv epi_scanner/tests'

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
