.ONESHELL:

include .env

SERVICES:=
SERVICE:=
# options: dev, prod
ENV:=$(ENV)

DOCKER=docker-compose \
	--env-file ../.env \
	--project-name infodengue-$(ENV) \
	--file docker/compose-base.yaml \
	--file docker/compose-$(ENV).yaml

dist:
	echo ${ENV}
# SHELL := /bin/bash

# this line will set the build args from env file
DECONARGS:=$(shell echo "$$(for i in `cat .env`; do out+="--build-arg $$i " ; done; echo $$out;out="")")
# GEN_ARGS:=$(eval BARGS=$(DECONARGS))

# .PHONY: dist-prod
# dist-prod:
#     @echo "Running docker build for PROD ..."
#         $(GEN_ARGS)
#     docker build -f Dockerfile -t $(TAG_SHA) $(BARGS) .
