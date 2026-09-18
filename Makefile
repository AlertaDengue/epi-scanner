-include .env

ENV ?= prod

COMPOSE_FILE = $(if $(filter dev,$(ENV)),docker-compose.dev.yaml,docker-compose.yaml)

.PHONY: install dev build start lint clean image up down logs

install:
	npm install

dev:
	npm run dev

build:
	npm run build

start:
	npm run start

lint:
	npm run lint

clean:
	rm -rf .next out node_modules

image:
	docker compose -f $(COMPOSE_FILE) build

up:
	docker compose -f $(COMPOSE_FILE) up -d

down:
	docker compose -f $(COMPOSE_FILE) down

logs:
	docker compose -f $(COMPOSE_FILE) logs -f