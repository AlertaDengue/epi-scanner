.PHONY: install dev build start lint clean docker-build docker-up docker-down

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

docker-build:
	docker build -t epi-scanner .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f
