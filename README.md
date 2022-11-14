# Dengue Epi-Scanner App

H2O Wave application that uses up-to-date dengue incidence data from Infodengue, to analyze Dengue's expansion wave in
the south of Brazil.

[![watch the video](https://img.youtube.com/vi/LQmMhVWVJUs/hqdefault.jpg)](https://youtu.be/LQmMhVWVJUs)

### Run with docker-compose :

**Pre-requisites**

* Docker installed and running
* docker-compose installed

```bash
$ docker-compose up

# When dependencies change and you need to force a rebuild
$ docker-compose up --build

# When finished
$ docker-compose down
```

### Run with virtualenv:

**Pre-requisites**

* pip
* poetry

```bash
# Installing dependencies and activating the virtualenv

$ poetry install
$ poetry shell
```

Start the wave server:

```bash
$ cd .venv/
$ ./waved
```

Then on another terminal, start the app,

```bash
$ wave run app.py
```

### Run with docker:

* Docker installed and running

```bash
# First build
$ docker build -t wave-app:latest .

# Subsequent builds
$ docker build --cache-from wave-app:latest -t wave-app:latest .

# To run as docker container with default streamlit port
$ docker run -p 10101:10101 wave-app:latest
```

You can open the app at http://localhost:10101/

#### Running with Docker-compose

Using docker compose makes it a little easier to build and run the app.

```bash
$ docker-compose up --build 
```



