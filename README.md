# Dengue Epi-Scanner App

H2O Wave application that uses up-to-date dengue incidence data from Infodengue, to analyze Dengue's expansion wave in
the south of Brazil.

[![watch the video](https://img.youtube.com/vi/LQmMhVWVJUs/hqdefault.jpg)](https://youtu.be/LQmMhVWVJUs)

### Run with miniconda environment:

**Pre-requisites**
* [Miniforge](https://github.com/conda-forge/miniforge) installed

 ```bash
 # Installing dependencies and activating the conda environment
$ mamba env create -f conda/dev.yaml
$ conda activate episcanner 
$ poetry install
```

```bash
# Then in the terminal, start the app.
$ cd epi_scanner/
$ wave run app.py
```

### Run with Docker:

**Pre-requisites**

* Docker installed and running
* docker-compose installed
* [Miniforge](https://github.com/conda-forge/miniforge) installed


#### Run with Dockerfile:

```bash
# First build
$ make build-dockerfile

# To run as docker container with default streamlit port
$ make run-dockerfile

# To kill use ctrl+c
```

#### Running with docker-compose

Using docker compose makes it a little easier to build and run the app.


```bash
# Build image
$ make docker-build

# Start container
$ make docker-start

# Stop and remove network and all containers
$ make docker-stop
```

*You can open the app at http://localhost:10101/*
