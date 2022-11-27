# Dengue Epi-Scanner App

H2O Wave application that uses up-to-date dengue incidence data from Infodengue, to analyze Dengue's expansion wave in
the south of Brazil.

[![watch the video](https://img.youtube.com/vi/LQmMhVWVJUs/hqdefault.jpg)](https://youtu.be/LQmMhVWVJUs)

### Run with miniforge environment:

**Pre-requisites**
* [Miniforge](https://github.com/conda-forge/miniforge) installed

 ```bash
 # Installing dependencies and activating the conda environment
$ mamba env create -f conda/dev.yaml
$ conda activate episcanner 
$ poetry install
```

```bash
# Then in the terminal, start the app
$ wave run --no-reload --no-autostart epi_scanner.app
```

### Running with docker-compose

**Pre-requisites**
* Docker installed and running
* docker-compose installed
* [Miniforge](https://github.com/conda-forge/miniforge) installed

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

### To update data for states or municipalities

```
# Run:

$ python scripts/create_data_parquet.py RJ 3304557 dengue
> epi_scanner/data/RJ_3304557_dengue.parquet

$ python scripts/create_data_parquet.py RJ dengue
> epi_scanner/data/RJ_dengue.parquet
```
