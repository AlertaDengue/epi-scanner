# Dengue Epi-Scanner App

H2O Wave application that uses up-to-date dengue incidence data from Infodengue, to analyze Dengue's expansion wave in
the south of Brazil.

[![watch the video](https://img.youtube.com/vi/LQmMhVWVJUs/hqdefault.jpg)](https://youtu.be/LQmMhVWVJUs)

### Run with docker-compose :

**Pre-requisites**

* Docker installed and running
* docker-compose installed
* [Miniforge](https://github.com/conda-forge/miniforge)) installed

### Run with miniconda:

**Pre-requisites**

 ([Download

 ```bash
 # Installing dependencies and activating the conda environment
$ mamba env create -f conda/dev.yaml
$ conda activate episcanner 
$ poetry install
```

```bash
# Then in the terminal, start the app.
$ cd app/
$ wave run app.py
```

### Run with docker:

* Docker installed and running

```bash
# First build
docker build -f docker/Dockerfile -t wave-app:latest $(for i in `cat .env`; do out+="--build-arg $i " ; done; echo $out;out="") .

# To run as docker container with default streamlit port
$ docker run -it --env-file .env wave-app:latest bash -c "wave run app.py"
```

You can open the app at http://localhost:10101/

#### Running with Docker-compose

Using docker compose makes it a little easier to build and run the app.


```bash
$ docker-compose -f docker/docker-compose.yml --env-file .env up

# When dependencies change and you need to force a rebuild
$ docker-compose -f docker/docker-compose.yml --env-file .env up --build

# When finished
$ docker-compose -f docker/docker-compose.yml --env-file .env down
```
