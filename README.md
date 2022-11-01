# Wave-docker-template
Template repository to create a dockerized H2O Wave application from scratch.

If you want to create a Wave application, select this repo as a templated when creating your app's repository.

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
$ docker run -p 8501:8501 wave-app:latest
```

You can open the app at http://localhost:10101/monitor


### Further developing your application
We recommend developing the app using the virtualenv instructions above. This way you can add dependencies like this:

```bash
# Adding pandas as a dependency
$ poetry add pandas
```

Then poetry will update your pyproject.toml so that no change to the Dockerfile is necessary.