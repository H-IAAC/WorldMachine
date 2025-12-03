#!/usr/bin/env bash

pip install generate-dockerignore-from-gitignore setuptools_scm

generate-dockerignore

VERSION=$(python3 -m setuptools_scm)
DOCKER_TAG=$(echo "$VERSION" | sed 's/[^a-zA-Z0-9._-]/-/g')

docker build --build-arg "VERSION=${VERSION}" -t "eltoncn/world-machine:${DOCKER_TAG}" .