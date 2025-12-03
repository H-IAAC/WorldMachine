#!/usr/bin/env bash

pip install generate-dockerignore-from-gitignore

generate-dockerignore

VERSION=$(git describe --tags --dirty --always)

docker build --build-arg "VERSION=${VERSION}" -t "eltoncn/world-machine:${VERSION}" .