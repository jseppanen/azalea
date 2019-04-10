#!/bin/bash -ex

cd $(dirname $0)/..

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
GIT_COMMIT=$(git rev-parse HEAD)
IMAGE_VERSION=${GIT_BRANCH}_${GIT_COMMIT:0:7}

rm -rf dist
python setup.py sdist
PACKAGE=$(ls -1 dist|grep azalea-.*\.tar\.gz)

cp dist/$PACKAGE docker
cp environment.yml docker
docker build \
    --build-arg PACKAGE=$PACKAGE \
    -t azalea:$IMAGE_VERSION \
    docker
rm -f docker/$PACKAGE docker/environment.yml
