#!/bin/bash
# This file checks if Docker is installed
# If so build the image an run it in a container.

if ! command -v docker > /dev/null 2>&1
then
    echo "Docker is not installed on your system"
    exit 1
fi

if docker info > /dev/null 2>&1; then
    DOCKER=(docker)
elif sudo docker info > /dev/null 2>&1; then
    echo "Docker requires sudo"
    DOCKER=(sudo docker)
else
    echo "Cannot connect to the Docker daemon. Is it running?" >&2
    exit 1
fi

"${DOCKER[@]}" build -t paramux .
"${DOCKER[@]}" run --rm -it \
    -v "$(pwd)/example_finder":/lattice-estimator/example_finder \
    paramux:latest
