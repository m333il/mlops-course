#!/bin/bash
set -e

GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

docker build \
    -f Dockerfile.serve \
    --build-arg MODEL_VERSION="${GIT_COMMIT}" \
    -t fashion-serve:v1 \
    -t fashion-serve:${GIT_COMMIT} \
    .
