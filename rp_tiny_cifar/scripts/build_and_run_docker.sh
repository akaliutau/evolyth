#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="tiny-cifar-model-golf"
docker build -t "$IMAGE_NAME" .
docker run --rm \
  -v "$PWD/runs:/rp/runs" \
  "$IMAGE_NAME"
