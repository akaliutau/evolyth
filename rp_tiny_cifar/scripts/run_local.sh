#!/usr/bin/env bash
set -euo pipefail

python train_eval.py --dry-run --dataset synthetic --run-id smoke
python train_eval.py \
  --dataset cifar10 \
  --data-dir ./data \
  --download-cifar \
  --run-id cifar_baseline \
  --epochs 2 \
  --train-subset 5000 \
  --val-subset 1000 \
  --max-steps 200
