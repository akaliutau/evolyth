# Tiny CIFAR-10 Model Golf RP Bundle

This is a self-contained research-problem bundle for an autonomous evolver demo.

The evolver may edit exactly one file:

```text
model.py
```

Everything else is the stable game: dataset loading, training, scoring, metrics extraction, run logging, and failure handling.

## What this demonstrates

The demo turns architecture search into a stable, comparable loop:

```text
model.py mutation -> dry-run -> train/eval -> JSONL metrics -> score -> leaderboard/Pareto choice
```

Failures are valid run outcomes. `train_eval.py` catches exceptions, writes `metrics.json`, and exits with code 0 so the outer evolutionary runner never has to guess whether a failed model counts.

## Folder layout

```text
rp_tiny_cifar/
  goal_prompt.md
  Dockerfile
  requirements.txt
  train_eval.py
  model.py                         # only mutable file
  immutable/
    data.py
    metrics.py
    print_util.py
    schema.py
  scripts/
    prepare_cifar.py
    run_local.sh
    build_and_run_docker.sh
  data/
    README.md
  runs/
```

## Quick local smoke test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_eval.py --dry-run --dataset synthetic --run-id smoke
```

Expected outputs:

```text
runs/smoke/events.jsonl
runs/smoke/metrics.json
runs/smoke/run_summary.md
runs/smoke/artifacts/model.pt
```

## Run on actual CIFAR-10

The first run can download CIFAR-10 through torchvision:

```bash
python scripts/prepare_cifar.py --data-dir ./data
python train_eval.py \
  --dataset cifar10 \
  --data-dir ./data \
  --run-id cifar_baseline \
  --epochs 2 \
  --train-subset 5000 \
  --val-subset 1000 \
  --max-steps 200
```

For no-network demo environments, use the deterministic synthetic fallback:

```bash
python train_eval.py --dry-run --dataset synthetic --run-id offline_smoke
```

## Docker

Build and run the smoke test:

```bash
./scripts/build_and_run_docker.sh
```

Run CIFAR-10 with data mounted from the host:

```bash
docker build -t tiny-cifar .
docker run --rm -v "$PWD/data:/rp/data" -v "$PWD/runs:/rp/runs" tiny-cifar \
  python train_eval.py --dataset cifar10 --data-dir /rp/data --download-cifar --epochs 2 --run-id docker_cifar
```

## Evolver contract

The autonomous evolver receives:

- `goal_prompt.md`
- parent `model.py`
- recent `runs/*/metrics.json`
- optional data-scientist feedback

It may only edit `model.py`. The model contract is:

```python
class Model(nn.Module):
    def __init__(self, num_classes: int = 10): ...
    def forward(self, x): ...  # x: [B, 3, 32, 32], logits: [B, 10]

def describe_model() -> str: ...  # optional
```

## Score

```text
score = val_accuracy
        - 0.015 * log10(parameter_count)
        - 0.010 * log10(model_bytes)
        - 0.005 * log10(latency_ms)
```

This rewards the Pareto tradeoff: better accuracy, fewer parameters, smaller artifact, and lower latency.

