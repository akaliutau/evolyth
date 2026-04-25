# Research Problem: Tiny CIFAR-10 Model Golf

You are evolving `model.py`.

## Goal

Create a PyTorch model that achieves the best validation accuracy on the provided CIFAR-10 task while minimizing model size and latency.

## You may edit

- `model.py` only

## You must not edit

- `train_eval.py`
- `immutable/*`
- `Dockerfile`
- `requirements.txt`
- `scripts/*`

## Required model contract

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        ...

    def forward(self, x):
        # x shape: [B, 3, 32, 32]
        # return logits shape: [B, 10]
        ...

def describe_model() -> str:
    return "short human-readable architecture idea"
```

## Hard constraints

- Must run on CPU and CUDA.
- Must accept input shape `[B, 3, 32, 32]`.
- Must output logits shape `[B, 10]`.
- Must train in dry-run mode.
- Must not download external data from `model.py`.
- Must not use pretrained weights.
- Must not write outside the run directory.
- Must not edit immutable files.

## Good mutation families

- Tiny CNN baseline
- Depthwise separable CNN
- Residual bottleneck CNN
- ConvMixer-lite
- Patch MLP mixer
- Squeeze-excitation lite
- Low-rank classifier head
- Width/depth tradeoffs
- Stronger normalization or activation choices

## Avoid

- Huge fully connected heads
- External pretrained models
- Internet access
- Multi-file rewrites
- Long warmup schedules
- Custom CUDA ops
- Anything that prevents a clean dry run

## Scoring intuition

Improve the Pareto score:

```text
score = val_accuracy
        - 0.015 * log10(parameter_count)
        - 0.010 * log10(model_bytes)
        - 0.005 * log10(latency_ms)
```

Before finalizing a mutation, run:

```bash
python train_eval.py --dry-run --dataset synthetic
```

For a real evaluation, run:

```bash
python train_eval.py --dataset cifar10 --epochs 2 --train-subset 5000 --val-subset 1000 --max-steps 200
```
