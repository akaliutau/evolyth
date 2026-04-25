from __future__ import annotations

import io
import math
import time
from typing import Iterable, Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def serialized_model_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(buffer.tell())


@torch.no_grad()
def measure_latency_ms(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 32,
    warmup: int = 5,
    repeats: int = 20,
) -> float:
    was_training = model.training
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device)

    for _ in range(max(0, warmup)):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(max(1, repeats)):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if was_training:
        model.train()
    return float((elapsed / max(1, repeats)) * 1000.0)


def pareto_score(val_accuracy: float, parameter_count: int, model_bytes: int, latency_ms: float) -> float:
    if val_accuracy is None or math.isnan(float(val_accuracy)):
        return -999.0
    return float(
        val_accuracy
        - 0.015 * math.log10(max(parameter_count, 1))
        - 0.010 * math.log10(max(model_bytes, 1))
        - 0.005 * math.log10(max(latency_ms, 1e-3))
    )
