from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from immutable.data import get_dataloaders
from immutable.metrics import (
    accuracy_from_logits,
    count_parameters,
    measure_latency_ms,
    pareto_score,
    serialized_model_bytes,
)
from immutable.print_util import JsonlLogger, write_json, write_text
from immutable.schema import Metrics, failure_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate mutable model.py for Tiny CIFAR-10 Model Golf.")
    p.add_argument("--run-id", default=None, help="Run identifier. Defaults to timestamp.")
    p.add_argument("--data-dir", default="data", help="Dataset directory.")
    p.add_argument("--out-dir", default="runs", help="Output directory for run artifacts.")
    p.add_argument("--dataset", choices=["cifar10", "synthetic"], default="cifar10")
    p.add_argument("--download-cifar", action="store_true", help="Allow torchvision to download CIFAR-10.")
    p.add_argument("--no-synthetic-fallback", action="store_true", help="Fail instead of using synthetic fallback.")
    p.add_argument("--dry-run", action="store_true", help="Very small run for shape/training contract checks.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=200, help="Max optimizer steps per epoch; 0 means no cap.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--train-subset", type=int, default=5000)
    p.add_argument("--val-subset", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--latency-batch-size", type=int, default=32)
    p.add_argument("--latency-repeats", type=int, default=20)
    return p.parse_args()


def normalize_args_for_dry_run(args: argparse.Namespace) -> None:
    if args.dry_run:
        args.dataset = "synthetic" if args.dataset == "cifar10" else args.dataset
        args.epochs = min(args.epochs, 1)
        args.max_steps = min(args.max_steps, 3) if args.max_steps else 3
        args.batch_size = min(args.batch_size, 32)
        args.train_subset = min(args.train_subset, 128)
        args.val_subset = min(args.val_subset, 64)
        args.latency_repeats = min(args.latency_repeats, 5)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def import_mutable_model(root: Path):
    model_path = root / "model.py"
    spec = importlib.util.spec_from_file_location("mutable_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mutable_model"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "Model"):
        raise AttributeError("model.py must define class Model(nn.Module)")
    return module


def check_contract(model: nn.Module, device: torch.device) -> None:
    model.eval()
    x = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        y = model(x)
    if tuple(y.shape) != (4, 10):
        raise ValueError(f"Model output must have shape [B, 10]; got {tuple(y.shape)}")
    if not torch.isfinite(y).all():
        raise ValueError("Model output contains NaN or Inf")


def train_one_epoch(
    *,
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_steps: int,
    logger: JsonlLogger,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    steps = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if not torch.isfinite(loss):
            raise ValueError("training loss became NaN or Inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        n = int(xb.shape[0])
        total_loss += float(loss.item()) * n
        total_count += n
        steps += 1
        if steps % 25 == 0:
            logger.log("train_step", epoch=epoch, step=steps, loss=float(loss.item()))
        if max_steps and steps >= max_steps:
            break
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(*, model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        n = int(xb.shape[0])
        total_loss += float(loss.item()) * n
        total_correct += float((logits.argmax(dim=1) == yb).sum().item())
        total_count += n
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def make_summary(metrics: Dict[str, Any]) -> str:
    lines = [
        f"# Run {metrics.get('run_id')}",
        "",
        f"- status: `{metrics.get('status')}`",
        f"- idea: {metrics.get('idea')}",
        f"- dataset_mode: `{metrics.get('dataset_mode')}`",
        f"- score: `{metrics.get('score')}`",
        f"- val_accuracy: `{metrics.get('val_accuracy')}`",
        f"- val_loss: `{metrics.get('val_loss')}`",
        f"- train_loss: `{metrics.get('train_loss')}`",
        f"- parameter_count: `{metrics.get('parameter_count')}`",
        f"- model_bytes: `{metrics.get('model_bytes')}`",
        f"- latency_ms: `{metrics.get('latency_ms')}`",
        f"- train_seconds: `{metrics.get('train_seconds')}`",
        f"- dry_run_passed: `{metrics.get('dry_run_passed')}`",
    ]
    if metrics.get("error_type"):
        lines.extend(["", "## Error", "", f"{metrics.get('error_type')}: {metrics.get('error_message')}"])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    normalize_args_for_dry_run(args)
    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    args.run_id = run_id
    root = Path(__file__).resolve().parent
    run_dir = root / args.out_dir / run_id
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(run_dir / "events.jsonl")

    start_time = time.perf_counter()
    dataset_mode = "unknown"
    device_str = "unknown"

    try:
        logger.log("run_start", run_id=run_id, args=vars(args))
        set_seed(args.seed)
        device = choose_device(args.device)
        device_str = str(device)
        logger.log("device_selected", device=device_str)

        model_module = import_mutable_model(root)
        model = model_module.Model(num_classes=10)
        if not isinstance(model, nn.Module):
            raise TypeError("Model() must return an nn.Module")
        idea = getattr(model_module, "describe_model", lambda: model.__class__.__name__)()
        model = model.to(device)
        check_contract(model, device)
        logger.log("contract_passed", idea=str(idea))

        data = get_dataloaders(
            dataset=args.dataset,
            data_dir=root / args.data_dir,
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            seed=args.seed,
            num_workers=args.num_workers,
            download_cifar=args.download_cifar,
            allow_synthetic_fallback=not args.no_synthetic_fallback,
        )
        dataset_mode = data.dataset_mode
        logger.log(
            "data_ready",
            dataset=args.dataset,
            dataset_mode=dataset_mode,
            train_size=data.train_size,
            val_size=data.val_size,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        last_train_loss = 0.0
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            last_train_loss = train_one_epoch(
                model=model,
                loader=data.train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                max_steps=args.max_steps,
                logger=logger,
                epoch=epoch,
            )
            val_loss, val_acc = evaluate(model=model, loader=data.val_loader, criterion=criterion, device=device)
            logger.log(
                "epoch_end",
                epoch=epoch,
                train_loss=last_train_loss,
                val_loss=val_loss,
                val_accuracy=val_acc,
                epoch_seconds=time.perf_counter() - epoch_start,
            )

        param_count = count_parameters(model)
        model_bytes = serialized_model_bytes(model)
        latency_ms = measure_latency_ms(
            model,
            device=device,
            batch_size=args.latency_batch_size,
            repeats=args.latency_repeats,
        )
        score = pareto_score(val_acc, param_count, model_bytes, latency_ms)
        train_seconds = time.perf_counter() - start_time

        torch.save(model.state_dict(), artifacts_dir / "model.pt")
        metrics = Metrics(
            status="success",
            run_id=run_id,
            dataset=args.dataset,
            dataset_mode=dataset_mode,
            idea=str(idea),
            score=score,
            val_accuracy=float(val_acc),
            val_loss=float(val_loss),
            train_loss=float(last_train_loss),
            parameter_count=param_count,
            model_bytes=model_bytes,
            latency_ms=latency_ms,
            train_seconds=train_seconds,
            dry_run_passed=bool(args.dry_run),
            device=device_str,
            epochs=args.epochs,
            train_subset=data.train_size,
            val_subset=data.val_size,
        ).to_dict()
        write_json(run_dir / "metrics.json", metrics)
        write_text(run_dir / "run_summary.md", make_summary(metrics))
        logger.log("run_complete", **metrics)
        return metrics

    except Exception as exc:
        train_seconds = time.perf_counter() - start_time
        metrics = failure_metrics(
            run_id=run_id,
            dataset=args.dataset,
            dataset_mode=dataset_mode,
            device=device_str,
            epochs=args.epochs,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            error_type=type(exc).__name__,
            error_message=f"{exc}\n\n{traceback.format_exc()}",
            train_seconds=train_seconds,
        )
        write_json(run_dir / "metrics.json", metrics)
        write_text(run_dir / "run_summary.md", make_summary(metrics))
        logger.log("run_failed", **metrics)
        return metrics


def main() -> None:
    args = parse_args()
    _ = run(args)
    # Always exit 0: model failures are data for the evolver.
    sys.exit(0)


if __name__ == "__main__":
    main()
