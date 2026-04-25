from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class Metrics:
    status: str
    run_id: str
    dataset: str
    dataset_mode: str
    idea: str
    score: float
    val_accuracy: float
    val_loss: float
    train_loss: float
    parameter_count: int
    model_bytes: int
    latency_ms: float
    train_seconds: float
    dry_run_passed: bool
    device: str
    epochs: int
    train_subset: int
    val_subset: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def failure_metrics(
    *,
    run_id: str,
    dataset: str,
    dataset_mode: str,
    device: str,
    epochs: int,
    train_subset: int,
    val_subset: int,
    error_type: str,
    error_message: str,
    train_seconds: float = 0.0,
) -> Dict[str, Any]:
    return Metrics(
        status="failed",
        run_id=run_id,
        dataset=dataset,
        dataset_mode=dataset_mode,
        idea="unavailable",
        score=-999.0,
        val_accuracy=0.0,
        val_loss=0.0,
        train_loss=0.0,
        parameter_count=0,
        model_bytes=0,
        latency_ms=0.0,
        train_seconds=train_seconds,
        dry_run_passed=False,
        device=device,
        epochs=epochs,
        train_subset=train_subset,
        val_subset=val_subset,
        error_type=error_type,
        error_message=error_message[:2000],
    ).to_dict()
