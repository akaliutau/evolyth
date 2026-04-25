from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .embed import embed_text
from .ids import utc_now


@dataclass
class RunRecord:
    run_id: str
    rp_id: str
    parent_id: str | None = None
    generation: int = 0
    status: str = "created"
    created_at: str = field(default_factory=utc_now)
    completed_at: str | None = None

    mutation_type: str = "baseline"
    mutation_summary: str = ""
    hypothesis: str = ""
    observation: str = ""
    next_belief: str = ""
    idea_tags: str = ""

    score: float = -999.0
    val_accuracy: float | None = None
    val_loss: float | None = None
    parameter_count: int | None = None
    model_bytes: int | None = None
    latency_ms: float | None = None
    train_seconds: float | None = None
    dry_run_passed: bool | None = None
    error_type: str | None = None

    artifact_path: str = ""
    model_path: str = ""
    metrics_path: str = ""
    events_path: str = ""
    summary_path: str = ""
    stdout_path: str = ""
    stderr_path: str = ""

    vector: list[float] = field(default_factory=list)

    @classmethod
    def from_metrics(
        cls,
        *,
        run_id: str,
        rp_id: str,
        metrics: dict[str, Any],
        parent_id: str | None = None,
        generation: int = 0,
        artifact_path: str = "",
        mutation_type: str = "baseline",
        mutation_summary: str = "",
        hypothesis: str = "",
    ) -> "RunRecord":
        summary = mutation_summary or str(metrics.get("model_description") or metrics.get("description") or "")
        record = cls(
            run_id=run_id,
            rp_id=rp_id,
            parent_id=parent_id,
            generation=generation,
            status=str(metrics.get("status", "unknown")),
            completed_at=utc_now(),
            mutation_type=mutation_type,
            mutation_summary=summary,
            hypothesis=hypothesis,
            observation=str(metrics.get("observation", "")),
            score=_float(metrics.get("score"), -999.0),
            val_accuracy=_float_or_none(metrics.get("val_accuracy", metrics.get("accuracy"))),
            val_loss=_float_or_none(metrics.get("val_loss")),
            parameter_count=_int_or_none(metrics.get("parameter_count", metrics.get("params"))),
            model_bytes=_int_or_none(metrics.get("model_bytes")),
            latency_ms=_float_or_none(metrics.get("latency_ms")),
            train_seconds=_float_or_none(metrics.get("train_seconds")),
            dry_run_passed=_bool_or_none(metrics.get("dry_run_passed")),
            error_type=_none_if_empty(metrics.get("error_type")),
            artifact_path=artifact_path,
        )
        record.vector = embed_text(record.search_text())
        return record

    def search_text(self) -> str:
        return "\n".join(
            x
            for x in [
                self.mutation_type,
                self.mutation_summary,
                self.hypothesis,
                self.observation,
                self.next_belief,
                self.idea_tags,
                self.status,
                self.error_type or "",
            ]
            if x
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d.get("vector"):
            d["vector"] = embed_text(self.search_text())
        return d


def _none_if_empty(x: Any) -> str | None:
    return None if x in (None, "", "None") else str(x)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "passed"}
