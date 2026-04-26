from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .records import RunRecord
from .rp import ResearchProblem


def read_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "failed",
            "error_type": "missing_metrics",
            "score": -999.0,
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "failed",
            "error_type": f"bad_metrics_json:{type(exc).__name__}",
            "score": -999.0,
        }
    if not isinstance(data, dict):
        return {
            "status": "failed",
            "error_type": "metrics_json_not_object",
            "score": -999.0,
        }
    return data


def extract_record(
    *,
    rp: ResearchProblem,
    run_id: str,
    parent_id: str | None,
    generation: int,
    artifact_path: str,
    mutation_type: str,
    mutation_summary: str,
    hypothesis: str,
    metrics_path: str | Path | None = None,
) -> RunRecord:
    metrics = read_metrics(Path(metrics_path) if metrics_path else rp.metrics_path(run_id))
    return RunRecord.from_metrics(
        run_id=run_id,
        rp_id=rp.rp_id,
        metrics=metrics,
        parent_id=parent_id,
        generation=generation,
        artifact_path=artifact_path,
        mutation_type=mutation_type,
        mutation_summary=mutation_summary,
        hypothesis=hypothesis,
    )
