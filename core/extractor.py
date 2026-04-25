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
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "status": "failed",
            "error_type": f"bad_metrics_json:{type(e).__name__}",
            "score": -999.0,
        }


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
) -> RunRecord:
    metrics = read_metrics(rp.metrics_path(run_id))
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
