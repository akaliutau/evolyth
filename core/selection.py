from __future__ import annotations

import math
from typing import Any

from .store import EvolutionStore


def select_parent(store: EvolutionStore) -> dict[str, Any] | None:
    rows = store.all_runs()
    if not rows:
        return None
    pareto_ids = {r["run_id"] for r in store.pareto_front()}
    children_count = {r["run_id"]: len(store.children(r["run_id"])) for r in rows}
    scored = [(parent_priority(r, r["run_id"] in pareto_ids, children_count[r["run_id"]]), r) for r in rows]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def parent_priority(row: dict[str, Any], is_pareto: bool, num_children: int) -> float:
    score = _num(row.get("score"), -999.0)
    status_bonus = 1.0 if row.get("status") == "succeeded" else -1.0
    pareto_bonus = 1.0 if is_pareto else 0.0
    explore = 1.0 / math.sqrt(1 + num_children)
    generation_penalty = 0.01 * _num(row.get("generation"), 0.0)
    return 0.55 * score + 0.25 * pareto_bonus + 0.15 * explore + 0.10 * status_bonus - generation_penalty


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
