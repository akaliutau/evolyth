from __future__ import annotations

from collections.abc import Iterable
from typing import Any

MAX_KEYS = ("score", "val_accuracy")
MIN_KEYS = ("parameter_count", "model_bytes", "latency_ms")


def leaderboard(records: Iterable[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    rows = [r for r in records if r.get("status") == "succeeded"]
    return sorted(rows, key=lambda r: float(r.get("score") or -999.0), reverse=True)[:limit]


def pareto_front(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [r for r in records if r.get("status") == "succeeded"]
    front: list[dict[str, Any]] = []
    for candidate in rows:
        if not any(_dominates(other, candidate) for other in rows if other is not candidate):
            front.append(candidate)
    return sorted(front, key=lambda r: float(r.get("score") or -999.0), reverse=True)


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    better_or_equal = True
    strictly_better = False

    for key in MAX_KEYS:
        av = _num(a.get(key), -999.0)
        bv = _num(b.get(key), -999.0)
        better_or_equal = better_or_equal and av >= bv
        strictly_better = strictly_better or av > bv

    for key in MIN_KEYS:
        av = _num(a.get(key), float("inf"))
        bv = _num(b.get(key), float("inf"))
        better_or_equal = better_or_equal and av <= bv
        strictly_better = strictly_better or av < bv

    return better_or_equal and strictly_better


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
