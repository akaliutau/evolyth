from __future__ import annotations

from pathlib import Path
from typing import Any

from .store import EvolutionStore
from .rp import ResearchProblem


def build_context_packet(store: EvolutionStore, rp: ResearchProblem, parent_id: str | None = None) -> str:
    parent = store.get(parent_id) if parent_id else _best_parent(store)
    top = store.leaderboard(3)
    front = store.pareto_front()[:5]

    parts = [
        "# Evolution Context",
        "",
        "## Research Problem",
        rp.goal_prompt.strip(),
        "",
        "## RP Contract",
        f"- Entry point folder: `{rp.path}`",
        f"- Mutable file: `{rp.mutable_file}` only",
        f"- Train script: `{rp.train_script}`",
        "- Preserve model input/output contract.",
        "- Run must produce metrics.json/events.jsonl and treat failures as data.",
        "",
    ]

    if parent:
        parts += [
            "## Parent Run",
            _run_line(parent),
            "",
            "### Parent model.py",
            "```python",
            _read(parent.get("model_path") or str(rp.model_path)),
            "```",
            "",
        ]

    parts += [
        "## Top Leaderboard",
        *_lines(top),
        "",
        "## Pareto Front",
        *_lines(front),
        "",
        "## Mutation Request",
        "Make exactly one bounded architecture mutation in model.py. Keep it simple, reliable, and easy to modify.",
        "Before finalizing, ensure the dry-run command still works.",
    ]
    return "\n".join(parts).strip() + "\n"


def _best_parent(store: EvolutionStore) -> dict[str, Any] | None:
    rows = store.leaderboard(1)
    return rows[0] if rows else None


def _lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none yet"]
    return [f"- {_run_line(r)}" for r in rows]


def _run_line(r: dict[str, Any]) -> str:
    return (
        f"{r.get('run_id')} status={r.get('status')} score={_fmt(r.get('score'))} "
        f"acc={_fmt(r.get('val_accuracy'))} params={r.get('parameter_count')} "
        f"bytes={r.get('model_bytes')} latency_ms={_fmt(r.get('latency_ms'))} "
        f"idea={r.get('mutation_summary') or r.get('mutation_type')}"
    )


def _fmt(x: Any) -> str:
    try:
        return f"{float(x):.4g}"
    except (TypeError, ValueError):
        return "?"


def _read(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else "# model snapshot unavailable"
