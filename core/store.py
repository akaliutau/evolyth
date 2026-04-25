from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pyarrow as pa

from .embed import VECTOR_DIM, embed_text
from .jsonl import append_jsonl
from .pareto import leaderboard, pareto_front
from .records import RunRecord


class EvolutionStore:
    """Write-through run registry: JSONL audit log + LanceDB + NetworkX cache."""

    def __init__(self, arena_root: str | Path, table_name: str = "runs"):
        import lancedb

        self.root = Path(arena_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.root / "evolution.jsonl"
        self.db = lancedb.connect(str(self.root / "lancedb"))
        self.table_name = table_name
        self.table = self._open_or_create_table()
        self.graph = nx.DiGraph()
        self._load_graph()

    def register(self, record: RunRecord | dict[str, Any]) -> dict[str, Any]:
        row = record.to_dict() if isinstance(record, RunRecord) else dict(record)
        row.setdefault("vector", embed_text(_search_text(row)))

        append_jsonl(self.audit_path, {"event": "run_registered", **_json_safe(row)})
        self._upsert(row)
        self.graph.add_node(row["run_id"], **row)
        if row.get("parent_id"):
            self.graph.add_edge(row["parent_id"], row["run_id"], edge_type="parent")
        return row

    def all_runs(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.table.to_arrow().to_pylist()]

    def get(self, run_id: str) -> dict[str, Any] | None:
        for row in self.all_runs():
            if row.get("run_id") == run_id:
                return row
        return None

    def leaderboard(self, limit: int = 10) -> list[dict[str, Any]]:
        return leaderboard(self.all_runs(), limit)

    def pareto_front(self) -> list[dict[str, Any]]:
        return pareto_front(self.all_runs())

    def lineage(self, run_id: str) -> list[str]:
        if run_id not in self.graph:
            return []
        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        paths = []
        for root in roots:
            try:
                paths.append(nx.shortest_path(self.graph, root, run_id))
            except nx.NetworkXNoPath:
                pass
        return min(paths, key=len) if paths else [run_id]

    def children(self, run_id: str) -> list[str]:
        return list(self.graph.successors(run_id)) if run_id in self.graph else []

    def search(self, query: str, limit: int = 10, where: str | None = None) -> list[dict[str, Any]]:
        q = self.table.search(embed_text(query))
        if where:
            q = q.where(where)
        return [dict(r) for r in q.limit(limit).to_list()]

    def _open_or_create_table(self):
        if self.table_name in self.db.table_names():
            return self.db.open_table(self.table_name)
        return self.db.create_table(self.table_name, schema=_schema())

    def _upsert(self, row: dict[str, Any]) -> None:
        try:
            (
                self.table.merge_insert("run_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([row])
            )
        except Exception:
            # Compatible fallback for older LanceDB versions.
            self.table.delete(f"run_id = '{_sql(str(row['run_id']))}'")
            self.table.add([row])

    def _load_graph(self) -> None:
        for row in self.all_runs():
            self.graph.add_node(row["run_id"], **row)
            if row.get("parent_id"):
                self.graph.add_edge(row["parent_id"], row["run_id"], edge_type="parent")


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("rp_id", pa.string()),
            pa.field("parent_id", pa.string()),
            pa.field("generation", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("completed_at", pa.string()),
            pa.field("mutation_type", pa.string()),
            pa.field("mutation_summary", pa.string()),
            pa.field("hypothesis", pa.string()),
            pa.field("observation", pa.string()),
            pa.field("next_belief", pa.string()),
            pa.field("idea_tags", pa.string()),
            pa.field("score", pa.float64()),
            pa.field("val_accuracy", pa.float64()),
            pa.field("val_loss", pa.float64()),
            pa.field("parameter_count", pa.int64()),
            pa.field("model_bytes", pa.int64()),
            pa.field("latency_ms", pa.float64()),
            pa.field("train_seconds", pa.float64()),
            pa.field("dry_run_passed", pa.bool_()),
            pa.field("error_type", pa.string()),
            pa.field("artifact_path", pa.string()),
            pa.field("model_path", pa.string()),
            pa.field("metrics_path", pa.string()),
            pa.field("events_path", pa.string()),
            pa.field("summary_path", pa.string()),
            pa.field("stdout_path", pa.string()),
            pa.field("stderr_path", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        ]
    )


def _search_text(row: dict[str, Any]) -> str:
    return "\n".join(str(row.get(k) or "") for k in ["mutation_type", "mutation_summary", "hypothesis", "observation", "next_belief", "idea_tags", "status", "error_type"])


def _sql(value: str) -> str:
    return value.replace("'", "''")


def _json_safe(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "vector"}
