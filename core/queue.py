from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ids import utc_now


@dataclass
class QueueItem:
    item_id: str
    source_run_id: str
    mutation_type: str
    proposed_mutation: str
    priority: float = 0.0
    status: str = "queued"
    created_at: str = field(default_factory=utc_now)


class MutationQueue:
    """Tiny durable priority queue backed by one JSON file."""

    def __init__(self, arena_root: str | Path):
        self.path = Path(arena_root).expanduser().resolve() / "queue.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def enqueue(self, item: QueueItem) -> None:
        rows = self._read()
        rows.append(asdict(item))
        self._write(rows)

    def list(self, status: str | None = None) -> list[dict]:
        rows = self._read()
        if status:
            rows = [r for r in rows if r.get("status") == status]
        return sorted(rows, key=lambda r: float(r.get("priority") or 0.0), reverse=True)

    def pop_next(self) -> dict | None:
        rows = self._read()
        queued = [r for r in rows if r.get("status") == "queued"]
        if not queued:
            return None
        item = max(queued, key=lambda r: float(r.get("priority") or 0.0))
        for r in rows:
            if r["item_id"] == item["item_id"]:
                r["status"] = "running"
        self._write(rows)
        return item

    def complete(self, item_id: str) -> None:
        rows = self._read()
        for r in rows:
            if r["item_id"] == item_id:
                r["status"] = "done"
        self._write(rows)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, rows: list[dict]) -> None:
        self.path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
