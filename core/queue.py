from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

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
    fingerprint: str = ""


class MutationQueue:
    """Tiny durable priority queue backed by one JSON file.

    Queue writes are locked and atomic. Exact duplicate mutation suggestions for
    the same source run are ignored so an over-eager reviewer cannot flood the
    queue with repeated work.
    """

    ACTIVE_STATUSES = {"queued", "running"}

    def __init__(self, arena_root: str | Path):
        self.path = Path(arena_root).expanduser().resolve() / "queue.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def enqueue(self, item: QueueItem) -> bool:
        """Append item unless an equivalent queued/running item already exists.

        Returns True when the item was written and False when it was skipped as a
        duplicate. Existing callers that ignore the return value keep working.
        """
        with self._lock():
            rows = self._read_unlocked()
            row = asdict(item)
            row["fingerprint"] = row.get("fingerprint") or queue_fingerprint(row)

            existing_fingerprints = {
                r.get("fingerprint") or queue_fingerprint(r)
                for r in rows
                if str(r.get("status") or "queued") in self.ACTIVE_STATUSES
            }
            if row["fingerprint"] in existing_fingerprints:
                return False

            existing_ids = {str(r.get("item_id")) for r in rows}
            while row["item_id"] in existing_ids:
                row["item_id"] = f"q_{uuid.uuid4().hex[:12]}"

            rows.append(row)
            self._write_unlocked(rows)
            return True

    def list(self, status: str | None = None) -> list[dict]:
        with self._lock():
            rows = self._read_unlocked()
        if status:
            rows = [r for r in rows if r.get("status") == status]
        return sorted(rows, key=lambda r: float(r.get("priority") or 0.0), reverse=True)

    def pop_next(self) -> dict | None:
        with self._lock():
            rows = self._read_unlocked()
            queued = [r for r in rows if r.get("status") == "queued"]
            if not queued:
                return None
            item = max(queued, key=lambda r: float(r.get("priority") or 0.0))
            for r in rows:
                if r.get("item_id") == item.get("item_id"):
                    r["status"] = "running"
            self._write_unlocked(rows)
            return dict(item)

    def complete(self, item_id: str) -> None:
        self._set_status(item_id, "done")

    def fail(self, item_id: str) -> None:
        self._set_status(item_id, "failed")

    def _set_status(self, item_id: str, status: str) -> None:
        with self._lock():
            rows = self._read_unlocked()
            for r in rows:
                if r.get("item_id") == item_id:
                    r["status"] = status
            self._write_unlocked(rows)

    def _read(self) -> list[dict]:
        with self._lock():
            return self._read_unlocked()

    def _write(self, rows: list[dict]) -> None:
        with self._lock():
            self._write_unlocked(rows)

    def _read_unlocked(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Queue file is not valid JSON: {self.path}") from exc
        if not isinstance(data, list):
            raise RuntimeError(f"Queue file must contain a JSON list: {self.path}")
        return [dict(r) for r in data if isinstance(r, dict)]

    def _write_unlocked(self, rows: list[dict]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        if fcntl is not None:
            with self.lock_path.open("a+", encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            return
        lock_dir = self.lock_path.with_suffix(".lock.d")
        while True:  # pragma: no cover - used only without fcntl
            try:
                lock_dir.mkdir()
                break
            except FileExistsError:
                import time

                time.sleep(0.05)
        try:
            yield
        finally:
            try:
                lock_dir.rmdir()
            except OSError:
                pass


def queue_fingerprint(row: dict) -> str:
    source = str(row.get("source_run_id") or "")
    mutation_type = str(row.get("mutation_type") or "safe_refinement").strip().lower()
    proposed = " ".join(str(row.get("proposed_mutation") or "").strip().lower().split())
    payload = json.dumps([source, mutation_type, proposed], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
