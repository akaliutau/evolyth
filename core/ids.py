from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


RUN_ID_RE = re.compile(r"^run_(\d+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def next_run_id(root: Path | str) -> str:
    """Return and reserve the next monotonic evolver run id.

    This is intentionally stateful. A scan-only max+1 allocator can reuse ids
    after partial failures, concurrent CLI invocations, or when a workspace was
    created but the run was not registered yet. We take an arena-local lock,
    reconcile a durable counter with existing runs/workspaces/audit data, write
    the counter atomically, and create the run folder before returning.
    """
    arena = Path(root).expanduser().resolve()
    arena.mkdir(parents=True, exist_ok=True)

    with _arena_lock(arena):
        counter_path = arena / ".run_counter"
        current = max(_read_counter(counter_path), _max_existing_run_number(arena))
        n = current + 1
        while (arena / "runs" / _format_run_id(n)).exists() or (arena / "workspaces" / _format_run_id(n)).exists():
            n += 1

        run_id = _format_run_id(n)
        _write_counter(counter_path, n)
        (arena / "runs" / run_id).mkdir(parents=True, exist_ok=False)
        return run_id


def _format_run_id(n: int) -> str:
    return f"run_{n:06d}"


def _read_counter(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip() or "0")
    except (FileNotFoundError, ValueError):
        return 0


def _write_counter(path: Path, value: int) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(f"{value}\n", encoding="utf-8")
    os.replace(tmp, path)


def _max_existing_run_number(arena: Path) -> int:
    nums: list[int] = []
    for folder in [arena / "runs", arena / "workspaces"]:
        if folder.exists():
            nums.extend(_run_number(p.name) for p in folder.iterdir() if p.is_dir())

    audit = arena / "evolution.jsonl"
    if audit.exists():
        for line in audit.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            nums.append(_run_number(str(row.get("run_id") or "")))

    return max(nums or [0])


def _run_number(name: str) -> int:
    match = RUN_ID_RE.match(name)
    return int(match.group(1)) if match else 0


@contextmanager
def _arena_lock(arena: Path) -> Iterator[None]:
    lock_path = arena / ".run_id.lock"
    if fcntl is not None:
        with lock_path.open("a+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return

    lock_dir = arena / ".run_id.lock.d"
    while True:  # pragma: no cover - used only on platforms without fcntl
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
