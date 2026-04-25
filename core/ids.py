from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def next_run_id(root: Path) -> str:
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    nums: list[int] = []
    for p in runs.iterdir():
        if p.is_dir() and p.name.startswith("run_"):
            try:
                nums.append(int(p.name.split("_", 1)[1]))
            except ValueError:
                pass
    return f"run_{(max(nums) + 1 if nums else 1):06d}"
