from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EventPrinter:
    """Print to stdout and mirror structured events to JSONL."""

    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, name: str, **fields: Any) -> None:
        row = {"event": name, **fields}
        line = json.dumps(row, sort_keys=True, default=str)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
