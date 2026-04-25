from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        payload: Dict[str, Any] = {
            "ts": time.time(),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        print(json.dumps(payload, sort_keys=True, default=str), flush=True)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
