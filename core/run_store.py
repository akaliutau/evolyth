from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .ids import utc_now
from .records import RunRecord
from .rp import ResearchProblem


class RunArtifacts:
    """Simple filesystem artifact store for debuggable run folders."""

    def __init__(self, arena_root: str | Path):
        self.root = Path(arena_root).expanduser().resolve()
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def path(self, run_id: str) -> Path:
        p = self.runs_dir / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def snapshot_before_run(self, rp: ResearchProblem, run_id: str) -> Path:
        p = self.path(run_id)
        shutil.copy2(rp.model_path, p / "model.py")
        (p / "goal_prompt.md").write_text(rp.goal_prompt, encoding="utf-8")
        return p

    def capture_after_run(self, rp: ResearchProblem, record: RunRecord) -> None:
        p = self.path(record.run_id)
        for src, name in [
            (rp.metrics_path(record.run_id), "metrics.json"),
            (rp.events_path(record.run_id), "events.jsonl"),
            (rp.summary_path(record.run_id), "run_summary.md"),
        ]:
            if src.exists():
                shutil.copy2(src, p / name)

        record.artifact_path = str(p)
        record.model_path = str(p / "model.py")
        record.metrics_path = str(p / "metrics.json")
        record.events_path = str(p / "events.jsonl")
        record.summary_path = str(p / "run_summary.md")
        self.write_manifest(record)

    def write_manifest(self, record: RunRecord, extra: dict[str, Any] | None = None) -> None:
        p = self.path(record.run_id)
        manifest = record.to_dict()
        manifest["manifest_written_at"] = utc_now()
        if extra:
            manifest.update(extra)
        (p / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
