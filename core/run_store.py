from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .ids import utc_now
from .records import RunRecord
from .rp import ResearchProblem


class RunArtifacts:
    """Filesystem artifact store rooted at arena/runs."""

    def __init__(self, arena_root: str | Path):
        self.root = Path(arena_root).expanduser().resolve()
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def path(self, run_id: str) -> Path:
        path = self.runs_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def snapshot_before_run(self, rp: ResearchProblem, run_id: str) -> Path:
        path = self.path(run_id)
        shutil.copy2(rp.model_path, path / "model.py")
        (path / "goal_prompt.md").write_text(rp.goal_prompt, encoding="utf-8")
        return path

    def capture_legacy_outputs(self, rp: ResearchProblem, run_id: str) -> None:
        """Copy outputs from older RPs that still write under workspace/runs/<run_id>."""
        path = self.path(run_id)
        for src, name in [
            (rp.metrics_path(run_id), "metrics.json"),
            (rp.events_path(run_id), "events.jsonl"),
            (rp.summary_path(run_id), "run_summary.md"),
        ]:
            dst = path / name
            if src.exists() and src.resolve() != dst.resolve():
                shutil.copy2(src, dst)

    def capture_after_run(self, rp: ResearchProblem, record: RunRecord) -> None:
        self.capture_legacy_outputs(rp, record.run_id)
        self.finalize_record(record)

    def finalize_record(self, record: RunRecord, extra: dict[str, Any] | None = None) -> None:
        path = self.path(record.run_id)
        record.artifact_path = str(path)
        record.model_path = str(path / "model.py")
        record.metrics_path = str(path / "metrics.json")
        record.events_path = str(path / "events.jsonl")
        record.summary_path = str(path / "run_summary.md")
        self.write_manifest(record, extra=extra)

    def write_manifest(self, record: RunRecord, extra: dict[str, Any] | None = None) -> None:
        path = self.path(record.run_id)
        manifest = record.to_dict()
        manifest["manifest_written_at"] = utc_now()
        if extra:
            manifest.update(extra)
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
