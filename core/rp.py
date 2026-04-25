from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ResearchProblem:
    """Filesystem contract for one stable research problem.

    The RP folder is the only entry point. Optional rp_contract.json may override
    commands, but the tiny-cifar conventions work without it.
    """

    path: Path
    rp_id: str
    goal_prompt: str
    mutable_file: str = "model.py"
    train_script: str = "train_eval.py"
    metrics_template: str = "runs/{run_id}/metrics.json"
    events_template: str = "runs/{run_id}/events.jsonl"
    summary_template: str = "runs/{run_id}/run_summary.md"
    smoke_args: list[str] = field(default_factory=lambda: ["--dry-run", "--dataset", "synthetic"])
    run_args: list[str] = field(default_factory=lambda: ["--dataset", "synthetic"])

    @classmethod
    def load(cls, rp_path: str | Path) -> "ResearchProblem":
        path = Path(rp_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"RP folder not found: {path}")

        goal_path = path / "goal_prompt.md"
        model_path = path / "model.py"
        train_path = path / "train_eval.py"
        missing = [str(p.name) for p in [goal_path, model_path, train_path] if not p.exists()]
        if missing:
            raise ValueError(f"Invalid RP folder {path}; missing: {', '.join(missing)}")

        cfg = {}
        cfg_path = path / "rp_contract.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        return cls(
            path=path,
            rp_id=str(cfg.get("rp_id") or path.name),
            goal_prompt=goal_path.read_text(encoding="utf-8"),
            mutable_file=str(cfg.get("mutable_file", "model.py")),
            train_script=str(cfg.get("train_script", "train_eval.py")),
            metrics_template=str(cfg.get("metrics_template", "runs/{run_id}/metrics.json")),
            events_template=str(cfg.get("events_template", "runs/{run_id}/events.jsonl")),
            summary_template=str(cfg.get("summary_template", "runs/{run_id}/run_summary.md")),
            smoke_args=list(cfg.get("smoke_args", ["--dry-run", "--dataset", "synthetic"])),
            run_args=list(cfg.get("run_args", ["--dataset", "synthetic"])),
        )

    def command(self, run_id: str, *, smoke: bool = False, extra_args: list[str] | None = None) -> list[str]:
        args = self.smoke_args if smoke else self.run_args
        return ["python", self.train_script, *args, "--run-id", run_id, *(extra_args or [])]

    def rp_file(self, relative: str) -> Path:
        return self.path / relative

    def metrics_path(self, run_id: str) -> Path:
        return self.path / self.metrics_template.format(run_id=run_id)

    def events_path(self, run_id: str) -> Path:
        return self.path / self.events_template.format(run_id=run_id)

    def summary_path(self, run_id: str) -> Path:
        return self.path / self.summary_template.format(run_id=run_id)

    @property
    def model_path(self) -> Path:
        return self.path / self.mutable_file
