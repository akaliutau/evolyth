from __future__ import annotations

import abc
import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .extractor import extract_record
from .ids import utc_now
from .records import RunRecord
from .rp import ResearchProblem
from .run_store import RunArtifacts


@dataclass
class RunSpec:
    rp: ResearchProblem
    run_id: str
    parent_id: str | None = None
    generation: int = 0
    smoke: bool = False
    extra_args: list[str] = field(default_factory=list)
    mutation_type: str = "baseline"
    mutation_summary: str = ""
    hypothesis: str = ""


class Executor(abc.ABC):
    @abc.abstractmethod
    async def execute(self, spec: RunSpec) -> RunRecord:
        raise NotImplementedError


async def _run_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: int | None,
    env: dict[str, str] | None = None,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=out,
            stderr=err,
            env=env,
        )
        try:
            return await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            err.write(f"\nExecutor timeout after {timeout_s}s\n".encode())
            return 124


def _record_from_artifacts(
    *,
    artifacts: RunArtifacts,
    spec: RunSpec,
    run_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    code: int,
    started: str,
    executor_name: str,
) -> RunRecord:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        metrics_path = spec.rp.metrics_path(spec.run_id)

    record = extract_record(
        rp=spec.rp,
        run_id=spec.run_id,
        parent_id=spec.parent_id,
        generation=spec.generation,
        artifact_path=str(run_dir),
        mutation_type=spec.mutation_type,
        mutation_summary=spec.mutation_summary,
        hypothesis=spec.hypothesis,
        metrics_path=metrics_path,
    )
    record.stdout_path = str(stdout_path)
    record.stderr_path = str(stderr_path)
    if code != 0 and record.status == "unknown":
        record.status = "failed"
        record.error_type = f"executor_exit_{code}"
    elif code != 0 and not record.error_type:
        record.error_type = f"executor_exit_{code}"
    record.observation = record.observation or f"{executor_name} started {started}; process exit code {code}."
    artifacts.finalize_record(record, extra={"executor": executor_name, "process_exit_code": code})
    return record


class LocalExecutor(Executor):
    """Run the RP command locally and write artifacts to arena/runs/<run_id>."""

    def __init__(self, artifacts: RunArtifacts, timeout_s: int | None = None):
        self.artifacts = artifacts
        self.timeout_s = timeout_s

    async def execute(self, spec: RunSpec) -> RunRecord:
        run_dir = self.artifacts.snapshot_before_run(spec.rp, spec.run_id)
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        cmd = spec.rp.command(spec.run_id, smoke=spec.smoke, extra_args=spec.extra_args)

        env = os.environ.copy()
        env.update({
            "ACR_RUN_ID": spec.run_id,
            "ACR_ARTIFACT_DIR": str(run_dir),
            "PYTHONUNBUFFERED": "1",
        })

        started = utc_now()
        code = await _run_subprocess(
            cmd,
            cwd=spec.rp.path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_s=self.timeout_s,
            env=env,
        )
        self.artifacts.capture_legacy_outputs(spec.rp, spec.run_id)
        return _record_from_artifacts(
            artifacts=self.artifacts,
            spec=spec,
            run_dir=run_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            code=code,
            started=started,
            executor_name="local",
        )


class CloudRunExecutor(Executor):
    """Run the RP through gcp_cloud_runner/application_cloud_runner.py.

    The cloud runner receives the same run id and syncs artifacts directly into
    arena/runs/<run_id>, matching LocalExecutor's on-disk contract.
    """

    def __init__(
        self,
        artifacts: RunArtifacts,
        *,
        cloud_runner_script: str | Path = "gcp_cloud_runner/application_cloud_runner.py",
        cloud_spec: str | Path | None = None,
        timeout_s: int | None = None,
        dataset: str | None = None,
        env: dict[str, str] | None = None,
        poll_interval_s: int = 15,
        log_format: str = "text",
        python_executable: str = sys.executable,
    ):
        self.artifacts = artifacts
        self.cloud_runner_script = Path(cloud_runner_script).expanduser()
        self.cloud_spec = Path(cloud_spec).expanduser() if cloud_spec else None
        self.timeout_s = timeout_s
        self.dataset = dataset
        self.env = dict(env or {})
        self.poll_interval_s = poll_interval_s
        self.log_format = log_format
        self.python_executable = python_executable

    async def execute(self, spec: RunSpec) -> RunRecord:
        run_dir = self.artifacts.snapshot_before_run(spec.rp, spec.run_id)
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        runner_script = self._resolve_runner_script()
        cloud_spec = self._resolve_cloud_spec(spec.rp)

        cmd = [
            self.python_executable,
            str(runner_script),
            "--app-dir",
            str(spec.rp.path),
            "--spec",
            str(cloud_spec),
            "--local-output-dir",
            str(run_dir),
            "--run-id",
            spec.run_id,
            "--poll-interval-seconds",
            str(self.poll_interval_s),
            "--log-format",
            self.log_format,
        ]
        if self.timeout_s is not None:
            cmd += ["--timeout-seconds", str(self.timeout_s)]
        if self.dataset:
            cmd += ["--dataset", self.dataset]
        for key, value in sorted(self.env.items()):
            cmd += ["--env", f"{key}={value}"]
        for arg in spec.extra_args:
            cmd.append(f"--command-arg={arg}")

        started = utc_now()
        code = await _run_subprocess(
            cmd,
            cwd=runner_script.parent.parent if runner_script.parent.name == "gcp_cloud_runner" else runner_script.parent,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_s=None,
            env=os.environ.copy(),
        )
        return _record_from_artifacts(
            artifacts=self.artifacts,
            spec=spec,
            run_dir=run_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            code=code,
            started=started,
            executor_name="cloud_run",
        )

    def _resolve_runner_script(self) -> Path:
        path = self.cloud_runner_script.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Cloud runner script not found: {path}")
        return path

    def _resolve_cloud_spec(self, rp: ResearchProblem) -> Path:
        path = (self.cloud_spec or (rp.path / "cloud_runner.yaml")).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Cloud runner spec not found: {path}. Pass --cloud-spec or add cloud_runner.yaml to the RP folder."
            )
        return path
