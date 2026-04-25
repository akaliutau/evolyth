from __future__ import annotations

import abc
import asyncio
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


class LocalExecutor(Executor):
    """Async local subprocess executor.

    It assumes Claude Code or a human has already edited rp/model.py. The RP path
    stays the single execution entry point.
    """

    def __init__(self, artifacts: RunArtifacts, timeout_s: int | None = None):
        self.artifacts = artifacts
        self.timeout_s = timeout_s

    async def execute(self, spec: RunSpec) -> RunRecord:
        run_dir = self.artifacts.snapshot_before_run(spec.rp, spec.run_id)
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        cmd = spec.rp.command(spec.run_id, smoke=spec.smoke, extra_args=spec.extra_args)

        started = utc_now()
        code = await self._run(cmd, cwd=spec.rp.path, stdout_path=stdout_path, stderr_path=stderr_path)
        record = extract_record(
            rp=spec.rp,
            run_id=spec.run_id,
            parent_id=spec.parent_id,
            generation=spec.generation,
            artifact_path=str(run_dir),
            mutation_type=spec.mutation_type,
            mutation_summary=spec.mutation_summary,
            hypothesis=spec.hypothesis,
        )
        record.stdout_path = str(stdout_path)
        record.stderr_path = str(stderr_path)
        if code != 0 and record.status == "unknown":
            record.status = "failed"
            record.error_type = f"executor_exit_{code}"
        record.observation = record.observation or f"Executor started {started}; process exit code {code}."
        self.artifacts.capture_after_run(spec.rp, record)
        return record

    async def _run(self, cmd: Sequence[str], *, cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(cwd), stdout=out, stderr=err)
            try:
                return await asyncio.wait_for(proc.wait(), timeout=self.timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                err.write(f"\nExecutor timeout after {self.timeout_s}s\n".encode())
                return 124


class ModalExecutor(Executor):
    """Connector placeholder: same interface as LocalExecutor.

    Implement this by uploading the RP folder, running the same command in Modal,
    downloading the run folder, then returning a RunRecord.
    """

    async def execute(self, spec: RunSpec) -> RunRecord:
        raise NotImplementedError("ModalExecutor is intentionally a small connector stub.")
