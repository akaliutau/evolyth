from __future__ import annotations

import abc
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .rp import ResearchProblem


@dataclass
class MutationResult:
    mutation_type: str = "safe_refinement"
    mutation_summary: str = ""
    hypothesis: str = ""
    changed_files: list[str] = field(default_factory=list)
    raw_output: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MutationResult":
        return cls(
            mutation_type=str(data.get("mutation_type") or "safe_refinement"),
            mutation_summary=str(data.get("mutation_summary") or data.get("summary") or ""),
            hypothesis=str(data.get("hypothesis") or ""),
            changed_files=[str(x) for x in data.get("changed_files", [])],
            raw_output=str(data.get("raw_output") or ""),
        )


class MutationAgent(abc.ABC):
    @abc.abstractmethod
    async def mutate(self, rp: ResearchProblem, context: str) -> MutationResult:
        raise NotImplementedError


class NoopAgent(MutationAgent):
    """Deterministic smoke-test agent. It does not edit code."""

    async def mutate(self, rp: ResearchProblem, context: str) -> MutationResult:
        return MutationResult(
            mutation_type="baseline",
            mutation_summary="No-op baseline; run the current model.py unchanged.",
            hypothesis="Establish a valid baseline before branching.",
            changed_files=[],
        )


class ClaudeCodeAgent(MutationAgent):
    """Claude Code as a bounded mutation operator.

    Claude edits only the isolated RP workspace. The orchestrator validates that
    only model.py changed before execution.
    """

    def __init__(
        self,
        executable: str = "claude",
        permission_mode: str = "auto",
        max_turns: int = 8,
        timeout_s: int | None = None,
    ):
        self.executable = executable
        self.permission_mode = permission_mode
        self.max_turns = max_turns
        self.timeout_s = timeout_s

    async def mutate(self, rp: ResearchProblem, context: str) -> MutationResult:
        prompt = _claude_mutation_prompt(rp, context)
        cmd = [
            self.executable,
            "--bare",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--permission-mode",
            self.permission_mode,
            "--allowedTools",
            "Read,Edit,Bash",
            "--max-turns",
            str(self.max_turns),
        ]
        print("running agent:")
        print(cmd)
        stdout, stderr, code = await _run_text_command(cmd, cwd=rp.path, timeout_s=self.timeout_s)
        if code != 0:
            raise RuntimeError(f"Claude Code mutation failed with exit code {code}: {stderr[-2000:]}")
        data = _parse_jsonish(stdout)
        if isinstance(data.get("result"), str):
            data = _parse_jsonish(data["result"])
        data.setdefault("raw_output", stdout)
        return MutationResult.from_dict(data)


def make_agent(kind: str, *, command: str | None = None, timeout_s: int | None = None) -> MutationAgent:
    if kind == "noop":
        return NoopAgent()
    if kind == "claude-code":
        return ClaudeCodeAgent(timeout_s=timeout_s)
    raise ValueError(f"Unknown agent kind: {kind}")


def _claude_mutation_prompt(rp: ResearchProblem, context: str) -> str:
    return f"""
You are a code mutation worker inside a stable evolution system.

Hard rules:
- Edit only `{rp.mutable_file}`.
- Do not edit train_eval.py, requirements.txt, Dockerfile, immutable/*, data/*, or any run logs.
- Preserve the public model contract described by the RP.
- Make exactly one bounded architecture/design mutation.
- Keep the implementation simple, reliable, and easy to modify.
- Do not add dependencies unless the RP already allows them.
- Do not use pretrained weights or external data.
- Run the dry-run command if it is cheap and available.

After editing, return ONLY valid JSON with this shape:
{{
  "mutation_type": "safe_refinement|capacity_increase|capacity_decrease|regularization_change|architecture_swap|latency_optimization|failed_fix|novel_exploration|baseline",
  "mutation_summary": "one sentence",
  "hypothesis": "why this should help",
  "changed_files": ["{rp.mutable_file}"]
}}

{context}
""".strip()


async def _run_json_command(command: list[str], payload: dict[str, Any], *, cwd: Path, timeout_s: int | None) -> dict[str, Any]:
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(json.dumps(payload).encode("utf-8")),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Command timed out after {timeout_s}s: {command}")
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))
    return _parse_jsonish(stdout.decode("utf-8", errors="replace"))


async def _run_text_command(command: list[str], *, cwd: Path, timeout_s: int | None) -> tuple[str, str, int]:
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return "", f"timeout after {timeout_s}s", 124
    return (
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        int(proc.returncode or 0),
    )


def _parse_jsonish(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    raise ValueError(f"Could not parse JSON output: {text[:1000]}")
