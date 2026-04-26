from __future__ import annotations

import abc
import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import Any


class Reviewer(abc.ABC):
    @abc.abstractmethod
    async def review(self, parent: dict[str, Any] | None, child: dict[str, Any], context: str) -> dict[str, Any]:
        raise NotImplementedError


class HeuristicReviewer(Reviewer):
    async def review(self, parent: dict[str, Any] | None, child: dict[str, Any], context: str) -> dict[str, Any]:
        return heuristic_review(parent, child)


class ExternalCommandReviewer(Reviewer):
    """Adapter for a review LLM/script that returns JSON.

    The command receives JSON on stdin:
      {parent, child, context}
    """

    def __init__(self, command: str | list[str], timeout_s: int | None = None):
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        self.timeout_s = timeout_s

    async def review(self, parent: dict[str, Any] | None, child: dict[str, Any], context: str) -> dict[str, Any]:
        return await _run_json_command(
            self.command,
            {"parent": parent, "child": child, "context": context},
            cwd=Path(child.get("artifact_path") or "."),
            timeout_s=self.timeout_s,
        )


class ClaudeCodeReviewer(Reviewer):
    """Claude Code in read-only data-scientist mode.

    It should not edit files. It returns structured review JSON that the queue
    can consume.
    """

    def __init__(self, executable: str = "claude", max_turns: int = 4, timeout_s: int | None = None):
        self.executable = executable
        self.max_turns = max_turns
        self.timeout_s = timeout_s if timeout_s is not None else _env_int("EVOLVER_CLAUDE_REVIEW_TIMEOUT_S", 300)

    async def review(self, parent: dict[str, Any] | None, child: dict[str, Any], context: str) -> dict[str, Any]:
        prompt = _review_prompt(parent, child, context)
        cmd = _claude_print_command(
            executable=self.executable,
            prompt=prompt,
            permission_mode="default",
            allowed_tools=["Read"],
            max_turns=self.max_turns,
        )
        cwd = Path(child.get("artifact_path") or ".")
        print(f"[evolver] Claude Code review start cwd={cwd} timeout_s={self.timeout_s}", flush=True)
        stdout, stderr, code = await _run_text_command(cmd, cwd=cwd, timeout_s=self.timeout_s)
        print(f"[evolver] Claude Code review finished exit_code={code}", flush=True)
        if code != 0:
            raise RuntimeError(_command_error("Claude Code review", cmd, code, stdout, stderr))
        data = _parse_jsonish(stdout)
        if data.get("is_error") is True:
            raise RuntimeError(_command_error("Claude Code review", cmd, code, stdout, stderr))
        if isinstance(data.get("result"), str):
            data = _parse_jsonish(data["result"])
        return data




def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _claude_print_command(
    *,
    executable: str,
    prompt: str,
    permission_mode: str,
    allowed_tools: list[str],
    max_turns: int,
) -> list[str]:
    return [
        executable,
        "--bare",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--permission-mode",
        permission_mode,
        "--allowedTools",
        *allowed_tools,
        "--max-turns",
        str(max_turns),
    ]

def make_reviewer(kind: str, *, command: str | None = None, timeout_s: int | None = None) -> Reviewer:
    if kind == "heuristic":
        return HeuristicReviewer()
    if kind == "claude-code":
        return ClaudeCodeReviewer(timeout_s=timeout_s)
    if kind == "external-command":
        if not command:
            raise ValueError("external-command reviewer requires --reviewer-command")
        return ExternalCommandReviewer(command, timeout_s=timeout_s)
    raise ValueError(f"Unknown reviewer kind: {kind}")


def heuristic_review(parent: dict[str, Any] | None, child: dict[str, Any]) -> dict[str, Any]:
    """Small deterministic DS-review substitute.

    Replace with an LLM reviewer when desired; keep the output schema stable.
    """
    valid = child.get("status") == "succeeded"
    parent_score = float(parent.get("score") or -999.0) if parent else -999.0
    child_score = float(child.get("score") or -999.0)
    improvement = valid and child_score > parent_score

    recs = []
    if not valid:
        recs.append({
            "mutation_type": "failed_fix",
            "description": "Fix the smallest error while preserving the parent architecture.",
            "expected_benefit": "Recover a valid run.",
            "priority": 0.9,
        })
    elif improvement:
        recs.extend([
            {
                "mutation_type": "safe_refinement",
                "description": "Make one small accuracy-oriented refinement without increasing parameters by more than 25%.",
                "expected_benefit": "Exploit a promising branch.",
                "priority": 0.75,
            },
            {
                "mutation_type": "capacity_decrease",
                "description": "Try a smaller channel or classifier width while preserving the winning topology.",
                "expected_benefit": "Improve Pareto score via compression.",
                "priority": 0.65,
            },
        ])
    else:
        recs.append({
            "mutation_type": "novel_exploration",
            "description": "Try one different lightweight architecture idea while keeping the RP contract unchanged.",
            "expected_benefit": "Escape a stagnant branch.",
            "priority": 0.45,
        })

    observation = _observation(parent, child, valid, improvement)
    next_belief = _next_belief(valid, improvement)
    return {
        "valid": valid,
        "is_improvement": improvement,
        "parent_score": parent_score,
        "child_score": child_score,
        "branch_recommendation": "continue" if improvement else "deprioritize_or_fix",
        "observation": observation,
        "next_belief": next_belief,
        "recommended_next_mutations": recs,
    }


def _observation(parent: dict[str, Any] | None, child: dict[str, Any], valid: bool, improvement: bool) -> str:
    if not valid:
        return f"Run failed with error_type={child.get('error_type')}."
    if not parent:
        return "Baseline completed and established the first comparable run."
    delta = float(child.get("score") or -999.0) - float(parent.get("score") or -999.0)
    return f"Run {'improved' if improvement else 'did not improve'} scalar score by {delta:.4g}."


def _next_belief(valid: bool, improvement: bool) -> str:
    if not valid:
        return "Prioritize a minimal failed_fix mutation before further exploration."
    if improvement:
        return "This branch is worth exploiting and compressing with bounded follow-up mutations."
    return "This exact direction is less promising; prefer exploration or a smaller bounded change."


def _review_prompt(parent: dict[str, Any] | None, child: dict[str, Any], context: str) -> str:
    return f"""
You are the Data Scientist reviewer in a stable evolution loop.
Do not edit files. Judge the child run against its parent and propose bounded next mutations.

Return ONLY valid JSON:
{{
  "valid": true,
  "is_improvement": true,
  "improvement_type": ["accuracy", "pareto"],
  "regression_type": [],
  "confidence": 0.0,
  "branch_recommendation": "continue|deprioritize_or_fix|stop",
  "observation": "what happened",
  "next_belief": "what this suggests",
  "recommended_next_mutations": [
    {{"mutation_type": "safe_refinement", "description": "...", "expected_benefit": "...", "priority": 0.7}}
  ]
}}

Parent:
{json.dumps(parent, indent=2, default=str)}

Child:
{json.dumps(child, indent=2, default=str)}

Context:
{context}
""".strip()


async def _run_json_command(command: list[str], payload: dict[str, Any], *, cwd: Path, timeout_s: int | None) -> dict[str, Any]:
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(json.dumps(payload).encode("utf-8")), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Reviewer command timed out after {timeout_s}s: {command}")
    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(_command_error("external review command", command, int(proc.returncode or 0), stdout_text, stderr_text))
    return _parse_jsonish(stdout_text)


async def _run_text_command(command: list[str], *, cwd: Path, timeout_s: int | None) -> tuple[str, str, int]:
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
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
    if start >= 0 and end > start:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    raise ValueError(f"Could not parse JSON output: {text[:1000]}")




def _command_error(name: str, command: list[str], code: int, stdout: str, stderr: str) -> str:
    def tail(text: str, limit: int = 2000) -> str:
        text = text.strip()
        return text[-limit:] if text else "<empty>"

    safe_cmd = ["<prompt>" if i > 0 and command[i - 1] in {"-p", "--print"} else part for i, part in enumerate(command)]
    return (
        f"{name} failed with exit code {code}\n"
        f"command: {shlex.join(safe_cmd)}\n"
        f"stdout tail:\n{tail(stdout)}\n"
        f"stderr tail:\n{tail(stderr)}"
    )
