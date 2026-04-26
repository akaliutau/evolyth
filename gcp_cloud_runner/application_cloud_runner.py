#!/usr/bin/env python3
"""Application Cloud Runner: upload source, create a Cloud Run GPU Job, observe, sync, clean up.

Actual runs do not build Docker images. The stable runner image is built separately
with deploy_runner_image.py from a base image plus requirements.txt. Each run:
  1. validates an external training repo against the YAML spec,
  2. packs selected sources into a tarball,
  3. uploads the tarball to GCS,
  4. creates a fresh Cloud Run Job using the prebuilt runner image,
  5. executes and polls until terminal state/timeout,
  6. syncs artifacts and grouped logs back locally,
  7. deletes the Cloud Run Job and temporary source upload on success.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

DEFAULT_EXCLUDES = [
    ".git/**",
    ".venv/**",
    "venv/**",
    "env/**",
    "__pycache__/**",
    "*.pyc",
    ".pytest_cache/**",
    ".mypy_cache/**",
    ".ruff_cache/**",
    ".DS_Store",
    ".env",
    "*.pem",
    "*.p12",
    "*-key.json",
    "artifacts/**",
    "outputs/**",
    "output/**",
    "checkpoints/**",
]

EVENT_ACRONYMS = {
    "artifact_sync_done": "SYN",
    "artifact_sync_failed": "ERR",
    "cleanup_done": "CLN",
    "cleanup_job_skipped": "CLN",
    "cmd_finish": "CMD",
    "cmd_output": "OUT",
    "cmd_start": "CMD",
    "execution_started": "EXE",
    "execution_state": "STA",
    "execution_timeout": "TMO",
    "files_validated": "VAL",
    "logs_collected": "LOG",
    "runner_error": "ERR",
    "runner_interrupted": "INT",
    "run_resolved": "RUN",
    "source_packed": "SRC",
    "source_uploaded": "SRC",
    "source_upload_deleted": "CLN",
}

LOG_FORMAT = "text"


class RunnerError(RuntimeError):
    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    tail: list[str] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_ts() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")


def compact(value: Any, limit: int = 180) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def event_message(event: str, fields: dict[str, Any]) -> str:
    if "line" in fields:
        return str(fields["line"])
    if event == "cmd_start":
        return "$ " + " ".join(str(x) for x in fields.get("cmd", []))
    if event == "cmd_finish":
        return f"rc={fields.get('returncode')} {' '.join(str(x) for x in fields.get('cmd', [])[:4])}"
    if event == "run_resolved":
        return f"run_id={fields.get('run_id')} job={fields.get('job_name')} out={fields.get('output_gcs_uri')}"
    if event == "files_validated":
        return f"{fields.get('count')} source files validated"
    if event == "source_packed":
        return f"{fields.get('count')} files -> {fields.get('path')}"
    if event == "source_uploaded":
        return str(fields.get("source_gcs_uri"))
    if event == "execution_state":
        msg = compact(fields.get("message"), 220)
        return f"{fields.get('execution_name')} {fields.get('state')}" + (f" {msg}" if msg else "")
    if event == "execution_started":
        return str(fields.get("execution_name"))
    if event == "logs_collected":
        return f"{fields.get('entries')} entries -> {fields.get('path')}"
    if event == "artifact_sync_done":
        return f"{fields.get('output_gcs_uri')} -> {fields.get('local_output_dir')}"
    if event == "artifact_sync_failed":
        return f"rc={fields.get('returncode')} {fields.get('output_gcs_uri')}"
    if event == "source_upload_deleted":
        return str(fields.get("source_gcs_uri"))
    if event == "cleanup_done":
        return f"job={fields.get('job_name')}"
    if event == "runner_error":
        return f"exit={fields.get('exit_code')} {fields.get('message')}"
    parts = []
    for k, v in sorted(fields.items()):
        if k == "cmd":
            continue
        parts.append(f"{k}={compact(v, 100)}")
    return " ".join(parts)


def emit(event: str, **fields: Any) -> None:
    if LOG_FORMAT == "json":
        print(json.dumps({"ts": utc_now(), "event": event, **fields}, sort_keys=True), flush=True)
        return
    code = EVENT_ACRONYMS.get(event, event[:3].upper())
    print(f"{short_ts()} {code} {event_message(event, fields)}", flush=True)


def load_dotenv(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_yaml(path: pathlib.Path) -> dict[str, Any]:
    if yaml is None:
        raise RunnerError("PyYAML is required. Install with: python -m pip install pyyaml", 64)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RunnerError(f"Spec must be a YAML mapping: {path}", 64)
    return data


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def env_get(spec: dict[str, Any], key: str, env_key: str | None = None, default: str | None = None) -> str:
    env_key = env_key or key.upper()
    value = spec.get(key) or os.environ.get(env_key) or default
    if value is None or str(value).strip() == "":
        raise RunnerError(f"Missing required setting: {key} or environment variable {env_key}", 64)
    return str(value)


def safe_name(raw: str, max_len: int = 63) -> str:
    value = re.sub(r"[^a-z0-9-]+", "-", raw.lower()).strip("-")
    value = re.sub(r"-+", "-", value) or "acr-job"
    if value[0].isdigit():
        value = f"acr-{value}"
    return value[:max_len].strip("-") or "acr-job"


def unique_cloud_job_name(name: str, run_id: str) -> tuple[str, str]:
    """Cloud Run Job name for one orchestration attempt.

    The evolver run_id remains stable for artifacts and lineage. The Cloud Run
    Job resource gets an extra per-attempt suffix so stale jobs from killed or
    partially failed orchestrations cannot collide with a retry of the same run.
    """
    attempt_id = uuid.uuid4().hex[:8]
    prefix = safe_name(f"{name}-{run_id}", max_len=63 - 1 - len(attempt_id))
    return safe_name(f"{prefix}-{attempt_id}", max_len=63), attempt_id


def parse_key_value(items: Iterable[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise RunnerError(f"Expected KEY=VALUE, got: {item!r}", 64)
        key, value = item.split("=", 1)
        key = key.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            raise RunnerError(f"Invalid environment variable name: {key!r}", 64)
        env[key] = value
    return env


def load_env_file(path: pathlib.Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        raise RunnerError(f"Env file not found: {path}", 64)
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def format_gcloud_env(env: dict[str, Any]) -> str | None:
    if not env:
        return None
    pairs = [f"{k}={str(v)}" for k, v in sorted(env.items())]
    joined = "\n".join(pairs)
    for delim in ["|", "^", "~", "%", "@", ";", "___ACR___"]:
        if delim not in joined:
            return f"^{delim}^" + delim.join(pairs)
    raise RunnerError("Could not find a safe delimiter for gcloud env var values", 64)


def memory_to_gib(memory: str) -> float | None:
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)(Gi|G|Mi|M)$", memory.strip(), re.I)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    return value / 1024.0 if unit in {"mi", "m"} else value


def timeout_to_seconds(value: str | int) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text.isdigit():
        return int(text)
    m = re.match(r"^([0-9]+)(s|m|h)$", text)
    if not m:
        raise RunnerError(f"Unsupported timeout value: {value!r}; use seconds, 30m, or 1h", 64)
    n = int(m.group(1))
    return n * {"s": 1, "m": 60, "h": 3600}[m.group(2)]


def timeout_for_gcloud(value: str | int) -> str:
    return f"{timeout_to_seconds(value)}s"


def run_cmd(args: list[str], *, check: bool = True, stream: bool = False) -> CommandResult:
    emit("cmd_start", cmd=args)
    if stream:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            tail = tail[-80:]
            if line:
                emit("cmd_output", line=line)
        rc = proc.wait()
        emit("cmd_finish", cmd=args, returncode=rc)
        if check and rc != 0:
            raise RunnerError(f"Command failed with exit code {rc}: {' '.join(args)}\n" + "\n".join(tail), rc)
        return CommandResult(args=args, returncode=rc, tail=tail)

    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    emit("cmd_finish", cmd=args, returncode=completed.returncode)
    if check and completed.returncode != 0:
        raise RunnerError(
            f"Command failed with exit code {completed.returncode}: {' '.join(args)}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}",
            completed.returncode,
        )
    return CommandResult(args=args, returncode=completed.returncode, stdout=stdout, stderr=stderr)


def run_json(args: list[str], *, check: bool = True) -> Any:
    result = run_cmd(args, check=check, stream=False)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        if check:
            raise RunnerError(f"Expected JSON from command but got: {result.stdout[-2000:]}", 2)
        return None


def normalize_rel(path: pathlib.Path | str) -> str:
    return pathlib.PurePosixPath(str(path).replace(os.sep, "/")).as_posix().lstrip("./")


def match_pattern(rel: str, pattern: str) -> bool:
    rel = rel.replace(os.sep, "/").lstrip("./")
    pat = pattern.replace(os.sep, "/").lstrip("./")
    if pat in {"**", "**/*", "*"}:
        return True
    if pat.endswith("/"):
        pat += "**"
    if pat.endswith("/**"):
        prefix = pat[:-3].rstrip("/")
        return rel == prefix or rel.startswith(prefix + "/")
    return fnmatch.fnmatchcase(rel, pat) or pathlib.PurePosixPath(rel).match(pat)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_files(app_dir: pathlib.Path, spec: dict[str, Any]) -> tuple[list[pathlib.Path], list[dict[str, Any]]]:
    files_spec = spec.get("files") or {}
    if not isinstance(files_spec, dict):
        raise RunnerError("spec.files must be a mapping", 64)
    include = files_spec.get("include") or ["**/*"]
    exclude = DEFAULT_EXCLUDES + list(files_spec.get("exclude") or [])
    required = list(files_spec.get("required") or [])
    hashes = files_spec.get("hashes") or {}
    if not isinstance(include, list) or not isinstance(exclude, list) or not isinstance(required, list):
        raise RunnerError("files.include, files.exclude, and files.required must be lists", 64)
    if not isinstance(hashes, dict):
        raise RunnerError("files.hashes must be a mapping of path: sha256", 64)

    all_files = [p for p in app_dir.rglob("*") if p.is_file()]
    selected: list[pathlib.Path] = []
    for path in all_files:
        rel = normalize_rel(path.relative_to(app_dir))
        if any(match_pattern(rel, str(p)) for p in exclude):
            continue
        if any(match_pattern(rel, str(p)) for p in include):
            selected.append(path)

    selected_rels = {normalize_rel(p.relative_to(app_dir)) for p in selected}
    for rel_raw in required:
        rel = normalize_rel(rel_raw)
        path = app_dir / rel
        if not path.exists():
            raise RunnerError(f"Required file is missing: {rel}", 64)
        if path.is_dir():
            present = any(r == rel or r.startswith(rel.rstrip("/") + "/") for r in selected_rels)
        else:
            present = rel in selected_rels
        if not present:
            raise RunnerError(f"Required path exists but is excluded from uploaded source bundle: {rel}", 64)

    manifest: list[dict[str, Any]] = []
    for path in sorted(selected):
        rel = normalize_rel(path.relative_to(app_dir))
        digest = sha256_file(path)
        expected = str(hashes.get(rel, "")).replace("sha256:", "")
        if expected and expected != digest:
            raise RunnerError(f"SHA256 mismatch for {rel}: expected {expected}, got {digest}", 64)
        manifest.append({"path": rel, "size": path.stat().st_size, "sha256": digest})

    if not manifest:
        raise RunnerError("No files selected for source bundle. Check spec.files.include/exclude.", 64)
    emit("files_validated", count=len(manifest), required=required)
    return selected, manifest


def pack_source(app_dir: pathlib.Path, selected: list[pathlib.Path], manifest: list[dict[str, Any]], spec: dict[str, Any], out_tar: pathlib.Path) -> None:
    with tarfile.open(out_tar, "w:gz") as tar:
        for src in selected:
            rel = normalize_rel(src.relative_to(app_dir))
            tar.add(src, arcname=rel, recursive=False)
        with tempfile.TemporaryDirectory(prefix="acr-meta-") as td:
            meta = pathlib.Path(td) / ".acr"
            meta.mkdir(parents=True)
            (meta / "file_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            (meta / "run_spec_resolved.json").write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
            tar.add(meta, arcname=".acr")
    emit("source_packed", path=str(out_tar), count=len(manifest), bytes=out_tar.stat().st_size)


def resolve_command(spec: dict[str, Any]) -> list[str]:
    runtime = spec.get("runtime") or {}
    command = runtime.get("command")
    if command is None:
        entrypoint = runtime.get("entrypoint")
        if not entrypoint:
            raise RunnerError("runtime.command or runtime.entrypoint is required", 64)
        command = ["python", str(entrypoint)]
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise RunnerError("runtime.command must be a non-empty list of strings", 64)
    return command


def validate_cloud_run_shape(spec: dict[str, Any]) -> dict[str, Any]:
    cloud = spec.get("cloud_run") or {}
    if not isinstance(cloud, dict):
        raise RunnerError("cloud_run must be a mapping", 64)
    gpu = int(cloud.get("gpu", 1))
    gpu_type = str(cloud.get("gpu_type", "nvidia-l4"))
    cpu = int(cloud.get("cpu", 4))
    memory = str(cloud.get("memory", "16Gi"))
    task_timeout = cloud.get("task_timeout", "3600s")
    timeout_seconds = timeout_to_seconds(task_timeout)
    if gpu != 1 or gpu_type != "nvidia-l4":
        raise RunnerError("This runner is configured for exactly one Cloud Run L4 GPU per job instance: gpu=1, gpu_type=nvidia-l4", 64)
    if cpu < 4:
        raise RunnerError("Cloud Run L4 GPU jobs require at least 4 CPU", 64)
    mem_gib = memory_to_gib(memory)
    if mem_gib is not None and mem_gib < 16:
        raise RunnerError("Cloud Run L4 GPU jobs require at least 16Gi memory", 64)
    if timeout_seconds > 3600:
        raise RunnerError("Cloud Run GPU job task_timeout is capped at 3600s in current Cloud Run GPU jobs", 64)
    return {**cloud, "gpu": gpu, "gpu_type": gpu_type, "cpu": cpu, "memory": memory, "task_timeout": timeout_for_gcloud(task_timeout)}


def image_uri(region: str, project_id: str, repo: str, image_name: str, tag: str) -> str:
    return f"{region}-docker.pkg.dev/{project_id}/{repo}/{image_name}:{tag}"


def resolve_runner_image(spec: dict[str, Any], project_id: str, region: str, repo: str, default_name: str) -> str:
    image = spec.get("image") or spec.get("runner_image") or {}
    if isinstance(image, str):
        return image
    if not isinstance(image, dict):
        raise RunnerError("spec.image must be a string URI or mapping", 64)
    uri = image.get("uri") or os.environ.get("RUNNER_IMAGE_URI")
    if uri:
        return str(uri)
    name = str(image.get("name") or f"{default_name}-runner")
    tag = str(image.get("tag") or os.environ.get("RUNNER_IMAGE_TAG") or "latest")
    return image_uri(region, project_id, repo, safe_name(name, 60), tag)


def resolve_dataset(spec: dict[str, Any], cli_dataset: str | None) -> dict[str, str] | None:
    ds = spec.get("dataset")
    if cli_dataset:
        ds = {"uri": cli_dataset}
    if not ds:
        return None
    if isinstance(ds, str):
        ds = {"uri": ds}
    if not isinstance(ds, dict):
        raise RunnerError("dataset must be a gs:// URI string or mapping", 64)
    uri = str(ds.get("uri") or "").strip()
    if not uri:
        return None
    if not uri.startswith("gs://"):
        raise RunnerError("dataset.uri must be a gs:// URI", 64)
    mode = str(ds.get("mode") or "auto").lower()
    if mode not in {"auto", "prefix", "object"}:
        raise RunnerError("dataset.mode must be one of: auto, prefix, object", 64)
    return {
        "uri": uri,
        "container_dir": str(ds.get("container_dir") or "/workspace/dataset"),
        "unpack": str(ds.get("unpack") or "auto"),
        "mode": mode,
    }


def newest_execution(project_id: str, region: str, job_name: str) -> str:
    data = run_json([
        "gcloud", "run", "jobs", "executions", "list",
        f"--job={job_name}", f"--region={region}", f"--project={project_id}", "--limit=5", "--format=json",
    ])
    if not isinstance(data, list) or not data:
        raise RunnerError("Could not determine Cloud Run execution name after jobs execute", 2)
    item = data[0]
    name = item.get("name") or item.get("metadata", {}).get("name")
    if not name:
        raise RunnerError(f"Execution list item has no name: {item}", 2)
    return str(name).split("/")[-1]


def execution_name_from_execute(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    candidates = [data.get("name"), data.get("metadata", {}).get("name"), data.get("metadata", {}).get("target")]
    for c in candidates:
        if isinstance(c, str) and "/executions/" in c:
            return c.split("/")[-1]
        if isinstance(c, str) and re.match(r"^[a-z0-9-]+$", c) and "operations" not in c:
            return c
    return None


def parse_execution_state(data: dict[str, Any]) -> tuple[str, str]:
    status = data.get("status") if isinstance(data.get("status"), dict) else data
    conditions = status.get("conditions") or data.get("conditions") or []
    messages: list[str] = []
    for c in conditions:
        if not isinstance(c, dict):
            continue
        typ = str(c.get("type", ""))
        state = str(c.get("state") or c.get("status") or "")
        reason = str(c.get("reason") or "")
        msg = str(c.get("message") or "")
        if typ:
            messages.append(f"{typ}:{state}:{reason}:{msg}".strip(":"))
        if typ.lower() in {"completed", "succeeded"}:
            low = state.lower()
            if low in {"true", "condition_succeeded", "succeeded"}:
                return "SUCCEEDED", msg or reason
            if low in {"false", "condition_failed", "failed"}:
                return "FAILED", msg or reason
    completion = status.get("completionTime") or status.get("completionTimestamp") or data.get("completionTime")
    failed = status.get("failedCount") or status.get("failed")
    succeeded = status.get("succeededCount") or status.get("succeeded")
    if completion and failed:
        return "FAILED", "; ".join(messages)
    if completion and succeeded:
        return "SUCCEEDED", "; ".join(messages)
    if completion:
        return "COMPLETED_UNKNOWN", "; ".join(messages)
    return "RUNNING", "; ".join(messages)


def describe_execution(project_id: str, region: str, execution_name: str) -> dict[str, Any]:
    data = run_json([
        "gcloud", "run", "jobs", "executions", "describe", execution_name,
        f"--region={region}", f"--project={project_id}", "--format=json",
    ], check=True)
    if not isinstance(data, dict):
        raise RunnerError("Could not describe Cloud Run execution", 2)
    return data


def parse_cloud_payload(entry: dict[str, Any]) -> tuple[str, str, str]:
    ts = str(entry.get("timestamp") or entry.get("receiveTimestamp") or "")[:19]
    payload = entry.get("jsonPayload")
    if isinstance(payload, dict):
        event = str(payload.get("event") or "json_payload")
        line = payload.get("line") or payload.get("message")
        if line is None:
            rest = {k: v for k, v in payload.items() if k not in {"event", "ts"}}
            line = json.dumps(rest, sort_keys=True) if rest else ""
        return ts, event, str(line)
    text = entry.get("textPayload")
    if isinstance(text, str):
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data = json.loads(stripped)
                event = str(data.get("event") or "json_payload")
                line = data.get("line") or data.get("message")
                if line is None:
                    rest = {k: v for k, v in data.items() if k not in {"event", "ts"}}
                    line = json.dumps(rest, sort_keys=True) if rest else ""
                return ts, event, str(line)
            except json.JSONDecodeError:
                pass
        return ts, "cmd_output", stripped
    proto = entry.get("protoPayload")
    if proto is not None:
        return ts, "proto_payload", json.dumps(proto, sort_keys=True)
    return ts, "unknown", json.dumps(entry, sort_keys=True)


def sanitize_event_name(event: str) -> str:
    return safe_name(event.replace("_", "-"), 80).replace("-", "_")


def collect_logs(project_id: str, region: str, job_name: str, execution_name: str, out_dir: pathlib.Path, limit: int = 500) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    filters = [
        f'resource.type="cloud_run_job" AND resource.labels.job_name="{job_name}" AND labels."run.googleapis.com/execution_name"="{execution_name}"',
        f'resource.type="cloud_run_job" AND resource.labels.job_name="{job_name}"',
    ]
    entries: Any = []
    for flt in filters:
        result = run_json([
            "gcloud", "logging", "read", flt,
            f"--project={project_id}", "--freshness=7d", f"--limit={limit}", "--format=json",
        ], check=False)
        if isinstance(result, list) and result:
            entries = result
            break

    grouped: dict[str, list[tuple[str, str]]] = {}
    if isinstance(entries, list):
        for e in entries:
            if not isinstance(e, dict):
                continue
            ts, event, line = parse_cloud_payload(e)
            if not ts:
                ts = short_ts()
            grouped.setdefault(event, []).append((ts, line))
    for event, rows in grouped.items():
        rows.sort(key=lambda x: x[0])
        code = EVENT_ACRONYMS.get(event, event[:3].upper())
        path = out_dir / f"{sanitize_event_name(event)}.log"
        path.write_text("".join(f"{ts} {code} {line}\n" for ts, line in rows), encoding="utf-8")
    emit("logs_collected", entries=len(entries) if isinstance(entries, list) else 0, path=str(out_dir))


def sync_artifacts(output_gcs_uri: str, local_output_dir: pathlib.Path) -> None:
    local_output_dir.mkdir(parents=True, exist_ok=True)
    result = run_cmd(["gcloud", "storage", "rsync", "-r", output_gcs_uri, str(local_output_dir)], check=False, stream=True)
    if result.returncode != 0:
        emit("artifact_sync_failed", returncode=result.returncode, output_gcs_uri=output_gcs_uri, local_output_dir=str(local_output_dir))
    else:
        emit("artifact_sync_done", output_gcs_uri=output_gcs_uri, local_output_dir=str(local_output_dir))


def write_summary(path: pathlib.Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def delete_gcs_object(uri: str, project_id: str) -> None:
    run_cmd(["gcloud", "storage", "rm", uri, f"--project={project_id}"], check=False, stream=True)
    emit("source_upload_deleted", source_gcs_uri=uri)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an external Python repo as a Cloud Run L4 GPU Job without rebuilding the image.")
    parser.add_argument("--app-dir", required=True, help="Path to the external local repo/folder containing the training/test code")
    parser.add_argument("--spec", required=True, help="YAML spec describing source files, runner image, command, cloud resources, and artifacts")
    parser.add_argument("--env", action="append", default=[], help="Run-specific environment override passed to the Cloud Run execution; repeat KEY=VALUE")
    parser.add_argument("--env-file", action="append", default=[], help="File with KEY=VALUE lines to pass as run-specific env overrides")
    parser.add_argument("--dataset", default=None, help="Optional gs:// dataset URI overriding spec.dataset.uri")
    parser.add_argument("--run-id", default=None, help="Optional external run id. Defaults to a timestamp/uuid id.")
    parser.add_argument("--command-arg", action="append", default=[], help="Append one argument to runtime.command; repeat for multiple args")
    parser.add_argument("--local-output-dir", default=None, help="Where to sync artifacts after the job finishes")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="Local orchestration timeout; defaults to cloud_run.task_timeout + 15 minutes")
    parser.add_argument("--poll-interval-seconds", type=int, default=15, help="Execution polling interval")
    parser.add_argument("--keep-job-on-failure", action="store_true", help="Keep the Cloud Run Job resource if execution fails")
    parser.add_argument("--no-cleanup-job", action="store_true", help="Do not delete the Cloud Run Job resource after execution")
    parser.add_argument("--keep-source-upload", action="store_true", help="Keep the uploaded source tarball in GCS after success")
    parser.add_argument("--delete-source-on-failure", action="store_true", help="Also delete uploaded source tarball if the job fails")
    parser.add_argument("--log-format", choices=["text", "json"], default="text", help="stdout event format; text is concise, json is machine-readable")
    args = parser.parse_args()

    global LOG_FORMAT
    LOG_FORMAT = args.log_format

    app_dir = pathlib.Path(args.app_dir).resolve()
    spec_path = pathlib.Path(args.spec).resolve()
    if not app_dir.is_dir():
        raise RunnerError(f"app-dir not found or not a directory: {app_dir}", 64)
    if not spec_path.exists():
        raise RunnerError(f"spec not found: {spec_path}", 64)

    load_dotenv(pathlib.Path.cwd() / ".env")
    load_dotenv(spec_path.parent / ".env")
    load_dotenv(app_dir / ".env")
    spec = expand_env(load_yaml(spec_path))

    project_id = env_get(spec, "project_id", "PROJECT_ID")
    region = env_get(spec, "region", "REGION")
    bucket = env_get(spec, "bucket", "BUCKET_NAME")
    repo = str(spec.get("artifact_repo") or os.environ.get("AR_REPO") or "application-cloud-runner")
    service_account = str(spec.get("service_account") or os.environ.get("SA_EMAIL") or f"acr-runner@{project_id}.iam.gserviceaccount.com")
    name = safe_name(str(spec.get("name") or app_dir.name), max_len=36)
    run_id = args.run_id.strip() if args.run_id else f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", run_id):
        raise RunnerError("run-id must be 1-128 chars and contain only letters, digits, dot, underscore, or dash", 64)
    job_name, job_attempt_id = unique_cloud_job_name(name, run_id)
    runner_image = resolve_runner_image(spec, project_id, region, repo, default_name=name)
    source_prefix = str((spec.get("source") or {}).get("gcs_prefix") or f"gs://{bucket}/acr-sources/{name}").rstrip("/")
    source_gcs_uri = f"{source_prefix}/{run_id}/source.tar.gz"
    output_prefix = str(((spec.get("artifacts") or {}).get("gcs_prefix") or f"gs://{bucket}/acr-runs/{name}").rstrip("/"))
    output_gcs_uri = f"{output_prefix}/{run_id}"
    local_output_dir = pathlib.Path(args.local_output_dir or ((spec.get("artifacts") or {}).get("local_dir") or f"./artifacts/{run_id}")).resolve()

    command = [*resolve_command(spec), *args.command_arg]
    cloud = validate_cloud_run_shape(spec)
    dataset = resolve_dataset(spec, args.dataset)
    task_timeout_seconds = timeout_to_seconds(cloud["task_timeout"])
    orchestration_timeout = args.timeout_seconds or (task_timeout_seconds + 900)
    if args.poll_interval_seconds < 2:
        raise RunnerError("poll interval must be >= 2 seconds", 64)

    run_env: dict[str, str] = {}
    for env_file in args.env_file:
        run_env.update(load_env_file(pathlib.Path(env_file)))
    run_env.update(parse_key_value(args.env))

    spec_env = spec.get("env") or {}
    if not isinstance(spec_env, dict):
        raise RunnerError("spec.env must be a mapping", 64)
    artifact_dir = str((spec.get("artifacts") or {}).get("container_dir") or "/workspace/artifacts")
    workdir = str((spec.get("runtime") or {}).get("workdir") or "/workspace/app")
    deploy_env: dict[str, Any] = {
        **{str(k): str(v) for k, v in spec_env.items()},
        "PROJECT_ID": project_id,
        "GOOGLE_CLOUD_PROJECT": project_id,
        "REGION": region,
        "ACR_COMMAND_JSON": json.dumps(command),
        "ACR_ARTIFACT_DIR": artifact_dir,
        "ACR_WORKDIR": workdir,
    }
    execute_env: dict[str, Any] = {
        **run_env,
        "ACR_RUN_ID": run_id,
        "ACR_SOURCE_GCS_URI": source_gcs_uri,
        "ACR_OUTPUT_GCS_URI": output_gcs_uri,
    }
    if dataset:
        execute_env.update({
            "ACR_DATASET_URI": dataset["uri"],
            "ACR_DATASET_DIR": dataset["container_dir"],
            "ACR_DATASET_UNPACK": dataset["unpack"],
            "ACR_DATASET_MODE": dataset["mode"],
            "DATASET_URI": dataset["uri"],
            "DATASET_DIR": dataset["container_dir"],
        })

    emit("run_resolved", run_id=run_id, job_name=job_name, job_attempt_id=job_attempt_id, image_uri=runner_image, source_gcs_uri=source_gcs_uri, output_gcs_uri=output_gcs_uri, local_output_dir=str(local_output_dir))

    selected, manifest = collect_files(app_dir, spec)
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="acr-source-"))
    tar_path = tmp_dir / "source.tar.gz"
    summary: dict[str, Any] = {
        "run_id": run_id,
        "job_name": job_name,
        "job_attempt_id": job_attempt_id,
        "execution_name": None,
        "image_uri": runner_image,
        "source_gcs_uri": source_gcs_uri,
        "dataset": dataset,
        "command": command,
        "output_gcs_uri": output_gcs_uri,
        "local_output_dir": str(local_output_dir),
        "status": "initializing",
        "started_at": utc_now(),
    }
    delete_job = not args.no_cleanup_job
    execution_name: str | None = None
    terminal_state = "UNKNOWN"
    job_created = False
    try:
        pack_source(app_dir, selected, manifest, spec, tar_path)
        run_cmd(["gcloud", "storage", "cp", str(tar_path), source_gcs_uri, f"--project={project_id}"], check=True, stream=True)
        emit("source_uploaded", source_gcs_uri=source_gcs_uri)

        run_cmd(["gcloud", "artifacts", "repositories", "describe", repo, f"--location={region}", f"--project={project_id}"], check=True)
        deploy_env_arg = format_gcloud_env(deploy_env)
        create_cmd = [
            "gcloud", "run", "jobs", "create", job_name,
            "--image", runner_image,
            f"--region={region}",
            f"--project={project_id}",
            "--service-account", service_account,
            "--cpu", str(cloud["cpu"]),
            "--memory", str(cloud["memory"]),
            "--gpu", "1",
            "--gpu-type", "nvidia-l4",
            "--no-gpu-zonal-redundancy",
            "--tasks", str(cloud.get("tasks", 1)),
            "--parallelism", str(cloud.get("parallelism", 1)),
            "--max-retries", str(cloud.get("max_retries", 0)),
            "--task-timeout", str(cloud["task_timeout"]),
        ]
        if deploy_env_arg:
            create_cmd += ["--set-env-vars", deploy_env_arg]
        run_cmd(create_cmd, check=True, stream=True)
        job_created = True

        execute_cmd = [
            "gcloud", "run", "jobs", "execute", job_name,
            f"--region={region}", f"--project={project_id}", "--format=json",
        ]
        execute_env_arg = format_gcloud_env(execute_env)
        if execute_env_arg:
            execute_cmd += ["--update-env-vars", execute_env_arg]
        execution_data = run_json(execute_cmd, check=True)
        execution_name = execution_name_from_execute(execution_data) or newest_execution(project_id, region, job_name)
        summary["execution_name"] = execution_name
        emit("execution_started", execution_name=execution_name)

        deadline = time.monotonic() + orchestration_timeout
        last_message = ""
        while True:
            desc = describe_execution(project_id, region, execution_name)
            terminal_state, last_message = parse_execution_state(desc)
            write_summary(local_output_dir / "_acr" / "execution_describe_latest.json", desc)
            emit("execution_state", execution_name=execution_name, state=terminal_state, message=last_message)
            if terminal_state in {"SUCCEEDED", "FAILED", "COMPLETED_UNKNOWN"}:
                break
            if time.monotonic() > deadline:
                terminal_state = "TIMEOUT"
                emit("execution_timeout", execution_name=execution_name, timeout_seconds=orchestration_timeout)
                run_cmd(["gcloud", "run", "jobs", "executions", "cancel", execution_name, f"--region={region}", f"--project={project_id}", "--quiet"], check=False, stream=True)
                break
            time.sleep(args.poll_interval_seconds)

        collect_logs(project_id, region, job_name, execution_name, local_output_dir / "_acr" / "logs")
        sync_artifacts(output_gcs_uri, local_output_dir)
        summary["status"] = terminal_state.lower()
        summary["finished_at"] = utc_now()
        write_summary(local_output_dir / "_acr" / "local_run_summary.json", summary)

        if terminal_state != "SUCCEEDED":
            raise RunnerError(f"Cloud Run execution ended with state {terminal_state}: {last_message}", 2 if terminal_state == "FAILED" else 3)
        return 0
    finally:
        failed = terminal_state not in {"SUCCEEDED"}
        if execution_name and failed and args.keep_job_on_failure:
            emit("cleanup_job_skipped", reason="keep_job_on_failure", job_name=job_name)
        elif delete_job and job_created:
            run_cmd(["gcloud", "run", "jobs", "delete", job_name, f"--region={region}", f"--project={project_id}", "--quiet"], check=False, stream=True)
        if (not failed and not args.keep_source_upload) or (failed and args.delete_source_on_failure):
            delete_gcs_object(source_gcs_uri, project_id)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        emit("cleanup_done", job_name=job_name, source_bundle=str(tar_path))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as exc:
        emit("runner_error", message=str(exc), exit_code=exc.exit_code)
        raise SystemExit(exc.exit_code)
    except KeyboardInterrupt:
        emit("runner_interrupted")
        raise SystemExit(130)


