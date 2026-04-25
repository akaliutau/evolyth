#!/usr/bin/env python3
"""Container-side launcher for Application Cloud Runner.

Runs the user command, streams logs to Cloud Run, writes run metadata, and uploads
all files from ACR_ARTIFACT_DIR to ACR_OUTPUT_GCS_URI even when the command fails.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(event: str, **fields: Any) -> None:
    payload = {"ts": utc_now(), "event": event, **fields}
    print(json.dumps(payload, sort_keys=True), flush=True)


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if not bucket:
        raise ValueError(f"Missing bucket in {uri!r}")
    return bucket, prefix.rstrip("/")


def upload_tree(local_dir: pathlib.Path, output_gcs_uri: str) -> None:
    try:
        from google.cloud import storage  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive in runtime image
        raise RuntimeError(
            "google-cloud-storage is required in the job image. The generated "
            "Dockerfile installs it automatically; custom Dockerfiles must do the same."
        ) from exc

    bucket_name, prefix = parse_gs_uri(output_gcs_uri)
    client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID"))
    bucket = client.bucket(bucket_name)

    if not local_dir.exists():
        log("artifact_dir_missing", local_dir=str(local_dir))
        return

    files = [p for p in local_dir.rglob("*") if p.is_file()]
    for path in files:
        rel = path.relative_to(local_dir).as_posix()
        blob_name = f"{prefix}/{rel}" if prefix else rel
        bucket.blob(blob_name).upload_from_filename(str(path))
    log("artifacts_uploaded", count=len(files), output_gcs_uri=output_gcs_uri)


def write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    artifact_dir = pathlib.Path(os.environ.get("ACR_ARTIFACT_DIR", "/workspace/artifacts")).resolve()
    output_gcs_uri = os.environ.get("ACR_OUTPUT_GCS_URI", "").strip()
    command_json = os.environ.get("ACR_COMMAND_JSON", "").strip()
    run_id = os.environ.get("ACR_RUN_ID", "unknown-run")

    if not output_gcs_uri:
        print("ACR_OUTPUT_GCS_URI is required", file=sys.stderr, flush=True)
        return 64
    if not command_json:
        print("ACR_COMMAND_JSON is required", file=sys.stderr, flush=True)
        return 64

    try:
        command = json.loads(command_json)
    except json.JSONDecodeError as exc:
        print(f"ACR_COMMAND_JSON is not valid JSON: {exc}", file=sys.stderr, flush=True)
        return 64
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        print("ACR_COMMAND_JSON must be a non-empty JSON list of strings", file=sys.stderr, flush=True)
        return 64

    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "hostname": socket.gethostname(),
        "command": command,
        "artifact_dir": str(artifact_dir),
        "output_gcs_uri": output_gcs_uri,
        "start_time": utc_now(),
        "cloud_run_task_index": os.environ.get("CLOUD_RUN_TASK_INDEX"),
        "cloud_run_task_count": os.environ.get("CLOUD_RUN_TASK_COUNT"),
        "cloud_run_execution": os.environ.get("CLOUD_RUN_EXECUTION"),
        "env_keys": sorted(k for k in os.environ if not k.lower().endswith(("key", "secret", "token", "password"))),
    }
    write_json(artifact_dir / "_acr" / "run_start.json", metadata)

    log("command_start", run_id=run_id, command=command, cwd=os.getcwd())
    started = time.monotonic()
    return_code = 1
    error: str | None = None
    try:
        proc = subprocess.run(command, cwd=os.environ.get("ACR_WORKDIR", "/workspace/app"), check=False)
        return_code = int(proc.returncode)
    except BaseException as exc:  # capture and upload details before exiting
        error = "".join(traceback.format_exception(exc))
        print(error, file=sys.stderr, flush=True)
        return_code = 70
    finally:
        finished = utc_now()
        duration = time.monotonic() - started
        result = {
            **metadata,
            "finish_time": finished,
            "duration_seconds": round(duration, 3),
            "return_code": return_code,
            "status": "succeeded" if return_code == 0 else "failed",
        }
        if error:
            result["exception"] = error
        write_json(artifact_dir / "_acr" / "run_result.json", result)
        if return_code != 0:
            write_json(artifact_dir / "_acr" / "failure.json", result)
        try:
            upload_tree(artifact_dir, output_gcs_uri)
        except BaseException as upload_exc:
            upload_error = "".join(traceback.format_exception(upload_exc))
            print(upload_error, file=sys.stderr, flush=True)
            # Preserve app failure if present; otherwise surface upload failure.
            if return_code == 0:
                return_code = 74
        log("command_finish", run_id=run_id, return_code=return_code)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
