#!/usr/bin/env python3
"""Container-side launcher for Application Cloud Runner.

The image contains this launcher and dependencies only. For every Cloud Run Job
execution it downloads a source tarball from GCS into ACR_WORKDIR, optionally
syncs a dataset URI into ACR_DATASET_DIR, runs the user command, streams output
as JSON events for Cloud Logging, and uploads artifacts even when the command
fails.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from google.api_core import exceptions as gcs_exceptions


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


def storage_client():
    try:
        from google.cloud import storage  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("google-cloud-storage must be installed in the runner image") from exc
    return storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID"))


def ensure_empty_dir(path: pathlib.Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_extract_tar(tar_path: pathlib.Path, dest: pathlib.Path) -> None:
    dest_resolved = dest.resolve()
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest_resolved) + os.sep) and target != dest_resolved:
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        # Python 3.12 warns unless a filter is explicit. data rejects dangerous
        # tar metadata such as absolute paths, device files, and unsafe links.
        tar.extractall(dest, filter="data")


def safe_extract_zip(zip_path: pathlib.Path, dest: pathlib.Path) -> None:
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest_resolved) + os.sep) and target != dest_resolved:
                raise RuntimeError(f"Unsafe zip member path: {name}")
        zf.extractall(dest)


def maybe_unpack(path: pathlib.Path, dest: pathlib.Path, mode: str = "auto") -> pathlib.Path:
    mode = (mode or "auto").lower()
    if mode in {"false", "no", "never"}:
        return path
    should_unpack = mode in {"true", "yes", "always"}
    if mode == "auto":
        suffixes = "".join(path.suffixes).lower()
        should_unpack = suffixes.endswith(('.tar.gz', '.tgz', '.tar', '.zip'))
    if not should_unpack:
        return path
    if path.name.endswith((".tar.gz", ".tgz", ".tar")):
        safe_extract_tar(path, dest)
        path.unlink(missing_ok=True)
        return dest
    if path.name.endswith(".zip"):
        safe_extract_zip(path, dest)
        path.unlink(missing_ok=True)
        return dest
    return path


def download_object(uri: str, dest_file: pathlib.Path) -> None:
    bucket_name, blob_name = parse_gs_uri(uri)
    client = storage_client()
    bucket = client.bucket(bucket_name)
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    bucket.blob(blob_name).download_to_filename(str(dest_file))


ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip")


def looks_like_archive(uri: str) -> bool:
    return uri.lower().rstrip("/").endswith(ARCHIVE_SUFFIXES)


def gcs_access_error(uri: str, exc: BaseException) -> RuntimeError:
    return RuntimeError(
        "Cannot read dataset from "
        f"{uri}. Check that the Cloud Run service account has storage.objects.get/list "
        "permission on the dataset bucket, that the object/prefix exists, and that "
        "billing is enabled for the project that owns the bucket. Original error: "
        f"{type(exc).__name__}: {exc}"
    )


def download_uri_to_dir(uri: str, dest_dir: pathlib.Path, *, unpack: str = "auto", mode: str = "auto") -> int:
    """Download a GCS dataset URI into dest_dir.

    mode=prefix treats uri as a GCS folder/prefix and never probes an exact
    object. This avoids surprising 403s from blob.exists() on prefix-like paths
    such as gs://bucket/datasets/v1.

    mode=object treats uri as a single GCS object. With unpack=auto, archives are
    extracted into dest_dir.

    mode=auto chooses object only for archive-looking URIs, otherwise prefix. For
    a single non-archive object, set dataset.mode: object.
    """
    bucket_name, prefix = parse_gs_uri(uri)
    client = storage_client()
    bucket = client.bucket(bucket_name)
    ensure_empty_dir(dest_dir)

    mode = (mode or "auto").lower()
    if mode not in {"auto", "prefix", "object"}:
        raise RuntimeError(f"Unsupported dataset mode {mode!r}; use auto, prefix, or object")
    if mode == "auto":
        mode = "object" if looks_like_archive(uri) else "prefix"

    try:
        if mode == "object":
            if not prefix:
                raise RuntimeError(f"Dataset object URI must include an object path: {uri}")
            target = dest_dir / pathlib.Path(prefix).name
            bucket.blob(prefix).download_to_filename(str(target))
            maybe_unpack(target, dest_dir, unpack)
            return 1

        prefix_slash = prefix.rstrip("/") + "/" if prefix else ""
        count = 0
        for blob in client.list_blobs(bucket_name, prefix=prefix_slash):
            if blob.name.endswith("/"):
                continue
            rel = blob.name[len(prefix_slash):] if prefix_slash and blob.name.startswith(prefix_slash) else pathlib.Path(blob.name).name
            if not rel:
                continue
            target = dest_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target))
            count += 1
        if count == 0:
            raise RuntimeError(f"No objects found at dataset prefix {uri}; use dataset.mode: object for a single non-archive object")
        return count
    except (gcs_exceptions.Forbidden, gcs_exceptions.NotFound, gcs_exceptions.GoogleAPICallError) as exc:
        raise gcs_access_error(uri, exc) from exc


def upload_tree(local_dir: pathlib.Path, output_gcs_uri: str) -> None:
    bucket_name, prefix = parse_gs_uri(output_gcs_uri)
    client = storage_client()
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


def parse_command(command_json: str) -> list[str]:
    try:
        command = json.loads(command_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ACR_COMMAND_JSON is not valid JSON: {exc}") from exc
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise RuntimeError("ACR_COMMAND_JSON must be a non-empty JSON list of strings")
    return command


def main() -> int:
    artifact_dir = pathlib.Path(os.environ.get("ACR_ARTIFACT_DIR", "/workspace/artifacts")).resolve()
    workdir = pathlib.Path(os.environ.get("ACR_WORKDIR", "/workspace/app")).resolve()
    output_gcs_uri = os.environ.get("ACR_OUTPUT_GCS_URI", "").strip()
    source_gcs_uri = os.environ.get("ACR_SOURCE_GCS_URI", "").strip()
    command_json = os.environ.get("ACR_COMMAND_JSON", "").strip()
    run_id = os.environ.get("ACR_RUN_ID", "unknown-run")
    dataset_uri = os.environ.get("ACR_DATASET_URI", "").strip()
    dataset_dir = pathlib.Path(os.environ.get("ACR_DATASET_DIR", "/workspace/dataset")).resolve()
    dataset_unpack = os.environ.get("ACR_DATASET_UNPACK", "auto")
    dataset_mode = os.environ.get("ACR_DATASET_MODE", "auto")

    if not output_gcs_uri:
        print("ACR_OUTPUT_GCS_URI is required", file=sys.stderr, flush=True)
        return 64
    if not source_gcs_uri:
        print("ACR_SOURCE_GCS_URI is required", file=sys.stderr, flush=True)
        return 64
    if not command_json:
        print("ACR_COMMAND_JSON is required", file=sys.stderr, flush=True)
        return 64

    try:
        command = parse_command(command_json)
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 64

    artifact_dir.mkdir(parents=True, exist_ok=True)
    acr_dir = artifact_dir / "_acr"
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "hostname": socket.gethostname(),
        "command": command,
        "artifact_dir": str(artifact_dir),
        "workdir": str(workdir),
        "source_gcs_uri": source_gcs_uri,
        "dataset_uri": dataset_uri or None,
        "dataset_dir": str(dataset_dir) if dataset_uri else None,
        "dataset_mode": dataset_mode if dataset_uri else None,
        "output_gcs_uri": output_gcs_uri,
        "start_time": utc_now(),
        "cloud_run_task_index": os.environ.get("CLOUD_RUN_TASK_INDEX"),
        "cloud_run_task_count": os.environ.get("CLOUD_RUN_TASK_COUNT"),
        "cloud_run_execution": os.environ.get("CLOUD_RUN_EXECUTION"),
        "env_keys": sorted(k for k in os.environ if not k.lower().endswith(("key", "secret", "token", "password"))),
    }
    write_json(acr_dir / "run_start.json", metadata)

    return_code = 1
    error: str | None = None
    started = time.monotonic()
    try:
        source_archive = pathlib.Path("/tmp/acr-source.tar.gz")
        log("source_download_start", source_gcs_uri=source_gcs_uri, workdir=str(workdir))
        download_object(source_gcs_uri, source_archive)
        ensure_empty_dir(workdir)
        safe_extract_tar(source_archive, workdir)
        log("source_downloaded", source_gcs_uri=source_gcs_uri, workdir=str(workdir))

        if dataset_uri:
            log("dataset_download_start", dataset_uri=dataset_uri, dataset_dir=str(dataset_dir), mode=dataset_mode)
            count = download_uri_to_dir(dataset_uri, dataset_dir, unpack=dataset_unpack, mode=dataset_mode)
            os.environ["DATASET_DIR"] = str(dataset_dir)
            os.environ["DATASET_URI"] = dataset_uri
            log("dataset_downloaded", dataset_uri=dataset_uri, dataset_dir=str(dataset_dir), count=count)

        log("command_start", run_id=run_id, command=command, cwd=str(workdir))
        proc = subprocess.Popen(command, cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            log("cmd_output", line=line)
        return_code = int(proc.wait())
    except BaseException as exc:
        error = "".join(traceback.format_exception(exc))
        log("runner_exception", message=str(exc))
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
        write_json(acr_dir / "run_result.json", result)
        if return_code != 0:
            write_json(acr_dir / "failure.json", result)
        try:
            upload_tree(artifact_dir, output_gcs_uri)
        except BaseException as upload_exc:
            upload_error = "".join(traceback.format_exception(upload_exc))
            log("artifact_upload_failed", message=str(upload_exc))
            print(upload_error, file=sys.stderr, flush=True)
            if return_code == 0:
                return_code = 74
        log("command_finish", run_id=run_id, return_code=return_code)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
