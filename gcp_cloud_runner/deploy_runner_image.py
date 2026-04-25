#!/usr/bin/env python3
"""Build and push the reusable Application Cloud Runner image.

This is intentionally separate from application_cloud_runner.py. Use it when the
base image or requirements change. Regular training runs should only upload
source bundles and reuse the pushed image.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

ROOT = pathlib.Path(__file__).resolve().parent


def safe_name(raw: str, max_len: int = 63) -> str:
    value = re.sub(r"[^a-z0-9-]+", "-", raw.lower()).strip("-")
    value = re.sub(r"-+", "-", value) or "acr-runner"
    return value[:max_len].strip("-") or "acr-runner"


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


def load_spec(path: pathlib.Path | None) -> dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise SystemExit("PyYAML is required for --spec. Install with: python -m pip install pyyaml")
    data = yaml.safe_load(os.path.expandvars(path.read_text(encoding="utf-8")))
    return data if isinstance(data, dict) else {}


def run(args: list[str], *, stream: bool = True) -> None:
    print("$ " + " ".join(args), flush=True)
    if stream:
        subprocess.run(args, check=True)
    else:
        subprocess.run(args, check=True, capture_output=True, text=True)


def image_uri(region: str, project_id: str, repo: str, image_name: str, tag: str) -> str:
    return f"{region}-docker.pkg.dev/{project_id}/{repo}/{image_name}:{tag}"


def resolve_image(spec: dict[str, Any], project_id: str, region: str, repo: str, cli_uri: str | None, cli_name: str | None, cli_tag: str | None) -> str:
    if cli_uri:
        return cli_uri
    image = spec.get("image") or spec.get("runner_image") or {}
    if isinstance(image, str):
        return image
    image = image if isinstance(image, dict) else {}
    uri = image.get("uri") or os.environ.get("RUNNER_IMAGE_URI")
    if uri:
        return str(uri)
    name = safe_name(str(cli_name or image.get("name") or f"{spec.get('name', 'acr')}-runner"), 60)
    tag = str(cli_tag or image.get("tag") or os.environ.get("RUNNER_IMAGE_TAG") or "latest")
    return image_uri(region, project_id, repo, name, tag)


def apply_cleanup_policy(project_id: str, region: str, repo: str, keep_images: int) -> None:
    policy = [
        {
            "name": "delete-old-runner-images",
            "action": {"type": "Delete"},
            "condition": {"tagState": "any", "olderThan": "1d"},
        },
        {
            "name": "keep-recent-runner-images",
            "action": {"type": "Keep"},
            "mostRecentVersions": {"keepCount": keep_images},
        },
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(policy, f, indent=2)
        tmp = f.name
    try:
        run([
            "gcloud", "artifacts", "repositories", "set-cleanup-policies", repo,
            f"--project={project_id}", f"--location={region}", f"--policy={tmp}", "--no-dry-run",
        ])
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/push the stable Application Cloud Runner image from a base image and requirements.txt.")
    parser.add_argument("--spec", default=None, help="Optional cloud_runner.yaml to read project/image/build defaults")
    parser.add_argument("--base-image", default=None, help="Base image, e.g. pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime")
    parser.add_argument("--requirements", default=None, help="requirements.txt to install into the runner image")
    parser.add_argument("--apt-package", action="append", default=[], help="Extra apt package to install; repeatable")
    parser.add_argument("--pip-package", action="append", default=[], help="Extra pip package to install; repeatable")
    parser.add_argument("--image-uri", default=None, help="Full image URI to push")
    parser.add_argument("--image-name", default=None, help="Image name if --image-uri is not provided")
    parser.add_argument("--image-tag", default=None, help="Image tag if --image-uri is not provided; default latest")
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--artifact-repo", default=None)
    parser.add_argument("--keep-images", type=int, default=2, help="Artifact Registry cleanup policy keep count")
    parser.add_argument("--no-cleanup-policy", action="store_true", help="Do not update Artifact Registry cleanup policy")
    args = parser.parse_args()

    load_dotenv(pathlib.Path.cwd() / ".env")
    spec_path = pathlib.Path(args.spec).resolve() if args.spec else None
    if spec_path:
        load_dotenv(spec_path.parent / ".env")
    spec = load_spec(spec_path)

    project_id = args.project_id or spec.get("project_id") or os.environ.get("PROJECT_ID")
    region = args.region or spec.get("region") or os.environ.get("REGION")
    repo = args.artifact_repo or spec.get("artifact_repo") or os.environ.get("AR_REPO") or "application-cloud-runner"
    if not project_id or not region:
        raise SystemExit("PROJECT_ID and REGION are required via args, spec, or environment")

    build = spec.get("build") or {}
    runtime = spec.get("runtime") or {}
    base_image = args.base_image or build.get("base_image") or runtime.get("base_image") or "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime"
    req_raw = args.requirements or build.get("requirements") or runtime.get("requirements")
    requirements: pathlib.Path | None = None
    if req_raw:
        req_path = pathlib.Path(str(req_raw))
        if not req_path.is_absolute() and spec_path:
            req_path = spec_path.parent / req_path
        requirements = req_path.resolve()
        if not requirements.exists():
            raise SystemExit(f"requirements file not found: {requirements}")

    apt_packages = list(build.get("apt_packages") or []) + args.apt_package
    pip_packages = list(build.get("extra_pip") or []) + args.pip_package
    img = resolve_image(spec, str(project_id), str(region), str(repo), args.image_uri, args.image_name, args.image_tag)

    build_dir = pathlib.Path(tempfile.mkdtemp(prefix="acr-runner-image-"))
    try:
        shutil.copy2(ROOT / "cloud_job.py", build_dir / "cloud_job.py")
        shutil.copy2(ROOT / "cloud_entrypoint.sh", build_dir / "cloud_entrypoint.sh")
        req_name = None
        if requirements:
            req_name = "requirements.app.txt"
            shutil.copy2(requirements, build_dir / req_name)
        dockerfile = [
            f"FROM {base_image}",
            "ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 ACR_WORKDIR=/workspace/app ACR_ARTIFACT_DIR=/workspace/artifacts",
            "WORKDIR /workspace",
        ]
        if apt_packages:
            dockerfile.append(
                "RUN apt-get update && apt-get install -y --no-install-recommends "
                + " ".join(str(p) for p in apt_packages)
                + " && rm -rf /var/lib/apt/lists/*"
            )
        dockerfile += [
            "RUN mkdir -p /opt/acr /workspace/app /workspace/artifacts /workspace/dataset",
            "COPY cloud_job.py /opt/acr/cloud_job.py",
            "COPY cloud_entrypoint.sh /opt/acr/cloud_entrypoint.sh",
            "RUN chmod +x /opt/acr/cloud_entrypoint.sh",
            "RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir google-cloud-storage pyyaml",
        ]
        if req_name:
            dockerfile += [f"COPY {req_name} /tmp/{req_name}", f"RUN python -m pip install --no-cache-dir -r /tmp/{req_name}"]
        if pip_packages:
            dockerfile.append("RUN python -m pip install --no-cache-dir " + " ".join(str(p) for p in pip_packages))
        dockerfile += ["ENTRYPOINT [\"/opt/acr/cloud_entrypoint.sh\"]"]
        (build_dir / "Dockerfile").write_text("\n".join(dockerfile) + "\n", encoding="utf-8")

        print(f"Building runner image: {img}", flush=True)
        print(f"Base image: {base_image}", flush=True)
        if requirements:
            print(f"Requirements: {requirements}", flush=True)
        run(["gcloud", "artifacts", "repositories", "describe", str(repo), f"--location={region}", f"--project={project_id}"])
        run(["gcloud", "builds", "submit", str(build_dir), "--tag", img, f"--project={project_id}"])
        if not args.no_cleanup_policy:
            apply_cleanup_policy(str(project_id), str(region), str(repo), args.keep_images)
        print(f"RUNNER_IMAGE_URI={img}")
        print("Use this image.uri in cloud_runner.yaml. Regular runs should call application_cloud_runner.py only.")
        return 0
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
