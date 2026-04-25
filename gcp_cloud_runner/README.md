# Application Cloud Runner

A small, spec-driven wrapper for running local Python training/test code as an isolated Google Cloud Run Job with exactly one NVIDIA L4 GPU per job instance.

It is designed for automated training pipelines that invoke the local runner via `asyncio.create_subprocess_exec`, watch JSONL progress on stdout, and then read synced artifacts from a local output directory.

## Files

```text
application_cloud_runner.py   # only local entry point: validate -> build -> run -> poll -> sync -> cleanup
cloud_job.py                  # container-side launcher: runs command and uploads artifacts
cloud_entrypoint.sh           # container ENTRYPOINT
cloud_runner.example.yaml     # reusable spec template
deploy_infra.sh               # one-time infrastructure bootstrap
requirements.local.txt        # local dependency for YAML parsing
examples/minimal_train/       # tiny smoke-test app
```

## One-time setup

Create `.env` next to `deploy_infra.sh` or export these variables:

```bash
PROJECT_ID=my-gcp-project
REGION=us-central1
BUCKET_NAME=my-training-artifacts
AR_REPO=application-cloud-runner
SA_EMAIL=acr-runner@my-gcp-project.iam.gserviceaccount.com
```

Then run:

```bash
gcp_cloud_runner/deploy_infra.sh
```

The script enables Cloud Run, Artifact Registry, Cloud Build, Cloud Storage, IAM, and logging APIs; creates the Docker repo, bucket, and runtime service account; and applies an Artifact Registry cleanup policy that deletes older images while keeping recent versions.

No service-account key is created. Use Application Default Credentials locally, or workload identity / a CI service account in automation.

## Local dependency

```bash
python -m pip install -r requirements.local.txt
```

## App contract

Your training app should:

1. read configuration from environment variables,
2. write every output/checkpoint/metric needed by the pipeline under `$ACR_ARTIFACT_DIR`, and
3. exit non-zero on failure.

The container launcher uploads `$ACR_ARTIFACT_DIR` to `$ACR_OUTPUT_GCS_URI` even when the app fails. It also writes `_acr/run_start.json`, `_acr/run_result.json`, and `_acr/failure.json` for observability.

## YAML spec

Start from `cloud_runner.example.yaml`. The important sections are:

```yaml
files:
  include: ["train.py", "src/**", "requirements.txt"]
  required: ["train.py", "requirements.txt"]
  exclude: ["data/**", "artifacts/**", "*.pt", "*.pth"]
  hashes: {}
runtime:
  base_image: pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
  requirements: requirements.txt
  command: ["python", "train.py"]
cloud_run:
  gpu: 1
  gpu_type: nvidia-l4
  cpu: 4
  memory: 16Gi
  task_timeout: 3600s
artifacts:
  container_dir: /workspace/artifacts
  gcs_prefix: gs://${BUCKET_NAME}/training-runs/my-job
env:
  PYTHONUNBUFFERED: "1"
```

The runner validates that all required files exist, included files are not accidentally excluded, and optional SHA256 hashes match before it builds the container.

## Run once

```bash
python gcp_cloud_runner/application_cloud_runner.py \
  --app-dir minimal_train \
  --spec minimal_train/cloud_runner.yaml \
  --local-output-dir minimal_train/artifacts/run-001 \
  --env EPOCHS=3 \
  --env DATA_URI=gs://evo-training-data/datasets/v1
```

What happens:

1. validate the app folder against YAML,
2. create a temporary build context,
3. generate a GPU-friendly Dockerfile,
4. submit the image with Cloud Build,
5. create a unique Cloud Run Job using 1 L4 GPU, 4 CPU, and 16Gi memory by default,
6. execute the job with per-run env overrides,
7. poll the Cloud Run execution until success, failure, or local timeout,
8. collect Cloud Logging entries and the latest execution description,
9. sync artifacts from GCS to `--local-output-dir`, and
10. delete the Cloud Run Job resource. Images are normally removed by Artifact Registry retention; pass `--delete-image` for explicit tag deletion after completion.

## Async parent integration

```python
import asyncio
import json

async def run_training():
    proc = await asyncio.create_subprocess_exec(
        "python", "application_cloud_runner.py",
        "--app-dir", "/repo/trainer",
        "--spec", "/repo/trainer/cloud_runner.yaml",
        "--local-output-dir", "/repo/artifacts/job-123",
        "--env", "EPOCHS=5",
        "--env", "DATA_URI=gs://bucket/data/train",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    assert proc.stdout is not None
    async for raw in proc.stdout:
        line = raw.decode().rstrip()
        try:
            event = json.loads(line)
            print("runner event:", event["event"], event)
        except json.JSONDecodeError:
            print("runner output:", line)

    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"training job failed with exit code {rc}")

asyncio.run(run_training())
```

## Failure data

On failure or timeout, inspect:

```text
<local-output-dir>/_acr/local_run_summary.json
<local-output-dir>/_acr/execution_describe_latest.json
<local-output-dir>/_acr/logs/cloud_logging_tail.txt
<local-output-dir>/_acr/logs/cloud_logging_entries.json
<local-output-dir>/_acr/failure.json        # if the container uploaded it
```

## Notes and limits

Cloud Run L4 GPU jobs currently require at least 4 CPU and 16Gi memory and this implementation enforces `gpu: 1` and `gpu_type: nvidia-l4`. The default task timeout is 3600 seconds because Cloud Run GPU jobs are currently capped at one hour per task. For longer training, split work into resumable jobs or move the same spec pattern to a platform with longer GPU task lifetimes.
