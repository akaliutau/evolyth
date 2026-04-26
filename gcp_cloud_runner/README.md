# Application Cloud Runner v2

A small, spec-driven runner for executing an external Python training repo as a fresh Google Cloud Run Job 
with exactly one NVIDIA L4 GPU per job instance.

Version 2 separates **runner image deployment** from **per-run source execution**:

- the image is built only when the base image or requirements change;
- every actual run packs selected files from `--app-dir`, uploads them to GCS, creates a new Cloud Run Job from the prebuilt image, and lets the cloud launcher download/unpack the source inside the instance;
- optional `dataset` GCS input is downloaded before the training command;
- artifacts are uploaded back to GCS and synced locally;
- logs are concise on stdout and grouped by event type locally.

## Files

```text
application_cloud_runner.py   # per-run entry point: validate -> pack/upload source -> create job -> poll -> sync -> cleanup
deploy_runner_image.py        # separate image builder: base image + requirements.txt + generic launcher
deploy_runner_image.sh        # bash wrapper around deploy_runner_image.py
cloud_job.py                  # container-side launcher: downloads source/dataset, runs command, uploads artifacts
cloud_entrypoint.sh           # container ENTRYPOINT
deploy_infra.sh               # one-time infrastructure bootstrap
cloud_runner.example.yaml     # reusable spec template
requirements.local.txt        # local dependency for YAML parsing
examples/minimal_train/       # tiny smoke-test app
```

## One-time infrastructure

Create `.env` next to the scripts or export these variables:

```bash
PROJECT_ID=my-gcp-project
REGION=us-central1
BUCKET_NAME=my-training-artifacts
AR_REPO=application-cloud-runner
SA_EMAIL=acr-runner@my-gcp-project.iam.gserviceaccount.com
```

Then run:

```bash
./deploy_infra.sh
```

The script enables Cloud Run, Artifact Registry, Cloud Build, Cloud Storage, IAM, and logging APIs; creates the Docker repo, bucket, and runtime service account; and applies an Artifact Registry cleanup policy that keeps only the most recent 2 image versions by default.

No service-account key is created. Use Application Default Credentials locally, or workload identity / a CI service account in automation.

## Local dependency

```bash
python -m pip install -r requirements.local.txt
```

## Build the reusable runner image

Build this only when the base image or Python dependencies change:

```bash
./deploy_runner_image.sh \
  --spec /path/to/cloud_runner.yaml \
  --requirements requirements.txt
```

You can also pass everything explicitly:

```bash
./deploy_runner_image.sh \
  --project-id "$PROJECT_ID" \
  --region "$REGION" \
  --artifact-repo "$AR_REPO" \
  --image-name acr-torch-runner \
  --image-tag latest \
  --base-image pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime \
  --requirements /repos/trainer/requirements.txt \
  --keep-images 2
```

The image contains only the generic launcher, GCS client, and dependencies. It does **not** contain the training source code.

## App contract

Your external training repo should:

1. read run parameters from environment variables;
2. read data from `$DATASET_DIR` when `dataset` is configured;
3. write all outputs/checkpoints/metrics under `$ACR_ARTIFACT_DIR`;
4. exit non-zero on failure.

The container launcher uploads `$ACR_ARTIFACT_DIR` to `$ACR_OUTPUT_GCS_URI` even when the app fails. It also writes `_acr/run_start.json`, `_acr/run_result.json`, and `_acr/failure.json` for observability.

## YAML spec

Start from `cloud_runner.example.yaml`. Important sections:

```yaml
name: torch-train
project_id: ${PROJECT_ID}
region: ${REGION}
bucket: ${BUCKET_NAME}
artifact_repo: ${AR_REPO}
service_account: ${SA_EMAIL}

image:
  name: acr-torch-runner
  tag: latest

build:
  base_image: pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
  requirements: requirements.txt

files:
  include: ["train.py", "src/**", "configs/**", "requirements.txt"]
  required: ["train.py"]
  exclude: ["data/**", "artifacts/**", "checkpoints/**", "*.pt", "*.pth"]
  hashes: {}

source:
  gcs_prefix: gs://${BUCKET_NAME}/acr-sources/torch-train

runtime:
  command: ["python", "train.py"]
  workdir: /workspace/app

dataset:
  uri: gs://${BUCKET_NAME}/datasets/example-dataset/data.tar.gz
  container_dir: /workspace/dataset
  # auto: archives are single objects, other URIs are prefixes/folders.
  # prefix: force folder/prefix download. object: force single-object download.
  mode: auto
  unpack: auto

cloud_run:
  gpu: 1
  gpu_type: nvidia-l4
  cpu: 4
  memory: 16Gi
  task_timeout: 3600s

artifacts:
  container_dir: /workspace/artifacts
  gcs_prefix: gs://${BUCKET_NAME}/training-runs/torch-train
```

`files.*` is evaluated against the external repo passed to `--app-dir`. The runner validates required files and optional SHA256 hashes before uploading any source.

`dataset` may be omitted. When present, it can point to a GCS prefix/folder, a single file, or a `.tar`, `.tar.gz`, `.tgz`, or `.zip` archive. The cloud launcher downloads it before executing `runtime.command` and sets both `DATASET_DIR` and `DATASET_URI`. By default, `dataset.mode: auto` treats archive-looking URIs as single objects and other URIs as prefixes/folders. Set `dataset.mode: object` for one non-archive file, or `dataset.mode: prefix` for a folder.

## Run once from a parent pipeline

Upload dataset if needed: 

```bash
gcloud storage cp cifar-10-batches-py.tar.gz \
  gs://evo-training-data/datasets/tiny-cifar10/cifar-10-batches-py.tar.gz
```

```bash
python gcp_cloud_runner/application_cloud_runner.py   \
  --app-dir /media/DATA/repos-work/tiny-cifar   \
  --spec  /media/DATA/repos-work/tiny-cifar/cloud_runner.yaml   \
  --local-output-dir ./artifacts/run-001   \
  --env EPOCHS=3   \
  --env LR=0.0003
```

What happens:

1. validates `/repos/external-training-project` against the YAML file rules;
2. packs selected files into `source.tar.gz`;
3. uploads the source bundle to `source.gcs_prefix/<run_id>/source.tar.gz`;
4. creates a unique Cloud Run Job from the prebuilt runner image;
5. executes the job with per-run env overrides;
6. the cloud launcher downloads/unpacks the source into `/workspace/app`;
7. the cloud launcher downloads the optional dataset into `/workspace/dataset`;
8. the command runs and writes artifacts to `$ACR_ARTIFACT_DIR`;
9. the local runner polls execution state until success, failure, or timeout;
10. artifacts and grouped logs are synced to `--local-output-dir`;
11. the Cloud Run Job is deleted and the temporary source tarball is deleted after successful runs.

## Concise stdout format

Default stdout is text, not JSON:

```text
2026-04-25T16:09:16 RUN run_id=20260425-160916-0d5a8c73 job=torch-train-20260425-160916-0d5a8c73 out=gs://bucket/training-runs/torch-train/20260425-160916-0d5a8c73
2026-04-25T16:09:17 VAL 42 source files validated
2026-04-25T16:09:28 STA torch-train-abcde RUNNING
2026-04-25T16:12:02 STA torch-train-abcde SUCCEEDED
```

For a fully machine-readable parent integration, pass `--log-format json`.

## Grouped local logs

The runner no longer writes `_acr/logs/cloud_logging_entries.json`.

Instead, Cloud Logging entries are parsed and grouped into files such as:

```text
<local-output-dir>/_acr/logs/cmd_output.log
<local-output-dir>/_acr/logs/command_start.log
<local-output-dir>/_acr/logs/command_finish.log
<local-output-dir>/_acr/logs/source_downloaded.log
<local-output-dir>/_acr/logs/dataset_downloaded.log
<local-output-dir>/_acr/logs/artifacts_uploaded.log
```

Each line uses second-precision timestamps and the same compact shape:

```text
2026-04-25T16:09:16 OUT epoch=0 loss=1.0000
```

## Async integration

```python
import asyncio

async def run_training():
    proc = await asyncio.create_subprocess_exec(
        "python", "application_cloud_runner.py",
        "--app-dir", "/repos/external-training-project",
        "--spec", "/repos/external-training-project/cloud_runner.yaml",
        "--local-output-dir", "/pipeline/artifacts/job-123",
        "--env", "EPOCHS=5",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    assert proc.stdout is not None
    async for raw in proc.stdout:
        print(raw.decode().rstrip())

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
<local-output-dir>/_acr/logs/*.log
<local-output-dir>/_acr/failure.json        # if the container uploaded it
```

By default, the uploaded source tarball is kept on failure for debugging. Use `--delete-source-on-failure` to remove it even when the job fails.

## Notes

Cloud Run L4 GPU jobs require at least 4 CPU and 16Gi memory. This implementation enforces `gpu: 1` and `gpu_type: nvidia-l4`. 
The task timeout default is 3600 seconds; for longer training, split work into resumable jobs or adapt the same source-bundle pattern 
to a longer-running GPU platform.

Since runners are designed for exploratory research and architectures tests, 1h runtime is sufficient for such tasks.
