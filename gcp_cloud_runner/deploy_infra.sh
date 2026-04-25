#!/usr/bin/env bash
set -euo pipefail

# One-time GCP bootstrap for Application Cloud Runner.
# Reads .env if present. Required: PROJECT_ID, REGION, BUCKET_NAME.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
elif [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  source ".env"
fi

: "${PROJECT_ID:?PROJECT_ID must be set}"
: "${REGION:?REGION must be set, for example us-central1}"
: "${BUCKET_NAME:?BUCKET_NAME must be set}"

AR_REPO="${AR_REPO:-app-cloud-runner}"
SA_NAME="${SA_NAME:-app-cloud-runner}"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
KEY_FILE_NAME="${SA_NAME}-key.json" 
RETENTION_DAYS="${RETENTION_DAYS:-3}"
KEEP_RECENT="${KEEP_RECENT:-2}"
REPO_DESCRIPTION="${REPO_DESCRIPTION:-Images for Application Cloud Runner jobs}"

printf '[infra] project=%s region=%s bucket=%s repo=%s service_account=%s\n' \
  "$PROJECT_ID" "$REGION" "$BUCKET_NAME" "$AR_REPO" "$SA_EMAIL"

gcloud config set project "$PROJECT_ID" >/dev/null

printf '[infra] Enabling APIs...\n'
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  logging.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com \
  serviceusage.googleapis.com >/dev/null

printf '[infra] Ensuring Artifact Registry repo...\n'
if ! gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$AR_REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --description="$REPO_DESCRIPTION" \
    --project="$PROJECT_ID"
fi

POLICY_FILE="$(mktemp)"
cat > "$POLICY_FILE" <<JSON
[
  {
    "name": "delete-acr-images-older-than-${RETENTION_DAYS}d",
    "action": {"type": "Delete"},
    "condition": {
      "tagState": "any",
      "olderThan": "${RETENTION_DAYS}d"
    }
  },
  {
    "name": "keep-recent-acr-images",
    "action": {"type": "Keep"},
    "mostRecentVersions": {
      "keepCount": ${KEEP_RECENT}
    }
  }
]
JSON
printf '[infra] Applying Artifact Registry cleanup policy: delete older than %sd, keep recent %s...\n' "$RETENTION_DAYS" "$KEEP_RECENT"
gcloud artifacts repositories set-cleanup-policies "$AR_REPO" \
  --project="$PROJECT_ID" \
  --location="$REGION" \
  --policy="$POLICY_FILE" \
  --no-dry-run >/dev/null
rm -f "$POLICY_FILE"

printf '[infra] Ensuring GCS bucket...\n'
if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project="$PROJECT_ID" \
    --location="$REGION" \
    --uniform-bucket-level-access
fi

# ==========================================
# 1. Create the Service Account
# ==========================================
printf '[infra] Ensuring runtime service account...\n'
if ! gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SA_NAME" \
    --project="$PROJECT_ID" \
    --description="Runtime service account for Application Cloud Runner jobs" \
    --display-name="Application Cloud Runner"
fi
# create ops are async, wait for eventual update
sleep 5

# ==========================================
# 2. Create & Save the API Key (JSON Key)
# ==========================================
if [ -f "$KEY_FILE_NAME" ]; then
    echo "✅ API Key already exists locally at ${KEY_FILE_NAME}. Skipping creation to prevent key rotation."
else
    echo "Generating and saving Service Account JSON key to ${KEY_FILE_NAME}..."
    gcloud iam service-accounts keys create $KEY_FILE_NAME \
        --iam-account=$SA_EMAIL

    # Secure the key file locally
    chmod 600 $KEY_FILE_NAME
    echo "✅ API Key successfully saved and secured: $KEY_FILE_NAME"
fi


printf '[infra] Granting runtime service-account roles...\n'
for role in roles/logging.logWriter roles/storage.objectAdmin roles/artifactregistry.reader; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$role" \
    --quiet >/dev/null
done

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
CLOUDBUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
printf '[infra] Granting Cloud Build push permissions to %s...\n' "$CLOUDBUILD_SA"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${CLOUDBUILD_SA}" \
  --role="roles/artifactregistry.writer" \
  --quiet >/dev/null || true

# Optional: grant a caller, CI service account, or pipeline identity permissions to execute jobs with env overrides.
# Example: RUNNER_INVOKER_MEMBER="serviceAccount:trainer-ci@PROJECT_ID.iam.gserviceaccount.com" ./deploy_infra.sh
if [[ -n "${RUNNER_INVOKER_MEMBER:-}" ]]; then
  printf '[infra] Granting runner caller permissions to %s...\n' "$RUNNER_INVOKER_MEMBER"
  for role in roles/run.developer roles/iam.serviceAccountUser roles/artifactregistry.admin roles/cloudbuild.builds.editor roles/logging.viewer roles/storage.objectAdmin; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
      --member="$RUNNER_INVOKER_MEMBER" \
      --role="$role" \
      --quiet >/dev/null
  done
fi

cat <<EOF
[infra] Done.
PROJECT_ID=${PROJECT_ID}
REGION=${REGION}
BUCKET_NAME=${BUCKET_NAME}
AR_REPO=${AR_REPO}
SA_EMAIL=${SA_EMAIL}

Add these to your runner spec or .env.
EOF

echo "=========================================="
echo "Deployment Complete!"
echo "API Key Path: $(pwd)/${KEY_FILE_NAME}"
echo "=========================================="
