#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

set -euo pipefail

if [ -f /etc/environment ]; then
  # shellcheck disable=SC1091
  source /etc/environment || true
fi

export PATH="${REPO_DIR}:${REPO_DIR}/google-cloud-sdk/bin:${HOME}/bin:/usr/local/bin:/usr/bin:/bin:/snap/bin:${PATH:-}"
export USE_GKE_GCLOUD_AUTH_PLUGIN=True

if [ -z "${HF_TOKEN:-}" ]; then
  if command -v gcloud &>/dev/null; then
    echo "HF_TOKEN not set. Attempting to fetch from GCP Secret Manager..."
    HF_TOKEN=$(gcloud secrets versions access latest --secret=bm-agent-hf-token --quiet 2>/dev/null || true)
    export HF_TOKEN
  fi
fi

K8S_NAMESPACE="${K8S_NAMESPACE:-default}"
TIMEOUT_SECONDS=36000

STREAM_PIDS=""

on_exit() {
    stop_log_streams_and_upload
    cleanup_1p1d
}

# Cleanup whenever exit
trap on_exit EXIT

init_env() {
    export USE_GKE_GCLOUD_AUTH_PLUGIN=True

    # Get credentials to GKE cluster
    echo "gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --project $PROJECT_NAME"
    gcloud container clusters get-credentials "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT_NAME"

    # Ensure namespace exists if not default
    if [ "$K8S_NAMESPACE" != "default" ]; then
        kubectl create namespace "$K8S_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    fi

    # Ensure HF_TOKEN is set
    echo "kubectl create secret generic hf-token-secret -n $K8S_NAMESPACE --from-literal=token=[redacted] --dry-run=client -o yaml | kubectl apply -f -"
    kubectl create secret generic hf-token-secret -n "$K8S_NAMESPACE" --from-literal=token="$HF_TOKEN" --dry-run=client -o yaml | kubectl apply -f -

    # Create storage class (cluster-wide resource)
    echo "kubectl apply -f ./.buildkite/kubernetes/manifests/storageclass.yaml"
    kubectl apply -f ./.buildkite/kubernetes/manifests/storageclass.yaml

    # PVCs live in their own manifest and are applied idempotently but never
    # deleted, so the HF model cache and XLA compile cache survive across runs.
    echo "kubectl apply -n $K8S_NAMESPACE -f ./.buildkite/kubernetes/manifests/v7x/persistent_disagg.yaml"
    kubectl apply -n "$K8S_NAMESPACE" -f ./.buildkite/kubernetes/manifests/v7x/persistent_disagg.yaml
}

apply_manifest_with_image_override() {
    local manifest_file=$1
    if [ -n "${DOCKER_IMAGE:-}" ]; then
        echo "Applying $manifest_file in namespace $K8S_NAMESPACE with image override: $DOCKER_IMAGE"
        sed "s|image: us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod:latest|image: $DOCKER_IMAGE|g" "$manifest_file" | kubectl apply -n "$K8S_NAMESPACE" -f -
    else
        echo "Applying $manifest_file in namespace $K8S_NAMESPACE"
        kubectl apply -n "$K8S_NAMESPACE" -f "$manifest_file"
    fi
}

# Resolve :latest to its current digest so each build tests the newest image
# (regression coverage) while every container within the run pins the same
# immutable bytes (the tag once changed mid-run, breaking a restart pull and
# invalidating the compile cache). An explicit DOCKER_IMAGE wins.
IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod"
resolve_docker_image() {
    if [ -n "${DOCKER_IMAGE:-}" ]; then
        echo "Using explicit DOCKER_IMAGE override: ${DOCKER_IMAGE}"
        return 0
    fi
    local digest
    digest=$(gcloud artifacts docker images describe "${IMAGE_REPO}:latest" --format='value(image_summary.digest)' 2>/dev/null || true)
    if [ -n "$digest" ]; then
        DOCKER_IMAGE="${IMAGE_REPO}@${digest}"
        export DOCKER_IMAGE
        echo "Resolved ${IMAGE_REPO}:latest -> ${DOCKER_IMAGE}"
    else
        echo "WARNING: could not resolve latest image digest; deploying the :latest tag directly."
    fi
}

deploy_1p1d() {
    resolve_docker_image
    apply_manifest_with_image_override ./.buildkite/kubernetes/manifests/v7x/jobset_disagg.yaml
}

cleanup_1p1d() {
    echo "kubectl delete -n $K8S_NAMESPACE -f ./.buildkite/kubernetes/manifests/v7x/jobset_disagg.yaml"
    kubectl delete -n "$K8S_NAMESPACE" -f ./.buildkite/kubernetes/manifests/v7x/jobset_disagg.yaml --ignore-not-found --wait=false || true
}

# Delete leftovers from a previous (possibly canceled) build and wait until the
# TPU pods are actually gone, so a fresh JobSet doesn't land on nodes whose TPU
# devices are still held by terminating pods (causes instant job failures).
pre_deploy_cleanup() {
    echo "--- Cleaning up any leftover resources from previous runs..."
    kubectl delete jobset vllm-torchtpu-pd-disagg -n "$K8S_NAMESPACE" --ignore-not-found --wait=false || true
    for i in $(seq 1 40); do
        REMAINING=$(kubectl get pods -n "$K8S_NAMESPACE" -l 'app in (vllm-prefill,vllm-decode,vllm-benchmark)' --no-headers 2>/dev/null | wc -l | tr -d ' ')
        if [ "$REMAINING" = "0" ]; then
            echo "No leftover pods."
            return 0
        fi
        echo "Waiting for $REMAINING leftover pod(s) to terminate... (${i}/40)"
        sleep 15
    done
    echo "WARNING: leftover pods still terminating after 10 minutes; continuing anyway."
}

# Continuously stream a pod's container log to a local file, resuming if the
# pod restarts or the connection drops. Runs in the background for the whole
# benchmark; files are uploaded as Buildkite artifacts at the end.
POD_LOG_DIR="${REPO_DIR}/pod_logs"
stream_pod_log() {
    local label=$1 name=$2 container=$3
    local tail_arg="--tail=-1"
    while true; do
        local pod
        pod=$(kubectl get pods -n "$K8S_NAMESPACE" -l "$label" -o jsonpath="{.items[0].metadata.name}" 2>/dev/null || true)
        if [ -n "$pod" ]; then
            kubectl logs -f "$pod" -c "$container" -n "$K8S_NAMESPACE" --timestamps "$tail_arg" >> "${POD_LOG_DIR}/${name}.log" 2>/dev/null || true
            # After a successful capture, only append new lines on reconnect
            # instead of re-dumping the whole history into the file.
            if [ -s "${POD_LOG_DIR}/${name}.log" ]; then
                tail_arg="--tail=0"
            fi
        fi
        sleep 5
    done
}

start_log_streams() {
    mkdir -p "$POD_LOG_DIR"
    stream_pod_log "app=vllm-prefill" "prefill" "vllm-tpu" &
    STREAM_PIDS="$!"
    stream_pod_log "app=vllm-decode" "decode" "vllm-tpu" &
    STREAM_PIDS="$STREAM_PIDS $!"
    stream_pod_log "app=vllm-benchmark" "benchmark" "benchmark-runner" &
    STREAM_PIDS="$STREAM_PIDS $!"
}

stop_log_streams_and_upload() {
    # shellcheck disable=SC2086
    kill $STREAM_PIDS 2>/dev/null || true
    if command -v buildkite-agent &>/dev/null && ls "$POD_LOG_DIR"/*.log &>/dev/null; then
        (cd "$REPO_DIR" && buildkite-agent artifact upload "pod_logs/*.log") || true
    fi
}

# Full pod logs are streamed to files and uploaded as artifacts (see above),
# so the failure dump only summarizes cluster state and recent log tails.
dump_diagnostics() {
    echo "+++ ===== FAILURE DIAGNOSTICS ====="
    local f
    for f in "$POD_LOG_DIR"/prefill.log "$POD_LOG_DIR"/decode.log "$POD_LOG_DIR"/benchmark.log; do
        if [ -f "$f" ]; then
            echo "===== last 150 lines of streamed $(basename "$f") ====="
            tail -150 "$f" || true
        fi
    done
    echo "===== JobSet describe ====="
    kubectl describe jobset/vllm-torchtpu-pd-disagg -n "$K8S_NAMESPACE" 2>/dev/null | sed -n '/^Status:/,$p' || true
    echo "===== Events (most recent last) ====="
    kubectl get events -n "$K8S_NAMESPACE" --sort-by=.lastTimestamp 2>/dev/null | tail -40 || true
    echo "===== Pods ====="
    kubectl get pods -n "$K8S_NAMESPACE" -o wide 2>/dev/null || true
    echo "===== END FAILURE DIAGNOSTICS ====="
}


if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN is not set."
  exit 1
fi

if [ -z "${PROJECT_NAME:-}" ]; then
  echo "Error: PROJECT_NAME is not set."
  exit 1
fi

if [ -z "${CLUSTER_NAME:-}" ]; then
  echo "Error: CLUSTER_NAME is not set."
  exit 1
fi

if [ -z "${ZONE:-}" ]; then
  echo "Error: ZONE is not set."
  exit 1
fi

# Initialize GKE environment
init_env

# Remove leftovers from previous runs before deploying
pre_deploy_cleanup

# The benchmark pod has already terminated by the time results are needed
# (restartPolicy: Never), so results can't be copied with kubectl cp. The
# benchmark container printed each result file between BEGIN/END markers;
# extract them from the streamed log. Also called on timeout so completed
# cells are preserved even when the full sweep does not finish.
extract_results() {
    echo "Extracting benchmark results from streamed benchmark log..."
    python3 - "$POD_LOG_DIR/benchmark.log" <<'PYEOF'
import re
import sys

path = sys.argv[1]
current_name = None
current_lines = []
extracted = set()
with open(path) as f:
    for raw in f:
        # kubectl logs --timestamps prefixes each line with an RFC3339 stamp.
        line = re.sub(r'^\S+Z ', '', raw.rstrip('\n'))
        begin = re.match(r'===BEGIN_RESULT_FILE (\S+)===', line)
        if begin:
            current_name = begin.group(1)
            current_lines = []
        elif line.startswith('===END_RESULT_FILE==='):
            if current_name:
                with open(current_name, 'w') as out:
                    out.write('\n'.join(current_lines) + '\n')
                extracted.add(current_name)
            current_name = None
        elif current_name is not None:
            current_lines.append(line)
print(f"Extracted {len(extracted)} result file(s): {sorted(extracted)}")
PYEOF
}

# Deploy JobSet disaggregated serving & benchmark
deploy_1p1d

# Stream all pod logs to files for artifact upload and post-mortem debugging
start_log_streams

START_TIME=$SECONDS

echo "Waiting for JobSet vllm-torchtpu-pd-disagg benchmark execution to complete or fail..."
COUNTER=0
while true; do
    if [ "$((SECONDS - START_TIME))" -ge "$TIMEOUT_SECONDS" ]; then
        echo "ERROR: JobSet benchmark timed out after $((TIMEOUT_SECONDS / 60)) minutes."
        # Preserve whatever cells completed before the timeout. The build
        # still fails and nothing is uploaded to BigQuery; the extracted
        # JSONs ride along as build artifacts for manual inspection.
        extract_results
        if command -v buildkite-agent &>/dev/null; then
            buildkite-agent artifact upload "pd_disagg_*.json" || true
        fi
        dump_diagnostics
        exit 1
    fi

    # Check if JobSet completed
    if [ "$(kubectl get jobset/vllm-torchtpu-pd-disagg -n "$K8S_NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Completed")].status}' 2>/dev/null)" = "True" ]; then
        echo "JobSet completed successfully!"
        break
    fi

    # Check if JobSet failed
    if [ "$(kubectl get jobset/vllm-torchtpu-pd-disagg -n "$K8S_NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)" = "True" ]; then
        echo "ERROR: JobSet benchmark execution failed!"
        dump_diagnostics
        exit 1
    fi

    # Every 5 minutes (20 iterations * 15s), print pod status
    if [ "$((COUNTER % 20))" -eq 0 ]; then
        echo "[Elapsed: $(( (SECONDS - START_TIME) / 60 ))m] Pod status in namespace $K8S_NAMESPACE:"
        kubectl get pods -n "$K8S_NAMESPACE" || true
    fi

    COUNTER=$((COUNTER + 1))
    sleep 15
done

echo "------------------------------------------------"
echo "JobSet benchmark completed! Total elapsed time: $(( (SECONDS - START_TIME) / 60 )) minutes."
echo "------------------------------------------------"

extract_results

shopt -s nullglob
RESULT_FILES=(pd_disagg_*.json)
shopt -u nullglob
if [ "${#RESULT_FILES[@]}" -eq 0 ]; then
    echo "ERROR: benchmark completed but produced no result files."
    dump_diagnostics
    exit 1
fi

if command -v buildkite-agent &>/dev/null; then
    buildkite-agent artifact upload "pd_disagg_*.json" || true
fi

# A cell whose requests failed describes a broken serving path, not a
# measurement: after the prefill server died mid-run, one nightly still
# went green and inserted six zero-throughput rows into BigQuery. Fail
# the build and skip GCS/BigQuery upload when any cell recorded failed
# requests; the raw JSONs remain available as build artifacts above.
# The result JSONs are single-line objects with one top-level "failed"
# field, so a grep is sufficient.
FAILED_CELLS=$(grep -l -E '"failed": *[1-9]' "${RESULT_FILES[@]}" || true)
if [ -n "$FAILED_CELLS" ]; then
    for f in $FAILED_CELLS; do
        echo "ERROR: $f recorded failed requests: $(grep -oE '"(completed|failed)": *[0-9]+' "$f" | tr '\n' ' ')"
    done
    echo "ERROR: failing the build and skipping GCS/BigQuery upload."
    dump_diagnostics
    exit 1
fi

BASE_RECORD_ID="gke-vllm-torchtpu-run-$(date +%Y%m%d-%H%M%S)"

for RESULT_FILE in "${RESULT_FILES[@]}"; do

    RECORD_ID="${BASE_RECORD_ID}-${RESULT_FILE%.*}"

    if [ -n "${GCS_BUCKET:-}" ]; then
        GCS_PATH="gs://$GCS_BUCKET/$RECORD_ID/$RESULT_FILE"
        echo "Uploading results to GCS: $GCS_PATH"
        gsutil cp "$RESULT_FILE" "$GCS_PATH"
    fi

    # Upload results to BigQuery
    if [ -n "${PROJECT_NAME:-}" ]; then
        echo "Parsing results and inserting into BigQuery..."
        BQ_PROJECT_ID="${PROJECT_NAME}" python3 ./.buildkite/scripts/parse_gke_results.py "$RESULT_FILE" "$RECORD_ID"
    fi
done
