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

# Two-host GPC correctness job (tpu_v7x_16_queue = 2 hosts x 8 chips).
#
# The Buildkite agent runs on the slice's first host. This script discovers
# the second host from the TPU VM's own metadata (the same slice plumbing
# run_multihost.sh uses), then runs
# scripts/vllm/integration/run_qwen35_gpc_correctness.sh in one container
# per host: the worker host carries replica B, the head host carries
# replica A and the smoke gates. Host networking makes replica B's API and
# replica A's offload store reachable across the slice.
#
# The global registry runs in its own bridge-network container on the head
# host: the binary only binds the IPv6 wildcard, these hosts run with IPv6
# disabled, and a private network namespace is the one place a job can
# re-enable it. The published port hands the listener back to the head
# host's IPv4 address, which both replicas dial.
#
# Weights stream from GCS via runai_streamer: these agents have only a
# 100 GB boot disk, which cannot hold the Docker image and a staged model
# snapshot at the same time.
set -euo pipefail

# Set HF_TOKEN from GCP Secret Manager if not already in /etc/environment
# (same bootstrap as run_in_docker.sh).
if ! grep -q "^HF_TOKEN=" /etc/environment 2>/dev/null; then
  gcloud secrets versions access latest --secret=bm-agent-hf-token --quiet | \
  sudo tee -a /etc/environment > /dev/null <<< "HF_TOKEN=$(cat)" || true
fi
if [ -f /etc/environment ]; then
  # shellcheck disable=SC1091
  source /etc/environment || true
fi

IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"
COMMIT_HASH="${BUILDKITE_COMMIT:-latest}"
IMAGE_TAG="${VLLM_TORCHTPU_IMAGE_TAG:-${IMAGE_REPO}:${COMMIT_HASH}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_GCS_PATH="${MODEL_GCS_PATH:-gs://tpu-inference-hf-llm-model-checkpoints/models--Qwen--Qwen3.5-35B-A3B-FP8/}"
RUN_ROOT="${RUN_ROOT:-/perf_eval_results/qwen35_gpc_ci}"
REGISTRY_PORT="${REGISTRY_PORT:-28500}"

SSH_USER="${SSH_USER:-$(whoami)}"
if [ ! -f ~/.ssh/id_rsa ]; then
  echo "--- Auto-generating SSH key for passwordless auth"
  mkdir -p ~/.ssh
  ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa -q
fi
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o ConnectTimeout=15 -i ~/.ssh/id_rsa)

ssh_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    # shellcheck disable=SC2029  # callers pass fully-built remote commands
    if ssh "${SSH_OPTS[@]}" "$@"; then
      return 0
    fi
    echo "ssh attempt ${attempt}/5 to ${1##*@} failed; retrying in $((attempt * 10))s"
    sleep $((attempt * 10))
  done
  return 1
}

# ---------------------------------------------------------------------------
# Discover slice IPs from GCP metadata
# ---------------------------------------------------------------------------
if [ -z "${WORKER_IPS:-}" ]; then
  ZONE="${ZONE:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}')}"
  TPU_NAME="${TPU_NAME:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/description" 2>/dev/null || echo "")}"
  if [ -z "$TPU_NAME" ] || [ -z "$ZONE" ]; then
    echo "ERROR: could not determine TPU_NAME/ZONE from metadata; set WORKER_IPS manually."
    exit 1
  fi
  echo "TPU_NAME=$TPU_NAME ZONE=$ZONE"
  ALL_IPS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format="value(networkEndpoints[].ipAddress)")
  ALL_IPS="${ALL_IPS//;/ }"
  ALL_IPS="${ALL_IPS//,/ }"
  # shellcheck disable=SC2206
  ALL_IPS_ARRAY=($ALL_IPS)
  HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-${ALL_IPS_ARRAY[0]}}"
  WORKER_IPS_LIST=("${ALL_IPS_ARRAY[@]:1}")
  WORKER_IPS=$(IFS=, ; echo "${WORKER_IPS_LIST[*]}")
fi
HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-$(hostname -I | awk '{print $1}')}"
IFS=',' read -r -a WORKER_IPS_ARRAY <<< "${WORKER_IPS}"
if [ "${#WORKER_IPS_ARRAY[@]}" -lt 1 ]; then
  echo "ERROR: this job needs a second slice host for replica B."
  exit 1
fi
WORKER_IP="${WORKER_IPS_ARRAY[0]}"
echo "Head IP: ${HEAD_INTERNAL_IP}  Worker IP: ${WORKER_IP}"

# ---------------------------------------------------------------------------
# The model URI runai_streamer will stream from. The bucket holds a
# HuggingFace hub-cache layout, so the servable directory (config.json plus
# safetensors) is the snapshot subdirectory, not the root.
# ---------------------------------------------------------------------------
MODEL_URI="${MODEL_URI:-}"
if [ -z "${MODEL_URI}" ]; then
  MODEL_URI="$(gcloud storage ls "${MODEL_GCS_PATH%/}/snapshots/" | sed -n '1p')"
  MODEL_URI="${MODEL_URI%/}"
fi
if [ -z "${MODEL_URI}" ] \
    || ! gcloud storage ls "${MODEL_URI}/config.json" >/dev/null; then
  echo "ERROR: no servable snapshot under ${MODEL_GCS_PATH} (resolved" \
    "MODEL_URI=${MODEL_URI:-<empty>}); expected a snapshot directory" \
    "holding config.json."
  exit 1
fi
echo "Model URI: ${MODEL_URI}"

# ---------------------------------------------------------------------------
# Results dir (no /mnt/disks on these agents; boot disk only)
# ---------------------------------------------------------------------------
PERSIST_ROOT="${HOME}/persist"
rm -rf "${PERSIST_ROOT}/perf_eval_results"
mkdir -p "${PERSIST_ROOT}/perf_eval_results"
chmod 777 "${PERSIST_ROOT}/perf_eval_results" 2>/dev/null || true
rm -rf perf_eval_results
mkdir -p perf_eval_results
# The agents' disks persist across jobs, and a stale staged model
# snapshot here can fill most of the 100 GB boot disk and starve the
# image pull.
rm -rf "${PERSIST_ROOT}/models"
echo "Head boot disk after cleanup: $(df -h / | tail -1)"

HEAD_LOGS_PID=""
cleanup() {
  echo "--- Tearing down GPC containers"
  if [ -n "${HEAD_LOGS_PID}" ]; then
    kill "${HEAD_LOGS_PID}" 2>/dev/null || true
  fi
  docker rm -f gpc-head gpc-registry >/dev/null 2>&1 || true
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${WORKER_IP}" \
    "docker rm -f gpc-worker >/dev/null 2>&1 || true" || true
}
trap cleanup EXIT INT TERM

cleanup

echo "--- Cleaning up old Docker images (head)"
bash "${SCRIPT_DIR}/cleanup_docker.sh" || true

echo "--- Pulling Docker image on the head host: ${IMAGE_TAG}"
docker pull "${IMAGE_TAG}"

echo "--- Preparing the worker host"
base64 < "${SCRIPT_DIR}/cleanup_docker.sh" > /tmp/cleanup_docker.b64
ssh_retry "${SSH_USER}@${WORKER_IP}" \
  "gcloud auth configure-docker us-central1-docker.pkg.dev --quiet >/dev/null 2>&1 || true; mkdir -p ~/gpc_ci"
ssh_retry "${SSH_USER}@${WORKER_IP}" "base64 -d > ~/gpc_ci/cleanup_docker.sh" < /tmp/cleanup_docker.b64
ssh_retry "${SSH_USER}@${WORKER_IP}" \
  "rm -rf ~/persist/models; bash ~/gpc_ci/cleanup_docker.sh || true; \
   echo \"Worker boot disk after cleanup: \$(df -h / | tail -1)\"; \
   docker pull '${IMAGE_TAG}'"

echo "--- Starting the global registry on the head host"
docker run -d --name gpc-registry \
  --sysctl net.ipv6.conf.all.disable_ipv6=0 \
  -p "${REGISTRY_PORT}:${REGISTRY_PORT}" \
  "${IMAGE_TAG}" \
  bash -c 'command -v global_registry_server >/dev/null || { \
     echo "global_registry_server not found: the tpu-raiden-torch pin \
predates the bundled GPC control plane; bump it in pyproject.toml"; \
     exit 1; }; \
   exec global_registry_server --port='"${REGISTRY_PORT}"

echo "--- Starting replica B on the worker host"
# Detached, so an interrupted ssh session cannot take the engine with it;
# the exit trap removes the container.
ssh_retry "${SSH_USER}@${WORKER_IP}" \
  "docker rm -f gpc-worker >/dev/null 2>&1 || true; \
   docker run -d --name gpc-worker --privileged --net=host --shm-size=64g \
     -e HF_TOKEN='${HF_TOKEN:-}' \
     -e GPC_MODEL_URI='${MODEL_URI}' \
     -e REGISTRY_PORT='${REGISTRY_PORT}' \
     -e GPC_ROLE=worker \
     -e GPC_HEAD_IP='${HEAD_INTERNAL_IP}' \
     -e GPC_MY_IP='${WORKER_IP}' \
     -w /root/torchtpu-vllm \
     '${IMAGE_TAG}' \
     bash scripts/vllm/integration/run_qwen35_gpc_correctness.sh"

echo "--- Starting the head-side test"
docker run -d --name gpc-head --privileged --net=host --shm-size=64g \
  -v "${PERSIST_ROOT}/perf_eval_results:/perf_eval_results" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e RUN_ROOT="${RUN_ROOT}" \
  -e GPC_MODEL_URI="${MODEL_URI}" \
  -e REGISTRY_PORT="${REGISTRY_PORT}" \
  -e GPC_ROLE=head \
  -e GPC_HEAD_IP="${HEAD_INTERNAL_IP}" \
  -e GPC_MY_IP="${HEAD_INTERNAL_IP}" \
  -e GPC_PEER_IP="${WORKER_IP}" \
  -w /root/torchtpu-vllm \
  "${IMAGE_TAG}" \
  bash -c 'umask 000; rm -rf /perf_eval_results/*; bash scripts/vllm/integration/run_qwen35_gpc_correctness.sh'
docker logs -f gpc-head &
HEAD_LOGS_PID=$!

# ---------------------------------------------------------------------------
# Supervise both containers: a dead replica B would otherwise leave the
# head polling an unreachable endpoint for the full startup timeout.
# ---------------------------------------------------------------------------
EXIT_CODE=1
while :; do
  if [ "$(docker inspect -f '{{.State.Running}}' gpc-head 2>/dev/null)" != "true" ]; then
    EXIT_CODE="$(docker inspect -f '{{.State.ExitCode}}' gpc-head 2>/dev/null || echo 1)"
    echo "--- Head test finished with exit code ${EXIT_CODE}"
    break
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' gpc-registry 2>/dev/null)" != "true" ]; then
    echo "ERROR: the registry container exited; aborting the head test."
    docker rm -f gpc-head >/dev/null 2>&1 || true
    EXIT_CODE=1
    break
  fi
  # Only an explicit "false" aborts: a dropped ssh probe must not kill a
  # healthy run.
  worker_running="$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${WORKER_IP}" \
    "docker inspect -f '{{.State.Running}}' gpc-worker 2>/dev/null" \
    2>/dev/null || echo unknown)"
  if [ "${worker_running}" = "false" ]; then
    echo "ERROR: replica B's container exited; aborting the head test."
    docker rm -f gpc-head >/dev/null 2>&1 || true
    EXIT_CODE=1
    break
  fi
  sleep 30
done

echo "--- Collecting the registry and replica B logs"
docker logs gpc-registry --tail 2000 \
  > "${PERSIST_ROOT}/perf_eval_results/registry.log" 2>&1 || true
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${WORKER_IP}" "docker logs gpc-worker --tail 5000" \
  > "${PERSIST_ROOT}/perf_eval_results/worker_server_b.log" 2>&1 || true

echo "--- Copying results back for artifact upload"
cp -r "${PERSIST_ROOT}/perf_eval_results"/* perf_eval_results/ 2>/dev/null || true
find perf_eval_results/ -type l ! -exec test -e {} \; -delete 2>/dev/null || true

exit "${EXIT_CODE}"
