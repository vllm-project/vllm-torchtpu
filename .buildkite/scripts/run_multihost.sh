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

# Run a command inside a Ray-clustered container spanning a multi-host TPU
# slice (e.g. tpu_v7x_16_queue = 2 hosts x 8 chips).
#
# The Buildkite agent runs on the slice's head host. This script:
#   1. Discovers the worker host IPs from GCP metadata (same mechanism as
#      tpu-inference's .buildkite/scripts/run_multihost.sh, which runs on the
#      same agents).
#   2. Starts a Ray head container locally and a Ray worker container on each
#      worker host (via scripts/multihost/run_cluster.sh), all from the CI
#      image for this commit, with TPU_MULTIHOST_BACKEND=ray so vllm-torchtpu
#      picks the Ray distributed executor.
#   3. Runs the given command inside the head container (repo lives at
#      /root/torchtpu-vllm in the image).
#   4. Copies /perf_eval_results back to the workspace for artifact upload.
#
# These hosts have no attached data disk (100G boot disk only), so serve
# large models from GCS via --load-format runai_streamer (see MODEL_URI in
# scripts/vllm/benchmarking/run_benchmarks.sh) rather than an HF download.
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

if [ "$#" -lt 1 ]; then
  echo "ERROR: Usage: $0 <command_to_run_in_head_container...>"
  exit 1
fi

IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"
COMMIT_HASH="${BUILDKITE_COMMIT:-latest}"
IMAGE_TAG="${VLLM_TORCHTPU_IMAGE_TAG:-${IMAGE_REPO}:${COMMIT_HASH}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_CLUSTER="${REPO_DIR}/scripts/multihost/run_cluster.sh"

SSH_USER="${SSH_USER:-$(whoami)}"
if [ ! -f ~/.ssh/id_rsa ]; then
  echo "--- Auto-generating SSH key for passwordless auth"
  mkdir -p ~/.ssh
  ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa -q
fi
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o ConnectTimeout=15 -i ~/.ssh/id_rsa)

# Worker-host ssh sessions intermittently drop ("client_loop: send
# disconnect: Broken pipe", exit 255) — retry setup commands with backoff.
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
echo "Head IP: ${HEAD_INTERNAL_IP}  Worker IPs: ${WORKER_IPS}"
IFS=',' read -r -a WORKER_IPS_ARRAY <<< "${WORKER_IPS}"

# JAX_PLATFORMS="" prevents JAX from initializing a competing TPU PjRt client.
# Ray's TPU accelerator manager otherwise rewrites the slice bounds to 1,1,1
# for each one-TPU actor before TorchTPU initializes.

# ---------------------------------------------------------------------------
# Results dir on host (no /mnt/disks on these agents; boot disk only)
# ---------------------------------------------------------------------------
PERSIST_ROOT="${HOME}/persist"
rm -rf "${PERSIST_ROOT}/perf_eval_results"
mkdir -p "${PERSIST_ROOT}/perf_eval_results"
chmod 777 "${PERSIST_ROOT}/perf_eval_results" 2>/dev/null || true
HOST_HF_HOME="${HOME}/hf_home"   # tokenizer/config only; weights stream from GCS
mkdir -p "${HOST_HF_HOME}"
rm -rf perf_eval_results
mkdir -p perf_eval_results

# ---------------------------------------------------------------------------
# Cleanup on exit: dump server log, stop containers on all hosts
# ---------------------------------------------------------------------------
cleanup() {
  echo "--- Cleaning up Ray containers"
  for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker rm -f node >/dev/null 2>&1 || true" || true
  done
  docker rm -f node >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cleanup

# Free disk before pulling this commit's image (100G boot disk only).
echo "--- Cleaning up old Docker images"
bash "${SCRIPT_DIR}/cleanup_docker.sh" || true

# ---------------------------------------------------------------------------
# Start Ray head container locally (run_cluster.sh blocks; background it)
# ---------------------------------------------------------------------------
# Env for the benchmark flow inside the container mirrors run_in_docker.sh.
# VLLM_DISABLE_COMPILE_CACHE: every DP engine's workers persist their own
# AOT compile artifacts to the container disk (torch _dynamo
# atomic_write_binary), duplicating what the torch_tpu tier cache already
# holds in /dev/shm; at DP>=4 this fills the ~80G boot disk (dev builds
# 173/174). CI containers are ephemeral, so the on-disk cache buys nothing.
CONTAINER_ENV=(
  -e VLLM_DISABLE_COMPILE_CACHE=1
  -e HF_TOKEN="${HF_TOKEN:-}"
  -e TPU_MULTIHOST_BACKEND=ray
  -e JAX_PLATFORMS=""
  -e RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
  -e TPU_SKIP_MDS_QUERY=1
  -e BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-}"
  -e FORCE_COLOR="1"
  -e TQDM_MININTERVAL="30"
  -e SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0"
  -e BUILDKITE_BRANCH="${BUILDKITE_BRANCH:-}"
  -e BUILDKITE_PULL_REQUEST="${BUILDKITE_PULL_REQUEST:-}"
  ${TPU_ACCELERATOR_TYPE:+-e TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE}"}
  ${VLLM_ENGINE_READY_TIMEOUT_S:+-e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S}"}
  ${MODEL_URI:+-e MODEL_URI="${MODEL_URI}"}
  -v "${PERSIST_ROOT}/perf_eval_results:/perf_eval_results"
)

# Stray Ray state (crashed prior runs) makes a fresh `ray start --head`
# abort with "Session name ... does not match persisted value": something
# keeps serving GCS on 6379 across container removal. Nuke ALL containers
# on both hosts (CI hosts run nothing else) and sudo-kill stray processes
# (container remnants are root-owned; plain pkill cannot touch them).
echo "--- Pre-start scorched-earth cleanup"
docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1 || true
sudo -n pkill -9 -f 'gcs_server|raylet|ray::' 2>/dev/null || pkill -9 -f 'gcs_server|raylet|ray::' 2>/dev/null || true
echo "port 6379 holders on head (should be empty):"
ss -tlnp 2>/dev/null | grep 6379 || true
for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
    "docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1 || true; sudo -n pkill -9 -f 'gcs_server|raylet|ray::' 2>/dev/null || pkill -9 -f 'gcs_server|raylet|ray::' 2>/dev/null || true" || true
done

# Supervised head start: the container only exists once the image pull
# completes, and ray-start can die on residual state — bounded wait per
# attempt with a clean retry instead of one long hang.
start_head() {
  bash "${RUN_CLUSTER}" "${IMAGE_TAG}" "${HEAD_INTERNAL_IP}" --head "${HOST_HF_HOME}" \
    "${CONTAINER_ENV[@]}" &
}
head_up=0
for head_attempt in 1 2 3; do
  echo "--- Starting Ray head container (attempt ${head_attempt}/3)"
  docker rm -f node >/dev/null 2>&1 || true
  start_head
  deadline=$((SECONDS + 900))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if docker exec node ray status >/dev/null 2>&1; then
      head_up=1
      break 2
    fi
    sleep 10
  done
  echo "head container/ray not up within 15 minutes; retrying"
done
if [ "$head_up" != "1" ]; then
  echo "ERROR: head container did not come up after 3 attempts"
  exit 1
fi
echo "Head container is up."

# ---------------------------------------------------------------------------
# Start Ray worker containers over ssh
# ---------------------------------------------------------------------------
for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
  echo "--- Starting Ray worker on ${worker_ip}"
  # No `docker system prune -a` here: it deletes the cached image (forcing
  # a full re-pull every run) and its disk churn can starve sshd on the
  # worker; targeted container cleanup is enough.
  ssh_retry "${SSH_USER}@${worker_ip}" "gcloud auth configure-docker us-central1-docker.pkg.dev --quiet >/dev/null 2>&1 || true; docker rm -f node abtest >/dev/null 2>&1 || true; mkdir -p ~/multihost ~/hf_home"
  base64 < "${RUN_CLUSTER}" > /tmp/run_cluster.b64
  ssh_retry "${SSH_USER}@${worker_ip}" "base64 -d > ~/multihost/run_cluster.sh" < /tmp/run_cluster.b64
  # The run itself is long-lived; restart it (with a container pre-clean)
  # if the ssh session drops early.
  # shellcheck disable=SC2029
  (
    for _attempt in 1 2 3; do
      ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
        "docker rm -f node >/dev/null 2>&1 || true; bash ~/multihost/run_cluster.sh '${IMAGE_TAG}' '${HEAD_INTERNAL_IP}' --worker \"\$HOME/hf_home\" -e HF_TOKEN='${HF_TOKEN:-}' -e VLLM_DISABLE_COMPILE_CACHE=1 -e TPU_MULTIHOST_BACKEND=ray -e JAX_PLATFORMS='' -e RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1 -e TPU_SKIP_MDS_QUERY=1" \
        && break
      echo "worker run_cluster ssh dropped (attempt ${_attempt}/3); restarting in 15s"
      sleep 15
    done
  ) &
done

EXPECTED_NODES=$(( ${#WORKER_IPS_ARRAY[@]} + 1 ))
echo "--- Waiting for ${EXPECTED_NODES} Ray nodes to be alive"
deadline=$((SECONDS + 1800))
while :; do
  alive=$(docker exec node python3 -c "import ray; ray.init(address='auto', logging_level='error'); print(sum(1 for n in ray.nodes() if n['Alive']))" 2>/dev/null || echo 0)
  if [ "${alive:-0}" -ge "$EXPECTED_NODES" ]; then
    echo "Ray cluster complete: ${alive} nodes."
    break
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "ERROR: Ray cluster incomplete after 30 minutes (alive=${alive:-0}/${EXPECTED_NODES})"
    docker exec node ray status || true
    exit 1
  fi
  sleep 15
done
docker exec node ray status || true

# ---------------------------------------------------------------------------
# Run the benchmark command inside the head container
# ---------------------------------------------------------------------------
set +e
echo "--- Running command in head container: $*"
docker exec -w /root/torchtpu-vllm node bash -c '
  umask 000
  rm -rf /perf_eval_results/*
  "$@"
' -- "$@"
EXIT_CODE=$?
set -e

echo "--- Copying results back for artifact upload"
cp -r "${PERSIST_ROOT}/perf_eval_results"/* perf_eval_results/ 2>/dev/null || true
find perf_eval_results/ -type l ! -exec test -e {} \; -delete 2>/dev/null || true

# Ray worker stderr (the real error on engine bring-up failures) lives in
# the containers' Ray session logs — archive them from every host before the
# containers are torn down.
echo "--- Collecting Ray session logs"
docker exec node bash -c 'cd /tmp/ray/session_latest/logs 2>/dev/null && tar czf - .' \
  > perf_eval_results/ray_logs_head.tgz 2>/dev/null || true
widx=0
for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
  widx=$((widx + 1))
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
    "docker exec node bash -c 'cd /tmp/ray/session_latest/logs 2>/dev/null && tar czf - .'" \
    > "perf_eval_results/ray_logs_worker${widx}.tgz" 2>/dev/null || true
done

exit "${EXIT_CODE}"
