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

# Backend implementation for Ray multi-host distributed execution.
# Can be sourced by run_multihost.sh or executed directly as a wrapper.
# shellcheck disable=SC2154

export MULTIHOST_RUN_LOG="${MULTIHOST_RUN_LOG:-perf_eval_results/multihost.log}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If executed directly instead of sourced, delegate to run_multihost.sh --backend ray
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  exec bash "${SCRIPT_DIR}/run_multihost.sh" --backend ray "$@"
fi

run_ray_multihost() {
  CONTAINER_ENV_HEAD+=(
    -e JAX_PLATFORMS=""
    -e RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
  )

  start_head() {
    bash "${RUN_CLUSTER}" "${IMAGE_TAG}" "${HEAD_INTERNAL_IP}" --head "${HOST_HF_HOME}" \
      "${CONTAINER_ENV_HEAD[@]}" &
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
    return 1
  fi
  echo "Head container is up."

  # Start Ray worker containers over ssh
  for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
    echo "--- Starting Ray worker on ${worker_ip}"
    ssh_retry "${SSH_USER}@${worker_ip}" "gcloud auth configure-docker us-central1-docker.pkg.dev --quiet >/dev/null 2>&1 || true; docker rm -f node abtest >/dev/null 2>&1 || true; mkdir -p ~/multihost ~/hf_home" || return 1
    base64 < "${RUN_CLUSTER}" > /tmp/run_cluster.b64 || return 1
    ssh_retry "${SSH_USER}@${worker_ip}" "base64 -d > ~/multihost/run_cluster.sh" < /tmp/run_cluster.b64 || return 1
    # shellcheck disable=SC2029
    (
      for _attempt in 1 2 3; do
        ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
          "docker rm -f node >/dev/null 2>&1 || true; bash ~/multihost/run_cluster.sh '${IMAGE_TAG}' '${HEAD_INTERNAL_IP}' --worker \"\$HOME/hf_home\" -e HF_TOKEN='${HF_TOKEN:-}' -e VLLM_DISABLE_COMPILE_CACHE=1 -e TPU_MULTIHOST_BACKEND=ray -e JAX_PLATFORMS='' -e RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1 -e TPU_SKIP_MDS_QUERY=1${CONTAINER_TMPDIR:+ -e TMPDIR='${CONTAINER_TMPDIR}' -e RAY_TMPDIR=/tmp}" \
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
      return 1
    fi
    sleep 15
  done
  docker exec node ray status || true

  # Run the benchmark command inside the head container
  set +e
  echo "--- Running command in head container: $*"
  docker exec -w /root/torchtpu-vllm node bash -c '
    umask 000
    rm -rf /perf_eval_results/*
    "$@"
  ' -- "$@" 2>&1 | tee "${MULTIHOST_RUN_LOG}"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  echo "~~~ Collecting Ray session logs"
  docker exec node bash -c 'cd /tmp/ray/session_latest/logs 2>/dev/null && tar czf - .' \
    > perf_eval_results/ray_logs_head.tgz 2>/dev/null || true
  widx=0
  for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
    widx=$((widx + 1))
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
      "docker exec node bash -c 'cd /tmp/ray/session_latest/logs 2>/dev/null && tar czf - .'" \
      > "perf_eval_results/ray_logs_worker${widx}.tgz" 2>/dev/null || true
  done

  return "${EXIT_CODE}"
}
