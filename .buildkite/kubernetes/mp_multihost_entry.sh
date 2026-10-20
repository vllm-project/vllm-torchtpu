#!/usr/bin/env bash
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

# Serves a benchmark config across both hosts of a slice without Ray.
#
#   mp_multihost_entry.sh <config_name> [run_eval_flow args...]
#
# Every host runs this with the same arguments and decides its own role from
# JOB_COMPLETION_INDEX; nothing outside the pods orchestrates them. Each pod starts its
# own half.
#
# vLLM's own multi-node DP is the mechanism: one `vllm serve` per host, all
# naming the same --data-parallel-address, the secondary hosts --headless with
# a --data-parallel-start-rank. The ranks find each other over that address,
# and src/vllm_torchtpu/distributed/tpu_mp_multihost.py turns the result into
# TORCH_TPU_SLICEBUILDER_ADDRESSES.
#
# Environment, from manifests/workloads/mp-multihost-slice.yaml:
#   HEAD_HOST   DNS name of index 0, which every host addresses.
#   NUM_HOSTS   hosts in the slice.
#   DP_LOCAL    DP replicas this host owns; DP_LOCAL x NUM_HOSTS must equal the
#               config's DATA_PARALLELISM.
set -uo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <config_name> [run_eval_flow args...]" >&2
  exit 2
fi
CONFIG_NAME="$1"
shift

: "${HEAD_HOST:?HEAD_HOST must name index 0 of the slice}"
: "${NUM_HOSTS:?NUM_HOSTS must be set}"
: "${DP_LOCAL:?DP_LOCAL must be the DP replicas owned by this host}"
INDEX="${JOB_COMPLETION_INDEX:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.buildkite/kubernetes/multihost_lib.sh
# shellcheck disable=SC1091
source "${HERE}/multihost_lib.sh"
pod_setup
use_mp_backend

PORT="${PORT:-8000}"
DP_RPC_PORT="${DP_RPC_PORT:-29500}"

# An address, not a name: index 0 *binds* a TCPStore here, and every host is
# handed the same value - the head's.
DP_ADDRESS="$(resolve "$HEAD_HOST" 300)" || exit 1
echo "--- host ${INDEX}/${NUM_HOSTS}: data-parallel address ${DP_ADDRESS}"

# Defaults for what a config may leave unset, applied before sourcing it.
MODEL_URI="${MODEL_URI:-}"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ENABLE_EP=false
QUANTIZATION=""
GPU_MEMORY_UTILIZATION="0.95"
KV_CACHE_DTYPE="fp8"
MAX_MODEL_LEN=16384
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=512
ATTENTION_BACKEND="CUSTOM"
ENABLE_PREFIX_CACHING=false
ASYNC_SCHEDULING=true
EXTRA_SERVE_ARGS=""
SERVER_READY_WAIT_MIN=180
CONFIG_FILE="${REPO_DIR}/scripts/vllm/benchmarking/configs/${CONFIG_NAME}.sh"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: config file not found: $CONFIG_FILE" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONFIG_FILE"

mp_serve_args || exit 1

if [ "$INDEX" = "0" ]; then
  echo "--- Starting the head server (DP ${DP_LOCAL} of ${DATA_PARALLELISM})"
  # One API frontend, not one per DP rank; see run_benchmarks.sh. Head only -
  # the headless host starts none and vLLM rejects the flag there.
  head_args=()
  [ "$DATA_PARALLELISM" -gt 1 ] && head_args+=(--api-server-count=1)
  vllm serve "${SERVE_ARGS[@]}" "${head_args[@]}" --host 0.0.0.0 --port "${PORT}" &
  serve_pid=$!

  echo "--- Waiting for /health, up to ${SERVER_READY_WAIT_MIN} minutes"
  deadline=$((SECONDS + SERVER_READY_WAIT_MIN * 60))
  ready=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    # The server answers only once every DP rank on every host has joined, so
    # this is also the wait for the other host.
    if curl -sf -o /dev/null --connect-timeout 2 "http://localhost:${PORT}/health"; then
      ready=1
      break
    fi
    # No point waiting out the deadline against a server that has died.
    if ! kill -0 "$serve_pid" 2>/dev/null; then
      echo "ERROR: the head server exited before it became healthy"
      break
    fi
    sleep 10
  done
  if [ "$ready" != "1" ]; then
    echo "ERROR: head server did not become healthy"
    kill "$serve_pid" 2>/dev/null || true
    exit 1
  fi
  echo "Head server is healthy."

  bash ./scripts/vllm/benchmarking/run_eval_flow.sh \
    --config "$CONFIG_NAME" --host localhost --port "$PORT" \
    --results-dir "${ARTIFACTS_DIR:-/tmp/perf_eval}/${CONFIG_NAME}" "$@"
  rc=$?

  kill "$serve_pid" 2>/dev/null || true
  # A failed upload decides the step only when the work itself passed.
  if ! publish_results && [ "$rc" -eq 0 ]; then
    rc=1
  fi
  exit "$rc"
fi

mp_headless_worker
exit $?
