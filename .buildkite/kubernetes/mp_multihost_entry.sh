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
# JOB_COMPLETION_INDEX; nothing outside the pods orchestrates them. This is the
# `mp` half of what run_multihost.sh dispatches on bare metal, where the head
# ssh's to the other VM and starts a container - there is no second VM to reach
# here, so each pod starts its own half.
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
REPO_DIR="${REPO_DIR:-/workspace/vllm-torchtpu}"
cd "$REPO_DIR" || { echo "$0: no such directory: $REPO_DIR" >&2; exit 2; }

PORT="${PORT:-8000}"
DP_RPC_PORT="${DP_RPC_PORT:-29500}"

# GKE hands every TPU pod the environment of a slice member, and this backend
# does not want it: tpu_mp_multihost.py builds the slice itself, publishing one
# TORCH_TPU_SLICEBUILDER_ADDRESSES entry per chip from a TCPStore rendezvous
# among the DP ranks. Left set, the injected coordinator is a second, contrary
# answer to the same question. Unlike single_host_engine.sh this does not pin a
# topology: the run really does span two hosts, and the backend computes
# TORCH_TPU_TOPOLOGY from the chip count.
export MEGASCALE_COORDINATOR_ADDRESS=''
export MEGASCALE_NUM_SLICES=''
export MEGASCALE_PORT=''
export MEGASCALE_SLICE_ID=''
export TPU_PROCESS_ADDRESSES=''
export TPU_PROCESS_PORT=''
export TPU_WORKER_HOSTNAMES=''

export TPU_MULTIHOST_BACKEND=mp
# In a pod the metadata server describes the node, not the workload.
export TPU_SKIP_MDS_QUERY=1

# Core dumps are gigabytes each and fill the node's ephemeral storage.
ulimit -c 0

[ -n "${ARTIFACTS_DIR:-}" ] && mkdir -p "$ARTIFACTS_DIR"

# An address, not a name: index 0 *binds* a TCPStore here, and every host is
# handed the same value - the head's - exactly as on bare metal. DNS for a
# headless service can lag the pod by a few seconds.
resolve_head() {
  local deadline=$((SECONDS + 300)) ip
  while [ "$SECONDS" -lt "$deadline" ]; do
    ip="$(getent hosts "$HEAD_HOST" | awk '{print $1; exit}')"
    if [ -n "$ip" ]; then echo "$ip"; return 0; fi
    sleep 5
  done
  echo "ERROR: ${HEAD_HOST} did not resolve within 5 minutes" >&2
  return 1
}
DP_ADDRESS="$(resolve_head)" || exit 1
echo "--- host ${INDEX}/${NUM_HOSTS}: data-parallel address ${DP_ADDRESS}"

# The defaults run_multihost_mp.sh applies before sourcing, so a config written
# for that path behaves the same here.
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
EXTRA_SERVE_ARGS=""
SERVER_READY_WAIT_MIN=180
CONFIG_FILE="${REPO_DIR}/scripts/vllm/benchmarking/configs/${CONFIG_NAME}.sh"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: config file not found: $CONFIG_FILE" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONFIG_FILE"

if [ "$((DP_LOCAL * NUM_HOSTS))" -ne "$DATA_PARALLELISM" ]; then
  echo "ERROR: DP_LOCAL ($DP_LOCAL) x NUM_HOSTS ($NUM_HOSTS) != DATA_PARALLELISM ($DATA_PARALLELISM)" >&2
  exit 1
fi

extra_args=""
[ "$ENABLE_EP" = "true" ] && extra_args="--enable-expert-parallel"
[ -n "$QUANTIZATION" ] && extra_args="$extra_args --quantization $QUANTIZATION"
extra_args="$extra_args --attention-backend $ATTENTION_BACKEND"
[ -n "$EXTRA_SERVE_ARGS" ] && extra_args="$extra_args $EXTRA_SERVE_ARGS"
prefix_caching_flag="--no-enable-prefix-caching"
[ "$ENABLE_PREFIX_CACHING" = "true" ] && prefix_caching_flag="--enable-prefix-caching"

serve_target="$MODEL"
if [ -n "$MODEL_URI" ]; then
  serve_target="$MODEL_URI"
  extra_args="$extra_args --load-format runai_streamer --served-model-name $MODEL"
fi

# shellcheck disable=SC2206
SERVE_ARGS=(
  "${serve_target}"
  --tensor-parallel-size="${TENSOR_PARALLELISM}"
  --data-parallel-size="${DATA_PARALLELISM}"
  --data-parallel-size-local="${DP_LOCAL}"
  --data-parallel-address="${DP_ADDRESS}"
  --data-parallel-rpc-port="${DP_RPC_PORT}"
  --max-model-len="${MAX_MODEL_LEN}"
  --max-num-batched-tokens="${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs="${MAX_NUM_SEQS}"
  --async-scheduling
  "${prefix_caching_flag}"
  --gpu-memory-utilization="${GPU_MEMORY_UTILIZATION}"
  --kv-cache-dtype="${KV_CACHE_DTYPE}"
  ${extra_args}
)

# Bounded, because the probe has to be able to fail: a Completed pod stays a
# ready endpoint of the headless service, so this keeps resolving after the
# head exits - to an address that drops rather than refuses, and an unbounded
# connect would hang there holding chips. Same reason as multihost_entry.sh.
head_listening() {
  timeout 5 bash -c "exec 3<>/dev/tcp/${DP_ADDRESS}/${PORT}" 2>/dev/null
}

publish_results() {
  [ -n "${ARTIFACTS_DIR:-}" ] || return 0
  [ -n "$(ls -A "$ARTIFACTS_DIR" 2>/dev/null)" ] || return 0
  if ! buildkite-agent artifact upload "$ARTIFACTS_DIR/**/*"; then
    echo "ERROR: artifacts were produced but could not be uploaded"
    return 1
  fi
}

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

start_rank=$((INDEX * DP_LOCAL))
echo "--- Starting headless host ${INDEX} at data-parallel-start-rank=${start_rank}"
vllm serve "${SERVE_ARGS[@]}" --headless --data-parallel-start-rank="${start_rank}" &
serve_pid=$!

# Not waited on: --headless never returns, and nothing here kills it. The head
# finishing is what ends the run, so this watches for that and leaves - a
# blocking worker would hold its chips until the JobSet deadline.
echo "host ${INDEX}: waiting for the head to serve"
deadline=$((SECONDS + SERVER_READY_WAIT_MIN * 60))
while [ "$SECONDS" -lt "$deadline" ]; do
  head_listening && break
  if ! kill -0 "$serve_pid" 2>/dev/null; then
    echo "ERROR: this host's server exited before the head came up"
    exit 1
  fi
  sleep 10
done

echo "host ${INDEX}: head is up; waiting for it to finish"
# Three consecutive refusals, not one: a DNS blip on the headless service looks
# identical to a closed port from here, and leaving is irreversible.
misses=0
while [ "$misses" -lt 3 ]; do
  if head_listening; then misses=0; else misses=$((misses + 1)); fi
  sleep 15
done
echo "host ${INDEX}: head is gone; exiting"
kill "$serve_pid" 2>/dev/null || true
exit 0
