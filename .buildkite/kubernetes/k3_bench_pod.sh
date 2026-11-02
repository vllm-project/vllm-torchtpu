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

# One host of the Kimi-K3 megakernel kernel bench (kimi-k3-kernel-bench.yaml):
# four pods of one 2x2x4 slice, each starting eight `python -m tools.megakernel`
# ranks (SLURM_* env, TP32_HOST_IPS) per command of K3_BENCH_COMMANDS.
#
#   K3_BENCH_COMMANDS  ";;"-separated tools.megakernel argument lines, run in
#                      order; @OUT@ is the results directory
#   K3_BENCH_LIBTPU    flags appended to LIBTPU_INIT_ARGS (optional)
#   K3_BENCH_FRESH     1: compile every command afresh (no compilation caches)
#
# Pod 0 copies the results and rank logs into ARTIFACTS_DIR and uploads them.
set -uo pipefail

: "${K3_BENCH_COMMANDS:?}" "${ROLE_HOST_PREFIX:?}" "${ROLE_HOST_DOMAIN:?}" "${NUM_HOSTS:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE="${JOB_COMPLETION_INDEX:-0}"
PRESHARDED="${K3_BENCH_PRESHARDED:-/cache/hf/hub/presharded/kimi-k3-tp32}"

# shellcheck source=.buildkite/kubernetes/multihost_lib.sh
# shellcheck disable=SC1091
source "${HERE}/multihost_lib.sh"
pod_setup
unset TPU_VISIBLE_CHIPS
export MEGASCALE_NUM_SLICES='' MEGASCALE_SLICE_ID='' MEGASCALE_PORT='' MEGASCALE_COORDINATOR_ADDRESS=''
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR"
export HF_HUB_OFFLINE=1 MEGAKERNEL_TOOL_WORKER=1 TPU_STDERR_LOG_LEVEL=0 TPU_PARALLEL_PRECOMPILE=1
export PJRT_DEVICE=TPU VLLM_TARGET_DEVICE=tpu
mkdir -p /dev/shm/xla_cache
export VLLM_XLA_CACHE_PATH=/dev/shm/xla_cache

for _ in $(seq 1 60); do
  [ -f "$PRESHARDED/layout.json" ] && break
  sleep 5
done
[ -f "$PRESHARDED/layout.json" ] || { echo "FATAL: $PRESHARDED/layout.json not visible" >&2; exit 1; }

# The pods' addresses in index order: harness.init() reads TP32_HOST_IPS.
ips=""
for ((i = 0; i < NUM_HOSTS; i++)); do
  ip="$(resolve "${ROLE_HOST_PREFIX}${i}.${ROLE_HOST_DOMAIN}" 1800)" || exit 1
  ips="${ips:+$ips,}$ip"
done
export TP32_HOST_IPS="$ips"
echo "node $NODE: TP32_HOST_IPS=$TP32_HOST_IPS"

OUT=/dev/shm/k3-bench
mkdir -p "$OUT"
session="${ROLE_HOST_DOMAIN}"
k=0
status=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  line="${line//@OUT@/$OUT}"
  k=$((k + 1))
  echo "=== command $k on node $NODE: $line"
  export MASTER_PORT=$((29600 + k))
  export TORCH_TPU_BASE_PORT=$((8070 + 100 * k))
  # Identical on every pod: the pod set's handshake key.
  TP32_SESSION="$(printf '%s-%d' "$session" "$k" | cksum | cut -d' ' -f1)"
  export TP32_SESSION
  if [ "${K3_BENCH_FRESH:-0}" = 1 ]; then
    fresh="/dev/shm/fresh-cmd$k"; mkdir -p "$fresh"
    export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$fresh" TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="$fresh"
    export TORCH_TPU_TIER2_COMPILATION_CACHE="fresh-cmd$k" TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="fresh-cmd$k"
    export VLLM_XLA_CACHE_PATH="$fresh/xla"
  fi
  if [ -n "${K3_BENCH_LIBTPU:-}" ]; then
    export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }$K3_BENCH_LIBTPU"
  fi
  pids=()
  for l in 0 1 2 3 4 5 6 7; do
    r=$((NODE * 8 + l))
    # shellcheck disable=SC2086  # the command line is split into arguments
    SLURM_PROCID=$r SLURM_LOCALID=$l SLURM_NODEID=$NODE \
      python3 -m tools.megakernel --presharded "$PRESHARDED" --json "$OUT/results-cmd$k.jsonl" $line \
      > "$OUT/cmd$k-node$NODE-rank$(printf '%02d' "$r").log" 2>&1 &
    pids+=($!)
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "=== command $k on node $NODE finished rc=$rc"
  tail -n 40 "$OUT/cmd$k-node$NODE-rank$(printf '%02d' $((NODE * 8))).log"
  if [ "$rc" != 0 ]; then
    status=1
    grep -h -i -E "error|traceback|exception" "$OUT/cmd$k-node$NODE-rank"*.log | head -40
    break
  fi
done < <(printf '%s\n' "$K3_BENCH_COMMANDS" | sed 's/;;/\n/g')

if [ "$NODE" = 0 ] && [ -n "${ARTIFACTS_DIR:-}" ]; then
  echo "=== results"
  cat "$OUT"/results-cmd*.jsonl 2>/dev/null || true
  mkdir -p "$ARTIFACTS_DIR/k3-bench"
  # Results, saved tokens and rank-0 logs; traces and LLO dumps when written.
  cp -a "$OUT"/results-cmd*.jsonl "$ARTIFACTS_DIR/k3-bench/" 2>/dev/null
  cp -a "$OUT"/*.npz "$ARTIFACTS_DIR/k3-bench/" 2>/dev/null
  cp -a "$OUT"/cmd*-node0-rank00.log "$ARTIFACTS_DIR/k3-bench/" 2>/dev/null
  for d in "$OUT"/*/; do
    [ -d "$d" ] && cp -a "$d" "$ARTIFACTS_DIR/k3-bench/"
  done
  if ! publish_results && [ "$status" -eq 0 ]; then
    status=1
  fi
fi
exit "$status"
