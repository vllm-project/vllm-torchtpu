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

set -euo pipefail

if [ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

PORT="${PORT:-8000}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
CONC="${CONC:-64}"
SHARDING="${SHARDING:-DP8_EP}"

MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen3.5-397B-A17B-FP8}"
case "$MODEL" in
  gs://*) LOAD_FORMAT="${LOAD_FORMAT:-runai_streamer}" ;;
esac
MAX_MODEL_LEN=$((ISL + OSL + 1024))

case "$SHARDING" in
  DP8_EP)
    DP_SIZE=8
    SHARDING_ARGS=(
      --tensor-parallel-size=1
      --data-parallel-size=8
      --enable-expert-parallel
    )
    export DP_SCHED_ENABLED=0
    export USE_MOE_FUSED_EP_KERNEL=1
    ;;
  TP8_EP)
    DP_SIZE=1
    SHARDING_ARGS=(
      --tensor-parallel-size=8
      --data-parallel-size=1
      --enable-expert-parallel
    )
    export USE_MOE_FUSED_EP_KERNEL=0
    export TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS="${TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS:-1024}"
    case "$TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS" in
      ''|*[!0-9]*) echo "ERROR: invalid hierarchical all-reduce threshold" >&2; exit 1 ;;
    esac
    ;;
  *) echo "ERROR: unknown SHARDING=$SHARDING" >&2; exit 1 ;;
esac

if [ "$USE_MOE_FUSED_EP_KERNEL" = "1" ]; then
  export TPU_MOE_FUSED_EP_STEP_MIN_TOKENS="${TPU_MOE_FUSED_EP_STEP_MIN_TOKENS:-33}"
  export TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER="${TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER:-1}"
  export MOE_FUSED_EP_V2_ROWBLK="${MOE_FUSED_EP_V2_ROWBLK:-1}"
  export MOE_FUSED_EP_V2_SKIP_PADDED_ROWS="${MOE_FUSED_EP_V2_SKIP_PADDED_ROWS:-1}"
fi

GLOBAL_MAX_NUM_BATCHED_TOKENS="${GLOBAL_MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$(( (GLOBAL_MAX_NUM_BATCHED_TOKENS + DP_SIZE - 1) / DP_SIZE ))}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(( (CONC + DP_SIZE - 1) / DP_SIZE ))}"
PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-1}"
[ "$MAX_NUM_SEQS" -lt 1 ] && MAX_NUM_SEQS=1
SHARDING_ARGS+=(--prefill-schedule-interval="$PREFILL_SCHEDULE_INTERVAL")

# Apply tuned runtime defaults.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

export TPU_ACCELERATOR_TYPE=tpu7x
# Use tuned compiler and kernel settings.
export USE_MOE_SPARSE_CORE=1
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-2048}"

DEFAULT_LIBTPU_INIT_ARGS=" --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false"

# Use hierarchical all-reduce for large TP activations.
if [ "$SHARDING" = "TP8_EP" ] && [ "$TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS" -gt 0 ]; then
  TP_HIERARCHICAL_AR_MIN_BYTES=$((TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS * 4096 * 2))
  DEFAULT_LIBTPU_INIT_ARGS+=" --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=$TP_HIERARCHICAL_AR_MIN_BYTES"
fi

# Allow the fused row gather to overlap routing.
if [ "${TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER:-0}" = "1" ]; then
  DEFAULT_LIBTPU_INIT_ARGS+=" --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144"
fi

export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS-$DEFAULT_LIBTPU_INIT_ARGS}"

export TPU_TOKEN_BUCKET_EXTRA="${TPU_TOKEN_BUCKET_EXTRA-4,8,48,3072}"

export TPU_ROPE_CACHE_TRUNCATE="${TPU_ROPE_CACHE_TRUNCATE:-1}"

export TPU_MOE_SKIP_PADDED_TOKENS=1

export RAGGED_GATHER_REDUCE_VERSION="${RAGGED_GATHER_REDUCE_VERSION:-v3}"

args=(
  "$MODEL"
  --served-model-name="$SERVED_NAME"
  --max-model-len="$MAX_MODEL_LEN"
  --max-num-batched-tokens="$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs="$MAX_NUM_SEQS"
  --no-enable-prefix-caching
  --gpu-memory-utilization=0.92
  --block-size=256
  --async-scheduling
  --port="$PORT"
  --language-model-only
  --limit-mm-per-prompt='{"image": 0, "video": 0}'
  --quantization=fp8
  --kv-cache-dtype=fp8
  --attention-backend CUSTOM
  --api-server-count=1
  "${SHARDING_ARGS[@]}"
)
[ -n "${LOAD_FORMAT:-}" ] && args+=(--load-format="$LOAD_FORMAT")


set -x
vllm serve "${args[@]}"
set +x
