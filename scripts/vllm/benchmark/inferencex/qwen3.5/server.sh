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

# Optionally activate a venv.
if [ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

PORT="${PORT:-8000}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
CONC="${CONC:-64}"
# Use the shared sweep sharding, with DP8_EP as the fallback.
# shellcheck source=./sharding_table.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sharding_table.sh"
SHARDING="${SHARDING:-$(sharding_for "$ISL" "$OSL" "$CONC" || echo DP8_EP)}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen3.5-397B-A17B-FP8}"
case "$MODEL" in
  gs://*) LOAD_FORMAT="${LOAD_FORMAT:-runai_streamer}" ;;
esac
# Keep modest context headroom without widening attention tables.
MAX_MODEL_LEN_BUFFER="${MAX_MODEL_LEN_BUFFER:-1024}"

# Empty values fall back to the generic scheduler settings below.
SHARDING_MAX_NUM_SEQS=""
SHARDING_MAX_NUM_BATCHED_TOKENS=""
SHARDING_PREFILL_SCHEDULE_INTERVAL=""

case "$SHARDING" in
  DP8_EP)
    # Data-parallel attention across all 8 cores + expert-parallel MoE.
    DP_SIZE=8
    # Use a default interval unless the point table overrides it.
    SHARDING_PREFILL_SCHEDULE_INTERVAL=2
    SHARDING_ARGS=(
      --tensor-parallel-size=1
      --data-parallel-size=8
      --enable-expert-parallel
    )
    # Disable the locally ported coordinated scheduler for this reproduction.
    export DP_SCHED_ENABLED=0
    export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-1}"
    # Use fused EP for prefill buckets and GMM dispatch for decode.
    if [ "$USE_MOE_FUSED_EP_KERNEL" = "1" ]; then
      export TPU_MOE_FUSED_EP_STEP_MIN_TOKENS="${TPU_MOE_FUSED_EP_STEP_MIN_TOKENS:-33}"
      # Overlap the activation-row gather with routing work.
      export TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER="${TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER:-1}"
    fi
    ;;
  TP8_EP)
    # Tensor-parallel attention across all 8 chips + expert-parallel MoE.
    DP_SIZE=1
    # Match sequence slots to concurrency and admit two 8k prefills.
    SHARDING_MAX_NUM_SEQS="$CONC"
    SHARDING_MAX_NUM_BATCHED_TOKENS=16384
    SHARDING_PREFILL_SCHEDULE_INTERVAL=1
    SHARDING_ARGS=(
      --tensor-parallel-size=8
      --data-parallel-size=1
      --enable-expert-parallel
    )
    # The fused EP kernel refuses tensor parallelism and cannot help here.
    export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-0}"
    ;;
  DP4TP2_EP)
    # Combine four-way DP with two-way TP and expert parallelism.
    DP_SIZE=4
    SHARDING_PREFILL_SCHEDULE_INTERVAL=256
    SHARDING_ARGS=(
      --tensor-parallel-size=2
      --data-parallel-size=4
      --enable-expert-parallel
    )
    # Keep the coordinated DP scheduler disabled for reproduction runs.
    export DP_SCHED_ENABLED=0
    # The fused EP kernel also refuses the TP2 component of this sharding.
    export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-0}"
    ;;
  *) echo "ERROR: unknown SHARDING='$SHARDING' (DP8_EP, TP8_EP, DP4TP2_EP are wired up)" >&2; exit 1 ;;
esac

# Keep gather-then-route disabled unless explicitly requested.

MAX_MODEL_LEN=$((ISL + OSL + MAX_MODEL_LEN_BUFFER))

# Set the fallback node-wide prefill token target.
GLOBAL_BATCHED_TOKENS_MIN="${GLOBAL_BATCHED_TOKENS_MIN:-8192}"
GLOBAL_BATCHED_TOKEN=$(( ISL > GLOBAL_BATCHED_TOKENS_MIN ? ISL : GLOBAL_BATCHED_TOKENS_MIN ))

# Derive untuned limits from the workload and DP size.
DEFAULT_MAX_NUM_BATCHED_TOKENS="${SHARDING_MAX_NUM_BATCHED_TOKENS:-$(( (GLOBAL_BATCHED_TOKEN + DP_SIZE - 1) / DP_SIZE ))}"
DEFAULT_MAX_NUM_SEQS="${SHARDING_MAX_NUM_SEQS:-$(( (CONC + DP_SIZE - 1) / DP_SIZE ))}"
DEFAULT_PREFILL_SCHEDULE_INTERVAL="$SHARDING_PREFILL_SCHEDULE_INTERVAL"

# Use measured scheduler settings for tuned DP8 points.
if [ "$ISL:$OSL" = "8192:1024" ]; then
  case "$SHARDING:$CONC" in
    # Avoid prefill throttling and unused sequence slots at c128.
    DP8_EP:128) REFERENCE_POINT="16    256   1" ;;
    # Balance TTFT and throughput with a wider c256 prefill chunk.
    DP8_EP:256) REFERENCE_POINT="32    320   2" ;;
    *)          REFERENCE_POINT="" ;;
  esac
  if [ -n "$REFERENCE_POINT" ]; then
    read -r DEFAULT_MAX_NUM_SEQS DEFAULT_MAX_NUM_BATCHED_TOKENS \
      DEFAULT_PREFILL_SCHEDULE_INTERVAL <<<"$REFERENCE_POINT"
  fi
fi

MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$DEFAULT_MAX_NUM_BATCHED_TOKENS}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$DEFAULT_MAX_NUM_SEQS}"
# Treat an empty override as unset.
PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-$DEFAULT_PREFILL_SCHEDULE_INTERVAL}"
[ "$MAX_NUM_SEQS" -lt 1 ] && MAX_NUM_SEQS=1
if [ -n "$PREFILL_SCHEDULE_INTERVAL" ]; then
  SHARDING_ARGS+=(--prefill-schedule-interval="$PREFILL_SCHEDULE_INTERVAL")
fi

# Increase API-server frontend wait time since cold init may take long.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

export TPU_ACCELERATOR_TYPE=tpu7x
export USE_MOE_SPARSE_CORE=1
# Use one-hot permutation only for small routed batches.
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-2048}"

# Apply the XLA collective settings used by tuned runs.
DEFAULT_LIBTPU_INIT_ARGS=" --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false"

# Offload the large row gather so it can overlap routing.
if [ "${TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER:-0}" = "1" ]; then
  DEFAULT_LIBTPU_INIT_ARGS+=" --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144"
fi

# Set to empty to serve with stock XLA defaults.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS-$DEFAULT_LIBTPU_INIT_ARGS}"

# Add finer buckets for small decode and mid-size prefill shapes.
export TPU_TOKEN_BUCKET_EXTRA="${TPU_TOKEN_BUCKET_EXTRA-4,8,48,3072}"

# Shrink the rotary cos_sin caches to max_model_len, text-only, to minimize
# xla layout data copy overhead.
export TPU_ROPE_CACHE_TRUNCATE="${TPU_ROPE_CACHE_TRUNCATE:-1}"

# Skip padded routes in every MoE execution path.
export TPU_MOE_SKIP_PADDED_TOKENS=1

# Measured on this benchmark at 8k1k c256 DP8_EP (2560 prompts): 29,163 ->
# 31,101 total tok/s, median TPOT 60.15 -> 56.21 ms, GSM8K 0.980 strict.
#
# Destination-major SparseCore combine. The MoE combine is ~13% of device time
# at prefill and v3 roughly halves it.
export RAGGED_GATHER_REDUCE_VERSION="${RAGGED_GATHER_REDUCE_VERSION:-v3}"
# One Pallas call for gmm1 -> activation -> gmm2, intermediate held in VMEM.
# Bit-exact against the two-call path; -8% on the MoE layer at the decode shape.
export USE_FUSED_MOE_GMM="${USE_FUSED_MOE_GMM:-1}"
# Stacked scheduling in the batched-RPA kernel. Worth +1.6% end to end on a
# mixed batch even though every fixed-shape microbenchmark says it is slower --
# the win is in how it schedules a mixed batch, not in any single shape.
# Leave USE_BATCHED_RPA_SEQ_ON_LANE off: that layout is slower here and, unlike
# this one, is not bit-identical to the mainline kernel.
export USE_BATCHED_RPA_LONGCTX="${USE_BATCHED_RPA_LONGCTX:-1}"
# Reuse empty KV packing lanes to reduce TP8 cache traffic.
export TPU_RPA_FOLD_KV_HEAD_DIM="${TPU_RPA_FOLD_KV_HEAD_DIM:-1}"

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
  # Use one frontend to reduce host contention and retain stats.
  --api-server-count=1
  "${SHARDING_ARGS[@]}"
)
# Pass --load-format when streaming from GCS.
[ -n "${LOAD_FORMAT:-}" ] && args+=(--load-format="$LOAD_FORMAT")

# Word-split extra flags (profiler config, one-off overrides) onto the command
# line. Matches how run_benchmarks.sh consumes EXTRA_SERVE_ARGS.
if [ -n "${EXTRA_SERVE_ARGS:-}" ]; then
  read -r -a extra_serve_args <<< "$EXTRA_SERVE_ARGS"
  args+=("${extra_serve_args[@]}")
fi

# Print the resolved configuration for sweep server reuse.
if [ -n "${PRINT_CONFIG:-}" ]; then
  echo "${args[*]}" \
    "USE_MOE_FUSED_EP_KERNEL=${USE_MOE_FUSED_EP_KERNEL:-}" \
    "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS=${TPU_MOE_FUSED_EP_STEP_MIN_TOKENS:-}" \
    "TPU_MOE_SKIP_PADDED_TOKENS=${TPU_MOE_SKIP_PADDED_TOKENS:-}" \
    "TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER=${TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER:-}"
  exit 0
fi

set -x
vllm serve "${args[@]}"
set +x
