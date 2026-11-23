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
SHARDING="${SHARDING:-DP8_EP}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
CONC="${CONC:-64}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen3.5-397B-A17B-FP8}"
case "$MODEL" in
  gs://*) LOAD_FORMAT="${LOAD_FORMAT:-runai_streamer}" ;;
esac
# Headroom over ISL+OSL (matches InferenceX qwen3.5: ISL+OSL+20)
MAX_MODEL_LEN_BUFFER="${MAX_MODEL_LEN_BUFFER:-20}"

case "$SHARDING" in
  DP8_EP)
    # Data-parallel attention across all 8 cores + expert-parallel MoE.
    DP_SIZE=8
    # New prefills are admitted every this many steps. Larger value trades
    # TTFT for throughput. Default 8 keeps TTFT low.
    PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-8}"
    SHARDING_ARGS=(
      --tensor-parallel-size=1
      --data-parallel-size=8
      --enable-expert-parallel
      --prefill-schedule-interval="$PREFILL_SCHEDULE_INTERVAL"
    )
    export DP_SCHED_ENABLED=1
    ;;
  TP8_EP)
    # Tensor-parallel attention across all 8 chips + expert-parallel MoE.
    DP_SIZE=1
    SHARDING_ARGS=(
      --tensor-parallel-size=8
      --data-parallel-size=1
      --enable-expert-parallel
      --mamba-ssm-cache-dtype="${GDN_STATE_DTYPE:-bfloat16}"
      --kernel-config='{"ir_op_priority":{"rms_norm":"native","fused_add_rms_norm":"native"}}'
    )
    export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-1}"
    export RAGGED_GATED_DELTA_RULE_IMPL="${RAGGED_GATED_DELTA_RULE_IMPL:-chunked_kernel_v3_pd}"
    export TPU_GDN_QKVZ_KERNEL_LAYOUT="${TPU_GDN_QKVZ_KERNEL_LAYOUT:-1}"
    export TPU_GDN_TABLES_IN_KERNEL="${TPU_GDN_TABLES_IN_KERNEL:-1}"
    export TPU_GDN_FUSED_NORM_QUANT="${TPU_GDN_FUSED_NORM_QUANT:-1}"
    export TPU_GDN_PREFILL_CHECKPOINT="${TPU_GDN_PREFILL_CHECKPOINT:-0}"
    export VLLM_TPU_FAST_TOKEN_SUBSTITUTION="${VLLM_TPU_FAST_TOKEN_SUBSTITUTION:-1}"
    export TPU_FUSED_QK_NORM_ROPE="${TPU_FUSED_QK_NORM_ROPE:-1}"
    export TPU_NORM_WEIGHT_HOIST="${TPU_NORM_WEIGHT_HOIST:-1}"
    export TPU_GREEDY_TOP_TOKENS="${TPU_GREEDY_TOP_TOKENS:-1}"
    export TPU_SMALL_ALL_REDUCE="${TPU_SMALL_ALL_REDUCE:-1}"
    export TPU_RPA_DECODE_BATCHED="${TPU_RPA_DECODE_BATCHED:-1}"
    export TPU_GATED_NORM_TILE="${TPU_GATED_NORM_TILE:-256}"
    export TPU_MOE_COUNTING_SORT="${TPU_MOE_COUNTING_SORT:-1}"
    export TPU_MOE_LOWTOK="${TPU_MOE_LOWTOK:-1}"
    export TPU_MOE_LOWTOK_FROM_SCORES="${TPU_MOE_LOWTOK_FROM_SCORES:-1}"
    export TPU_MOE_LOWTOK_SCORES_STRICT="${TPU_MOE_LOWTOK_SCORES_STRICT:-1}"
    export TPU_MOE_LOWTOK_QUANT_BLOCK="${TPU_MOE_LOWTOK_QUANT_BLOCK:-0}"
    export TPU_MOE_SEQUENCE_PARALLEL="${TPU_MOE_SEQUENCE_PARALLEL:-1}"
    export TPU_MOE_SEQUENCE_PARALLEL_FP8_GATHER="${TPU_MOE_SEQUENCE_PARALLEL_FP8_GATHER:-1}"
    export TPU_SP_MIN_ROWS="${TPU_SP_MIN_ROWS:-257}"
    export MOE_FUSED_EP_KERNEL_IMPL="${MOE_FUSED_EP_KERNEL_IMPL:-v3}"
    export MOE_FUSED_EP_SORTED_GATHER="${MOE_FUSED_EP_SORTED_GATHER:-1}"
    export MOE_KERNEL_THRESHOLD="${MOE_KERNEL_THRESHOLD:-256}"
    export TPU_TOKEN_BUCKET_EXTRA="${TPU_TOKEN_BUCKET_EXTRA-4,8,48,4608,5120,5632,6144,6656,7168,7680,8704,9216,9728}"
    export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144 --xla_tpu_dvfs_p_state=7 --xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers=${XLA_ALT_SCALE:-NO_SCALE}"
    if [ -n "${XLA_SCOPED_VMEM_KIB:-}" ]; then
      export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS} --xla_tpu_scoped_vmem_limit_kib=${XLA_SCOPED_VMEM_KIB}"
    fi
    ;;
  DP4TP2_EP)
    # Data-parallel attention across all 4 chips with tensor-parallel across 2 cores
    # on the same chip + expert-parallel MoE.
    DP_SIZE=4
    # New prefills are admitted every this many steps. Larger value trades
    # TTFT for throughput. Default 8 keeps TTFT low.
    PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-8}"
    SHARDING_ARGS=(
      --tensor-parallel-size=2
      --data-parallel-size=4
      --enable-expert-parallel
      --prefill-schedule-interval="$PREFILL_SCHEDULE_INTERVAL"
    )
    export DP_SCHED_ENABLED=1
    ;;
  *) echo "ERROR: unknown SHARDING='$SHARDING' (DP8_EP, TP8_EP, DP4TP2_EP are wired up)" >&2; exit 1 ;;
esac

MAX_MODEL_LEN=$((ISL + OSL + MAX_MODEL_LEN_BUFFER))

# Global num of batched tokens == max(ISL, GLOBAL_BATCHED_TOKENS_MIN).
GLOBAL_BATCHED_TOKENS_MIN="${GLOBAL_BATCHED_TOKENS_MIN:-16384}"
GLOBAL_BATCHED_TOKEN=$(( ISL > GLOBAL_BATCHED_TOKENS_MIN ? ISL : GLOBAL_BATCHED_TOKENS_MIN ))

MAX_NUM_BATCHED_TOKENS=$(((GLOBAL_BATCHED_TOKEN + DP_SIZE - 1) / DP_SIZE))

MAX_NUM_SEQS=$((CONC / DP_SIZE))
[ "$MAX_NUM_SEQS" -lt 1 ] && MAX_NUM_SEQS=1

# Increase API-server frontend wait time since cold init may take long.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

export TPU_ACCELERATOR_TYPE=tpu7x
export USE_MOE_SPARSE_CORE=1
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"

# Default to v3 which is a bit faster than v2.
export RAGGED_GATHER_REDUCE_VERSION="${RAGGED_GATHER_REDUCE_VERSION:-v3}"

# Add extra padding
# 4, 8 for low concurrency 4 and 8
# 48 for concurrency 256 when DP8+EP is enabled to reduce padding from 32->64.
# Set it to empty to opt out of the extra buckets.
export TPU_TOKEN_BUCKET_EXTRA="${TPU_TOKEN_BUCKET_EXTRA-4,8,48}"

# Shrink the rotary cos_sin caches to max_model_len, text-only, to minimize
# xla layout data copy overhead.
export TPU_ROPE_CACHE_TRUNCATE="${TPU_ROPE_CACHE_TRUNCATE:-1}"

# Skip padded tokens in the fused MoE so they activate no experts.
export TPU_MOE_SKIP_PADDED_TOKENS=1

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
  "${SHARDING_ARGS[@]}"
)
# Pass --load-format when streaming from GCS.
[ -n "${LOAD_FORMAT:-}" ] && args+=(--load-format="$LOAD_FORMAT")

set -x
vllm serve "${args[@]}"
set +x
