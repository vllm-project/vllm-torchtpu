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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
CONC="${CONC:-64}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
# The point's own settings (layout, slots, budget, release group, decode batch, placement) come from the point
# table, for a standalone launch as for a sweep; anything already set in the environment wins.
source "$SCRIPT_DIR/points.sh"
apply_point_defaults "$ISL" "$OSL" "$CONC"
SHARDING="${SHARDING:-DP8_EP}"
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen3.5-397B-A17B-FP8}"
case "$MODEL" in
  gs://*) LOAD_FORMAT="${LOAD_FORMAT:-runai_streamer}" ;;
esac
EXTRA_ARGS=()
# One threshold between the two MoE kernels, in rows per MoE call across the
# node: at or below it the few-token kernel, above it the fused
# expert-parallel kernel. Each layout sets its kernel settings from it. The
# few-token kernel's vector memory holds up to 337 rows of this model.
MOE_KERNEL_THRESHOLD="${MOE_KERNEL_THRESHOLD:-256}"
# Expert parallelism over the eight cores on both lines: the fused kernel's
# gate is the first row count per engine (or rank) above the threshold.
EP_SIZE=8

case "$SHARDING" in
  DP8_EP)
    # Data-parallel attention across all 8 cores + expert-parallel MoE, with
    # per-concurrency settings in points.sh.
    DP_SIZE=8
    # New prefills are admitted every step; the router's release rules below
    # line the engines' prompt steps up instead of a longer interval.
    PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-1}"
    SHARDING_ARGS=(
      --tensor-parallel-size=1
      --data-parallel-size=8
      --enable-expert-parallel
      --prefill-schedule-interval="$PREFILL_SCHEDULE_INTERVAL"
      # One API-server process, so the router's release rules below see
      # every arrival (vLLM would start one per engine).
      --api-server-count=1
    )
    export DP_SCHED_ENABLED=1
    # Reserve 1024 tokens of headroom over ISL+OSL.
    MAX_MODEL_LEN_BUFFER="${MAX_MODEL_LEN_BUFFER:-1024}"
    # The MoE kernels from MOE_KERNEL_THRESHOLD: the few-token kernel up to
    # it (an engine's rows gathered from all eight), the fused expert-parallel
    # kernel from the first row count per engine above it.
    export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-1}"
    export MOE_FUSED_EP_KERNEL_IMPL="${MOE_FUSED_EP_KERNEL_IMPL:-v3}"
    export TPU_MOE_LOWTOK_MAX_TOKENS="$MOE_KERNEL_THRESHOLD"
    export TPU_MOE_FUSED_EP_STEP_MIN_TOKENS=$(( MOE_KERNEL_THRESHOLD / EP_SIZE + 1 ))
    # The DP router holds new prompts and releases a group together: an
    # idle-engine trigger, a 5 s deadline, a 2 s quiet period.
    export VLLM_TPU_DP_PLACEMENT="${VLLM_TPU_DP_PLACEMENT:-release}"
    export VLLM_TPU_DP_RELEASE_THRESHOLD="${VLLM_TPU_DP_RELEASE_THRESHOLD:-8}"
    export VLLM_TPU_DP_RELEASE_DEADLINE_MS="${VLLM_TPU_DP_RELEASE_DEADLINE_MS:-5000}"
    export VLLM_TPU_DP_RELEASE_IDLE_TRIGGER="${VLLM_TPU_DP_RELEASE_IDLE_TRIGGER:-1}"
    export VLLM_TPU_DP_RELEASE_QUIET_MS="${VLLM_TPU_DP_RELEASE_QUIET_MS:-2000}"
    # Apply static per-layer expert placement when loading the checkpoint.
    # Set this to empty to retain the checkpoint's expert order.
    export TPU_EXPERT_PLACEMENT="${TPU_EXPERT_PLACEMENT-$SCRIPT_DIR/expert_placement_dp8_ep8.json}"
    # Split the vocabulary projection across engines (on at every DP8 point).
    export TPU_SHARDED_LM_HEAD="${TPU_SHARDED_LM_HEAD:-1}"
    # The counting-sort routing prep, the norm weights computed once, the
    # GDN output's gated norm and fp8 quantisation as one pass.
    export USE_MOE_COUNTING_SORT="${USE_MOE_COUNTING_SORT:-1}"
    export TPU_NORM_WEIGHT_HOIST="${TPU_NORM_WEIGHT_HOIST:-1}"
    export TPU_GDN_FUSED_NORM_QUANT="${TPU_GDN_FUSED_NORM_QUANT:-1}"
    export USE_BATCHED_RPA_LONGCTX="${USE_BATCHED_RPA_LONGCTX:-1}"
    export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-2048}"
    # The async all-gather flags
    # let the fused EP kernel's row gather run on the sparse cores. The
    # plugin prepends its own defaults and skips any flag already set here.
    DP8_LIBTPU_INIT_ARGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false"
    if [ "$USE_MOE_FUSED_EP_KERNEL" = "1" ]; then
      DP8_LIBTPU_INIT_ARGS+=" --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144"
    fi
    export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS-$DP8_LIBTPU_INIT_ARGS}"
    # The GDN recurrent state in bf16: half the state memory, and the decode
    # step's state loads and stores through the pool's packed 16-bit view.
    ;;
  TP8_EP)
    # Tensor-parallel attention across all 8 cores + expert-parallel MoE.
    DP_SIZE=1
    SHARDING_ARGS=(
      --tensor-parallel-size=8
      --data-parallel-size=1
      --enable-expert-parallel
    )
    # points.sh sets the attention kernel's decode batch and the expert
    # placement with copies per point. The few-token MoE kernel on decode-sized steps,
    # the attention kernel's decode VMEM limit, the GDN output's gated norm
    # and fp8 quantisation as one pass, the fused q/k norm, rope and gate,
    # and sequence parallelism for the prompt-sized steps (reduce-scatter
    # after attention; norms, router and shared expert on each core's own
    # rows; an all-gather for the experts), whose reduce-scatter runs on the
    # sparse cores.
    export TPU_RPA_DECODE_VMEM_MIB="${TPU_RPA_DECODE_VMEM_MIB:-32}"
    export TPU_GDN_FUSED_NORM_QUANT="${TPU_GDN_FUSED_NORM_QUANT:-1}"
    export TPU_FUSED_QK_NORM_ROPE="${TPU_FUSED_QK_NORM_ROPE:-1}"
    # Required for choosing exactly one holder of a replicated expert.
    export TPU_MOE_LOWTOK_FROM_SCORES="${TPU_MOE_LOWTOK_FROM_SCORES:-1}"
    export TPU_MOE_LOWTOK_SCORES_STRICT="${TPU_MOE_LOWTOK_SCORES_STRICT:-1}"
    # Above MOE_KERNEL_THRESHOLD rows a step is sequence parallel and its MoE
    # runs the fused expert-parallel kernel on each rank's own rows; at or
    # below it the rows stay replicated for the few-token kernel.
    export TPU_MOE_SEQUENCE_PARALLEL="${TPU_MOE_SEQUENCE_PARALLEL:-1}"
    export TPU_SP_MIN_ROWS=$(( MOE_KERNEL_THRESHOLD + 1 ))
    # The fp8 all-gather before the attention and GDN in-projections (each
    # rank quantises its own rows once), and the counting-sort routing prep.
    export TPU_SP_FP8_GATHER="${TPU_SP_FP8_GATHER:-1}"
    export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-1}"
    export MOE_FUSED_EP_KERNEL_IMPL="${MOE_FUSED_EP_KERNEL_IMPL:-v3}"
    export TPU_MOE_FUSED_EP_STEP_MIN_TOKENS=$(( MOE_KERNEL_THRESHOLD / EP_SIZE + 1 ))
    export USE_MOE_COUNTING_SORT="${USE_MOE_COUNTING_SORT:-1}"
    export TPU_MOE_LOWTOK_MAX_TOKENS="$MOE_KERNEL_THRESHOLD"
    # (1 + weight) of every norm kept as a buffer made once after loading
    # instead of recomputed in each forward; the fused q/k path reads the same
    # buffers, so the row buckets above and below its cap lift the same inputs.
    export TPU_NORM_WEIGHT_HOIST="${TPU_NORM_WEIGHT_HOIST:-1}"
    # the reduce-scatter and the fused kernel's row gather on the sparse cores
    TP8_LIBTPU_INIT_ARGS="--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144"
    export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS-$TP8_LIBTPU_INIT_ARGS}"
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

# Headroom over ISL+OSL (matches InferenceX qwen3.5: ISL+OSL+20; the DP8_EP
# case set its own above)
MAX_MODEL_LEN_BUFFER="${MAX_MODEL_LEN_BUFFER:-20}"
MAX_MODEL_LEN=$((ISL + OSL + MAX_MODEL_LEN_BUFFER))

if [ "$SHARDING" = "DP8_EP" ]; then
  # Per point (points.sh): the tokens per engine per forward step
  # (the prompt chunk) and the requests per engine, with room for the
  # router to place a prompt on the engine with the fewest in flight. The
  # defaults are the 64-user point's.
  GLOBAL_MAX_NUM_BATCHED_TOKENS="${GLOBAL_MAX_NUM_BATCHED_TOKENS:-8192}"
  MAX_NUM_BATCHED_TOKENS=$(((GLOBAL_MAX_NUM_BATCHED_TOKENS + DP_SIZE - 1) / DP_SIZE))
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-$((CONC * 2 / DP_SIZE))}"
else
  # Honor an explicit global prompt-token budget, including budgets below
  # the input length. Otherwise use max(ISL, GLOBAL_BATCHED_TOKENS_MIN).
  if [ -n "${GLOBAL_MAX_NUM_BATCHED_TOKENS:-}" ]; then
    GLOBAL_BATCHED_TOKEN=$GLOBAL_MAX_NUM_BATCHED_TOKENS
  else
    GLOBAL_BATCHED_TOKENS_MIN="${GLOBAL_BATCHED_TOKENS_MIN:-16384}"
    GLOBAL_BATCHED_TOKEN=$(( ISL > GLOBAL_BATCHED_TOKENS_MIN ? ISL : GLOBAL_BATCHED_TOKENS_MIN ))
  fi
  MAX_NUM_BATCHED_TOKENS=$(((GLOBAL_BATCHED_TOKEN + DP_SIZE - 1) / DP_SIZE))
  # The running slots: an explicit MAX_NUM_SEQS as given, else one per user.
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-$((CONC / DP_SIZE))}"
fi
[ "$MAX_NUM_SEQS" -lt 1 ] && MAX_NUM_SEQS=1

# Compile at boot so initialization reserves memory through the compile path.
# Set this to 0 to enable compile-cache reuse.
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"

# Increase API-server frontend wait time since cold init may take long.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

export TPU_ACCELERATOR_TYPE=tpu7x
export USE_MOE_SPARSE_CORE=1
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"

# Select the SparseCore gather-reduce implementation.
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
# Maximum rows for the TP small all-reduce and DP reduce-scatter combine.
# Set to 0 to disable both custom collectives.
export TPU_SMALL_ALL_REDUCE_MAX_ROWS="${TPU_SMALL_ALL_REDUCE_MAX_ROWS:-64}"
# BF16 halves recurrent-state storage relative to float32.
# Set GDN_STATE_DTYPE=float32 to use FP32 recurrent state.
GDN_STATE_DTYPE="${GDN_STATE_DTYPE:-bfloat16}"
EXTRA_ARGS+=(--mamba-ssm-cache-dtype "$GDN_STATE_DTYPE")

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
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
)
# Pass --load-format when streaming from GCS.
[ -n "${LOAD_FORMAT:-}" ] && args+=(--load-format="$LOAD_FORMAT")

if [ "${DRY_RUN:-0}" = "1" ]; then
  for key in SHARDING CONC MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
      MOE_FUSED_EP_KERNEL_IMPL TPU_MOE_SEQUENCE_PARALLEL MOE_KERNEL_THRESHOLD TPU_MOE_LOWTOK_MAX_TOKENS TPU_MOE_LOWTOK_FROM_SCORES \
      TPU_MOE_LOWTOK_SCORES_STRICT TPU_EXPERT_PLACEMENT \
      TPU_NORM_WEIGHT_HOIST TPU_SP_MIN_ROWS TPU_MOE_FUSED_EP_STEP_MIN_TOKENS \
      TPU_RPA_DECODE_BATCH TPU_RPA_DECODE_VMEM_MIB TPU_SP_FP8_GATHER \
      TPU_GDN_FUSED_NORM_QUANT TPU_FUSED_QK_NORM_ROPE TPU_SHARDED_LM_HEAD \
      GLOBAL_MAX_NUM_BATCHED_TOKENS VLLM_TPU_DP_RELEASE_THRESHOLD \
      GDN_STATE_DTYPE TPU_SMALL_ALL_REDUCE_MAX_ROWS TPU_SMALL_ALL_REDUCE_FORM \
      TPU_GREEDY_TOP_TOKENS TPU_GDN_TABLES_IN_KERNEL \
      TPU_GDN_DECODE_TILE TPU_GATED_NORM_TILE \
      TPU_TOKEN_BUCKET_EXTRA TPU_TOKEN_BUCKET_STEP LIBTPU_INIT_ARGS; do
    printf '%s=%s\n' "$key" "${!key-}"
  done
  printf 'vllm serve'; printf ' %q' "${args[@]}"; printf '\n'
  exit 0
fi

set -x
vllm serve "${args[@]}"
set +x
