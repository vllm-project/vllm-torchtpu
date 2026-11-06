#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Native decode hill-climb launcher. Same engine as serve.sh (synthetic KV mode),
# parameterized by layout and the knobs below. Run on every host:
#   LAYOUT=tp32dcp8 bash serve_sweep.sh <host-rank>
# Knobs (env, defaults = the 2026-10-30 baseline):
#   LAYOUT        dp8tp4 | dp4tp8dcp2 | dp4tp8 | tp32dcp8 | tp32dcp4 | dp2tp16dcp4   (layout.sh)
#                 single host: tp8 | tp8xdcp2 | dp2xtp4
#   GLOBAL_SEQS   total concurrent requests across the slice (max-num-seqs = GLOBAL_SEQS / engines)
#   MTP_K         draft tokens per step; 0 disables speculative decoding
#   CANDIDATE_SAMPLING 1 = non-greedy verify steps sample from per-vocab-shard top-k candidates when every request
#                 keeps <= 48 tokens (TPU_CANDIDATE_SAMPLING; same distributions, float sums over the candidates)
#   SYNTH_ACCEPT  "" (verify drafts against the target) | comma list of per-position unconditional acceptance rates,
#                 e.g. 0.832,0.692,0.576 (synthetic rejection: draft i is accepted with that marginal probability)
#   MNBT          --max-num-batched-tokens
#   KV_BYTES      --kv-cache-memory-bytes; "auto" lets --gpu-memory-utilization size the pool
#   ASYNC_SCHED   1 = --async-scheduling (baseline), 0 = --no-async-scheduling
#   FUSED_EP      0 | v2 | adaptive: fused expert-parallel MoE kernel on every step (adaptive needs torch_tpu >= dev20261017)
#   W4A8          1 = fp8 activations in the fused EP kernel (needs FUSED_EP); 0 = W4A16 as served
#   HOT_SET       "" (real/dummy routing) | "uniform" (balanced simulated routing) | "<size>:<rate>[:spread]"
#                 skewed simulated routing: <rate> of the tokens route only into <size> fixed experts
#   BLOCK_SIZE    --block-size (baseline 384)
#   EXPERT_BLOCK  "" (served block-16 scales) | 256 | 512: requantize expert scales so the matmul scales after the MXU
#   ATTN_BLOCK    "" (kernel default) | long: decode attention block bkv 8192, batch 4, 2 buffers, for >= 64K tokens
#                 per sequence (faster at 64-128K, slower at 24 x 32K; one KV head per core, measured)
#   VERIFY_BLOCK  "" (kernel default) | window: MTP verify-window attention computes the whole window per KV block
#                 (kernel default block, bit-identical) | long: the same with bkv 8192, batch 4, 2 buffers
#   DRAFT_ONE_QUERY 1 = MTP draft steps after the first attend with the one-query kernel at the verify block's
#                 KV block (TPU_BATCHED_RPA_DRAFT_ONE_QUERY, bit-identical); needs VERIFY_BLOCK
#   FUSED_DRAFT   1 = draft steps 1..K-1 as one compiled program (VLLM_TPU_FUSED_DRAFT_LOOP); single-engine layouts only
#   FAST_SUBST    1 = steady-state fast token substitution (VLLM_TPU_FAST_TOKEN_SUBSTITUTION)
#   PACKED_DISPATCH 1 = one MoE dispatch all-gather per layer for hidden states + router logits (TPU_MOE_PACKED_DISPATCH)
#   FUSED_COMBINE 1 = one EP all-reduce per MoE layer instead of DP reduce-scatter + TP all-reduce (TPU_MOE_FUSED_COMBINE)
#   LM_HEAD_HIDDEN_MAJOR 1 = store the LM head [hidden, vocab shard] so the vocab projection is a plain matmul
#                 (TPU_LM_HEAD_HIDDEN_MAJOR; reduction order: logits may differ in the last bf16 bit)
#   EMBED_SHARD   ep = split the token embedding over all DP x TP ranks (TPU_EMBED_EP_SHARD; 1/DP of the table per rank,
#                 one DP reduce-scatter per lookup, bitwise the vocab-parallel lookup); "" = vocab-parallel over TP
#   WEIGHTS       dummy (default) | real: real checkpoint at MODEL_PATH
#   KV_MODE       synthetic (default, needs dummy weights) | normal: real prefill with prefix caching
#   EXTRA_ARGS    appended verbatim to vllm serve
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname -- "$0")/../env.sh" "${1:?Pass host rank 0..3}"
# shellcheck disable=SC1091
source "$(dirname -- "$0")/layout.sh"
KV_BYTES=${KV_BYTES:-auto}
ASYNC_SCHED=${ASYNC_SCHED:-1}
FUSED_EP=${FUSED_EP:-0}
W4A8=${W4A8:-0}
BLOCK_SIZE=${BLOCK_SIZE:-384}
HOT_SET=${HOT_SET:-}
EXPERT_BLOCK=${EXPERT_BLOCK:-}
ATTN_BLOCK=${ATTN_BLOCK:-}
MOE_A2A=${MOE_A2A:-0}
A2A_SC_OFFLOAD=${A2A_SC_OFFLOAD:-}
VERIFY_BLOCK=${VERIFY_BLOCK:-}
DRAFT_ONE_QUERY=${DRAFT_ONE_QUERY:-0}
FUSED_DRAFT=${FUSED_DRAFT:-0}
FAST_SUBST=${FAST_SUBST:-0}
PACKED_DISPATCH=${PACKED_DISPATCH:-0}
FUSED_COMBINE=${FUSED_COMBINE:-0}
LM_HEAD_HIDDEN_MAJOR=${LM_HEAD_HIDDEN_MAJOR:-0}
EMBED_SHARD=${EMBED_SHARD:-}
WEIGHTS=${WEIGHTS:-dummy}
SYNTH_ACCEPT=${SYNTH_ACCEPT:-}
CANDIDATE_SAMPLING=${CANDIDATE_SAMPLING:-0}
KV_MODE=${KV_MODE:-synthetic}
read -r -a EXTRA_ARGV <<< "${EXTRA_ARGS:-}"
cd "$(dirname -- "$0")/.."

if [[ "$FUSED_DRAFT" == 1 ]]; then
  # The fused draft loop does not support EP across data-parallel engines (idle ranks replay the per-step trace).
  (( ENGINES == 1 )) || { echo "FUSED_DRAFT=1 needs a single-engine layout (got $ENGINES engines)" >&2; exit 2; }
  export VLLM_TPU_FUSED_DRAFT_LOOP=1
fi
[[ "$FAST_SUBST" == 1 ]] && export VLLM_TPU_FAST_TOKEN_SUBSTITUTION=1
case "$PACKED_DISPATCH" in
  0) ;;
  1) export TPU_MOE_PACKED_DISPATCH=1 ;;
  *) echo "PACKED_DISPATCH must be 0 or 1" >&2; exit 2 ;;
esac
case "$FUSED_COMBINE" in
  0) ;;
  1) export TPU_MOE_FUSED_COMBINE=1 ;;
  *) echo "FUSED_COMBINE must be 0 or 1" >&2; exit 2 ;;
esac
case "$CANDIDATE_SAMPLING" in
  0) ;;
  1) export TPU_CANDIDATE_SAMPLING=1 ;;
  *) echo "CANDIDATE_SAMPLING must be 0 or 1" >&2; exit 2 ;;
esac
case "$LM_HEAD_HIDDEN_MAJOR" in
  0) ;;
  1) export TPU_LM_HEAD_HIDDEN_MAJOR=1 ;;
  *) echo "LM_HEAD_HIDDEN_MAJOR must be 0 or 1" >&2; exit 2 ;;
esac
case "$EMBED_SHARD" in
  "") ;;
  ep) export TPU_EMBED_EP_SHARD=1 ;;
  *) echo "EMBED_SHARD must be empty or ep" >&2; exit 2 ;;
esac
[[ -n "$EXPERT_BLOCK" ]] && export MOE_REQUANTIZE_BLOCK_SIZE="$EXPERT_BLOCK"
case "$MOE_A2A" in
  0) ;;
  1) export TPU_MOE_RAGGED_A2A=1 ;;
  *) echo "MOE_A2A must be 0 or 1" >&2; exit 2 ;;
esac
case "$A2A_SC_OFFLOAD" in
  "") ;;
  0) LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS --xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all=false"; export LIBTPU_INIT_ARGS ;;
  *) echo "A2A_SC_OFFLOAD must be empty or 0" >&2; exit 2 ;;
esac
case "$ATTN_BLOCK" in
  "") ;;
  long) export TPU_BATCHED_RPA_DECODE_BLOCK=8192,4,2 ;;
  *) echo "ATTN_BLOCK must be empty or long" >&2; exit 2 ;;
esac
case "$VERIFY_BLOCK" in
  "") ;;
  window) export TPU_BATCHED_RPA_VERIFY_BLOCK=window ;;
  long) export TPU_BATCHED_RPA_VERIFY_BLOCK=8192,4,2 ;;
  *) echo "VERIFY_BLOCK must be empty, window or long" >&2; exit 2 ;;
esac
case "$DRAFT_ONE_QUERY" in
  0) ;;
  1) [[ -n "$VERIFY_BLOCK" ]] || { echo "DRAFT_ONE_QUERY=1 needs VERIFY_BLOCK" >&2; exit 2; }
     export TPU_BATCHED_RPA_DRAFT_ONE_QUERY=1 ;;
  *) echo "DRAFT_ONE_QUERY must be 0 or 1" >&2; exit 2 ;;
esac
LOAD_ARGS=(--load-format dummy)
[[ "$WEIGHTS" == real ]] && LOAD_ARGS=()
case "$KV_MODE" in
  synthetic)
    [[ "$WEIGHTS" == dummy ]] || { echo "KV_MODE=synthetic requires WEIGHTS=dummy" >&2; exit 2; }
    CACHE_ARGS=(--no-enable-prefix-caching --kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_connector_module_path":"synthetic_decode_connector","kv_role":"kv_both","kv_connector_extra_config":{"synthetic_decode_benchmark":true}}') ;;
  normal) CACHE_ARGS=(--enable-prefix-caching) ;;
  *) echo "KV_MODE must be synthetic or normal" >&2; exit 2 ;;
esac
SPEC_ARGS=()
if (( MTP_K > 0 )); then
  SYNTH_JSON=""
  [[ -n "$SYNTH_ACCEPT" ]] && SYNTH_JSON=",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_rates\":[$SYNTH_ACCEPT]"
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K$SYNTH_JSON}")
fi
KV_ARGS=()
if [[ "$KV_BYTES" != auto ]]; then KV_ARGS=(--kv-cache-memory-bytes "$KV_BYTES"); fi
SCHED_ARGS=(--async-scheduling); (( ASYNC_SCHED )) || SCHED_ARGS=(--no-async-scheduling)
case "$FUSED_EP" in
  0) ;;
  v2|1) export USE_MOE_FUSED_EP_KERNEL=1 MOE_FUSED_EP_KERNEL_MIN_TOKENS=1 ;;
  adaptive) export MOE_FUSED_EP_KERNEL_IMPL=adaptive MOE_FUSED_EP_KERNEL_MIN_TOKENS=1 ;;
  *) echo "FUSED_EP must be 0, v2 or adaptive" >&2; exit 2 ;;
esac
if [[ "$W4A8" == 1 ]]; then
  [[ "$FUSED_EP" != 0 ]] || { echo "W4A8=1 needs FUSED_EP=v2 or adaptive" >&2; exit 2; }
  export MOE_FUSED_EP_ENABLE_W4A8=1
fi
if [[ "$HOT_SET" == uniform ]]; then
  export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random
elif [[ -n "$HOT_SET" ]]; then
  IFS=: read -r HS_SIZE HS_RATE HS_SPREAD <<< "$HOT_SET"
  export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=hot_set TPU_MOE_HOT_SET_SIZE="$HS_SIZE" TPU_MOE_HOT_SET_RATE="$HS_RATE"
  [[ "${HS_SPREAD:-}" == spread ]] && export TPU_MOE_HOT_SET_SPREAD=1
fi
export TORCH_TPU_TIER2_COMPILATION_CACHE="native_decode_${LAYOUT}_${POD_IP}"
echo "HILLCLIMB layout=$LAYOUT host=$HOST_RANK engines=$ENGINES seqs/engine=$SEQS_PER_ENGINE mtp=$MTP_K mnbt=$MNBT compile=$COMPILE_SIZES kv=$KV_BYTES async=$ASYNC_SCHED fused_ep=$FUSED_EP w4a8=$W4A8 expert_block=${EXPERT_BLOCK:-16} attn_block=${ATTN_BLOCK:-default} verify_block=${VERIFY_BLOCK:-default} draft_one_query=$DRAFT_ONE_QUERY moe_a2a=$MOE_A2A a2a_sc_offload=${A2A_SC_OFFLOAD:-default} fused_draft=$FUSED_DRAFT fast_subst=$FAST_SUBST lm_head_hidden_major=$LM_HEAD_HIDDEN_MAJOR embed_shard=${EMBED_SHARD:-tp} packed_dispatch=$PACKED_DISPATCH fused_combine=$FUSED_COMBINE weights=$WEIGHTS kv_mode=$KV_MODE synth_accept=${SYNTH_ACCEPT:-none} candidate_sampling=$CANDIDATE_SAMPLING hot_set=${HOT_SET:-none} headless=${HEADLESS_ARGS[*]:-no}"

exec numactl --interleave=all "$VLLM_BIN" serve \
  --pipeline-parallel-size 1 \
  "${PAR_ARGS[@]}" "${HEADLESS_ARGS[@]}" \
  --distributed-executor-backend mp \
  --data-parallel-backend mp \
  --cpu-distributed-timeout-seconds 14400 \
  --safetensors-load-strategy lazy \
  --enable-ep-weight-filter \
  --dtype bfloat16 \
  --language-model-only \
  --max-model-len 1048576 \
  --gpu-memory-utilization 0.9 \
  --attention-backend CUSTOM \
  --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype bfloat16 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --no-disable-hybrid-kv-cache-manager \
  "${SCHED_ARGS[@]}" \
  --no-enforce-eager \
  --additional-config '{"tpu_online_fp8":{"enabled":true,"weight_scheme":"per_channel","targets":["model.layers.*.self_attn.qkv_proj","model.layers.*.self_attn.o_proj","model.layers.*.linear_attn.in_proj_qkvz","model.layers.*.linear_attn.in_proj_ba","model.layers.*.linear_attn.out_proj","model.layers.*.mlp.shared_expert.gate_up_proj","model.layers.*.mlp.shared_expert.down_proj"],"exclude":["model.layers.*.mlp.experts*"]}}' \
  --seed 42 \
  --generation-config vllm \
  --mamba-cache-mode align \
  "${LOAD_ARGS[@]}" \
  --hf-overrides '{"max_position_embeddings":1048576,"rope_parameters":{"partial_rotary_factor":0.25,"rope_theta":10000000,"type":"yarn","rope_type":"yarn","factor":4,"original_max_position_embeddings":262144}}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --block-size "$BLOCK_SIZE" \
  --prefill-context-parallel-size 1 \
  --max-num-batched-tokens "$MNBT" \
  --max-num-seqs "$SEQS_PER_ENGINE" \
  --compilation-config "{\"backend\":\"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor\",\"compile_sizes\":$COMPILE_SIZES,\"inductor_compile_config\":{\"enable_auto_functionalized_v2\":false,\"size_asserts\":false,\"alignment_asserts\":false,\"scalar_asserts\":false}}" \
  "${SPEC_ARGS[@]}" \
  --model "$MODEL_PATH" \
  --served-model-name qwen38-native-dummy \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-prompt-tokens-details \
  "${KV_ARGS[@]}" \
  "${CACHE_ARGS[@]}" \
  "${EXTRA_ARGV[@]}"
