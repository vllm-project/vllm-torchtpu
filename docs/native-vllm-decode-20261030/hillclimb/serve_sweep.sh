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
#   MNBT          --max-num-batched-tokens
#   KV_BYTES      --kv-cache-memory-bytes; "auto" lets --gpu-memory-utilization size the pool
#   ASYNC_SCHED   1 = --async-scheduling (baseline), 0 = --no-async-scheduling
#   FUSED_EP      1 = fused expert-parallel MoE kernel on every step (MOE_FUSED_EP_KERNEL_IMPL=adaptive)
#   BLOCK_SIZE    --block-size (baseline 384)
#   EXTRA_ARGS    appended verbatim to vllm serve
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname -- "$0")/../env.sh" "${1:?Pass host rank 0..3}"
# shellcheck disable=SC1091
source "$(dirname -- "$0")/layout.sh"
KV_BYTES=${KV_BYTES:-auto}
ASYNC_SCHED=${ASYNC_SCHED:-1}
FUSED_EP=${FUSED_EP:-0}
BLOCK_SIZE=${BLOCK_SIZE:-384}
read -r -a EXTRA_ARGV <<< "${EXTRA_ARGS:-}"
cd "$(dirname -- "$0")/.."

SPEC_ARGS=()
if (( MTP_K > 0 )); then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K}")
fi
KV_ARGS=()
if [[ "$KV_BYTES" != auto ]]; then KV_ARGS=(--kv-cache-memory-bytes "$KV_BYTES"); fi
SCHED_ARGS=(--async-scheduling); (( ASYNC_SCHED )) || SCHED_ARGS=(--no-async-scheduling)
if (( FUSED_EP )); then
  export MOE_FUSED_EP_KERNEL_IMPL=adaptive MOE_FUSED_EP_KERNEL_MIN_TOKENS=1
fi
export TORCH_TPU_TIER2_COMPILATION_CACHE="native_decode_${LAYOUT}_${POD_IP}"
echo "HILLCLIMB layout=$LAYOUT host=$HOST_RANK engines=$ENGINES seqs/engine=$SEQS_PER_ENGINE mtp=$MTP_K mnbt=$MNBT compile=$COMPILE_SIZES kv=$KV_BYTES async=$ASYNC_SCHED fused_ep=$FUSED_EP headless=${HEADLESS_ARGS[*]:-no}"

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
  --load-format dummy \
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
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_connector_module_path":"synthetic_decode_connector","kv_role":"kv_both","kv_connector_extra_config":{"synthetic_decode_benchmark":true}}' \
  "${EXTRA_ARGV[@]}"
