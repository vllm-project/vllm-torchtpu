#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
# shellcheck source-path=SCRIPTDIR
source "$(dirname -- "$0")/env.sh" "${1:?Pass host rank 0..3}"
MODE=${2:-synthetic}
case "$MODE" in
  synthetic)
    CACHE_ARGS=(--no-enable-prefix-caching --kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_connector_module_path":"synthetic_decode_connector","kv_role":"kv_both","kv_connector_extra_config":{"synthetic_decode_benchmark":true}}') ;;
  normal)
    CACHE_ARGS=(--enable-prefix-caching) ;;
  *) echo "Mode must be synthetic or normal" >&2; exit 2 ;;
esac
cd "$NATIVE_RUN_DIR"
exec numactl --interleave=all "$VLLM_BIN" serve \
  --pipeline-parallel-size 1 \
  --data-parallel-size 8 \
  --data-parallel-size-local 2 \
  --nnodes 1 \
  --distributed-executor-backend mp \
  --data-parallel-backend mp \
  --cpu-distributed-timeout-seconds 14400 \
  --master-port 29500 \
  --safetensors-load-strategy lazy \
  --enable-ep-weight-filter \
  --dtype bfloat16 \
  --language-model-only \
  --max-model-len 1048576 \
  --gpu-memory-utilization 0.9 \
  --cp-kv-cache-interleave-size 256 \
  --attention-backend CUSTOM \
  --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype bfloat16 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --no-disable-hybrid-kv-cache-manager \
  --async-scheduling \
  --no-enforce-eager \
  --additional-config '{"tpu_online_fp8":{"enabled":true,"weight_scheme":"per_channel","targets":["model.layers.*.self_attn.qkv_proj","model.layers.*.self_attn.o_proj","model.layers.*.linear_attn.in_proj_qkvz","model.layers.*.linear_attn.in_proj_ba","model.layers.*.linear_attn.out_proj","model.layers.*.mlp.shared_expert.gate_up_proj","model.layers.*.mlp.shared_expert.down_proj"],"exclude":["model.layers.*.mlp.experts*"]}}' \
  --seed 42 \
  --generation-config vllm \
  --mamba-cache-mode align \
  --load-format dummy \
  --hf-overrides '{"max_position_embeddings":1048576,"rope_parameters":{"partial_rotary_factor":0.25,"rope_theta":10000000,"type":"yarn","rope_type":"yarn","factor":4,"original_max_position_embeddings":262144}}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --block-size 384 \
  --tensor-parallel-size 4 \
  --data-parallel-hybrid-lb \
  --prefill-context-parallel-size 1 \
  --decode-context-parallel-size 1 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 12 \
  --compilation-config '{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[12,48,1024],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --data-parallel-address "$MASTER_ADDRESS" \
  --model "$MODEL_PATH" \
  --served-model-name qwen38-native-dummy \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-rpc-port 29650 \
  --enable-prompt-tokens-details \
  --kv-cache-memory-bytes 25769803776 \
  --data-parallel-start-rank "$DP_START_RANK" \
  "${CACHE_ARGS[@]}"
