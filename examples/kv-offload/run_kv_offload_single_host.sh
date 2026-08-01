#!/bin/bash
#
# Single-host launcher for TPU ↔ CPU KV cache offloading via vLLM's
# OffloadingConnector framework backed by the TPUCPUOffloadingSpec.
#
# What this does:
#   - Sizes a 400 GiB host-resident KV cache pool that overflows HBM
#     for prefix-cache-heavy workloads.
#   - Enables eviction-triggered ("lazy") store so only blocks evicted
#     from the HBM cache are written to the host pool — avoids the
#     double-write churn of eager mode.
#   - Bumps libtpu's premapped DMA buffer to 8 GiB so lazy host copy_
#     calls don't fall back to slow per-call pinning.
#
# Throughput sweet spot (observed on Qwen3-Coder-480B-A35B-FP8 with
# prefix_repetition workload of 96 prefixes × 1024 prompts × 16384
# prefix_len): +56% tok/s vs no-offload, 86% external prefix-cache hit.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
KV_OFFLOADING_SIZE="${KV_OFFLOADING_SIZE:-400}"

# Eviction-triggered offload — only writes evicted blocks to host.
# Set to "0" for legacy eager double-write mode.
export KV_OFFLOAD_LAZY_STORE="${KV_OFFLOAD_LAZY_STORE:-1}"

# libtpu premapped buffer envelope. Required at 8 GiB when
# KV_OFFLOADING_SIZE > ~350; smaller envelopes fall back to slow per-call
# pinning on each lazy `copy_` and bottleneck D2H bandwidth.
export TPU_PREMAPPED_BUFFER_SIZE=$((8 * 1024 * 1024 * 1024))
export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=$((8 * 1024 * 1024 * 1024))

# H2D staging buffer cap. Default 2048 blocks is usually fine; bump for
# very long prefixes that would otherwise be chunked across engine steps.
export KV_H2D_POOL_MAX_BLOCKS="${KV_H2D_POOL_MAX_BLOCKS:-2048}"

exec vllm serve "$MODEL" \
    --tensor-parallel-size="$TP_SIZE" \
    --enable-expert-parallel \
    --quantization fp8 \
    --gpu-memory-utilization="$GPU_MEM_UTIL" \
    --kv-offloading-size "$KV_OFFLOADING_SIZE" \
    --kv-transfer-config '{
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "TPUCPUOffloadingSpec",
            "spec_module_path": "vllm_torchtpu.offload.cpu_tpu"
        }
    }' \
    --async-scheduling \
    --port="$PORT"
