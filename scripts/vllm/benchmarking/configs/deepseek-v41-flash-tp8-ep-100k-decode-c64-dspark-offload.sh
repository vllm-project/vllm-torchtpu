#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Fixed C64 CI workload; keep it separate from the C4/C8/C16 baseline.
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_NUM_SEQS=64
CONCURRENCY_OPTIONS=64
COMPILE_SIZES='[64,384,8192]'
KV_CACHE_MEMORY_BYTES=21474836480
DSPARK_K=5
export TPU_DEEPSEEK_V41_IMPL=v2
export TPU_DEEPSEEK_V41_CED=1
export TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD=1
export TPU_DEEPSEEK_V41_SP=0
export TPU_DEEPSEEK_V41_LOCAL_Q=0
export USE_MOE_SPARSE_CORE=1
export USE_MOE_FUSED_EP_KERNEL=0
export MOE_FUSED_EP_ENABLE_W4A8=0
export MOE_FUSED_EP_V2_SHARDED_PLAN=0
# shellcheck source=/dev/null
source "$CONFIG_DIR/deepseek-v41-flash-tp8-ep-100k-decode-dspark.sh"
MAX_MODEL_LEN=114688
MAX_NUM_BATCHED_TOKENS=8192
KV_CACHE_DTYPE=fp8
# Long enough for all 64 requests to decode together after chunked prefill,
# including speculative steps while later requests are still prefilling.
ISL_OSL_CONFIGS="100000:12000"
