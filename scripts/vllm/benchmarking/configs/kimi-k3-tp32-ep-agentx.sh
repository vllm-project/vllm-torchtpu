#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Kimi-K3 on the v7x-32 pod: server geometry for the InferenceX AgentX trace
# replay (multi-turn agentic-coding sessions, 100k+ token contexts, heavy
# prefix reuse), driven by k3_agentx_sweep.sh. Derived from
# kimi-k3-tp32-ep.sh; differences:
#   - long context: MAX_MODEL_LEN default 131072 (the driver overrides it);
#     the aiperf client filters traces above it
#   - prefix caching via ENABLE_PREFIX_CACHING (driver --apc 1), which on the
#     hybrid KDA+MLA model needs --mamba-cache-mode align and the unified KV
#     pool (TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1, exported by the driver)
#   - throughput regime: MAX_NUM_SEQS = 2 x max concurrency (driver),
#     batched tokens 8192, gpu-memory-utilization 0.85 (0.7 leaves ~3 GiB/rank
#     of KV, far too little for concurrent 128k contexts)
#   - --block-size 256: with the 16-token default, prefill at 100k+ context is
#     3 to 8.5x slower and decode slows with context (#837); 256 is correct on
#     GSM8K at every concurrency
#   - tool-call + reasoning parsers so the chat endpoint accepts the
#     tool_calls / tool-result messages in the Claude Code traces
#   - compile_sizes: a decode step carries up to MAX_NUM_SEQS tokens and is
#     padded up to the next bucket, so the decode buckets follow MAX_NUM_SEQS
#     (16 covers up to 16 sequences; 32..512 are added up to the first bucket
#     that holds MAX_NUM_SEQS). Prefill chunks are 8192 with remainders padded to 1024/2048/4096.
#   - the built-in harness bench is a 4-prompt 8k/1k warmup; the replay is
#     driven by k3_agentx_sweep.sh against the kept-alive server
# K3_EXTRA_SERVE_ARGS (driver --extra-serve-args) is appended LAST, so it can
# override any flag set here (vLLM takes the last occurrence of a flag).
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""
ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=4
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
# Prefix caching on the hybrid needs the align cache mode (tpu_platform.py
# gate); it is meaningless without prefix caching, so add it conditionally.
APC_ARGS=""
if [ "$ENABLE_PREFIX_CACHING" = "true" ]; then APC_ARGS="--mamba-cache-mode align "; fi

# Decode buckets: 16, then 32, 64, ... up to the first one that holds
# MAX_NUM_SEQS tokens (a decode step of 48 sequences needs the 64 bucket).
K3_COMPILE_SIZES="16"
if [ "$MAX_NUM_SEQS" -gt 16 ]; then
    for _s in 32 64 128 256 512; do
        K3_COMPILE_SIZES="$K3_COMPILE_SIZES,$_s"
        [ "$_s" -ge "$MAX_NUM_SEQS" ] && break
    done
fi
K3_COMPILE_SIZES="$K3_COMPILE_SIZES,1024,2048,4096,8192"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 ${APC_ARGS}--enable-auto-tool-choice --tool-call-parser kimi_k3 --reasoning-parser kimi_k3 --compilation-config {\"compile_sizes\":[$K3_COMPILE_SIZES]}${K3_EXTRA_SERVE_ARGS:+ $K3_EXTRA_SERVE_ARGS}"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
