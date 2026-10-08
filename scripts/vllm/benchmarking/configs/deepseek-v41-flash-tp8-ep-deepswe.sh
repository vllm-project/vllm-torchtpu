#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# DeepSeek V4.1 Flash server for DeepSWE agent runs (run_deepswe.sh). Same
# runtime as deepseek-v41-flash-tp8-ep-100k-decode, plus what the agent needs:
# long context, tool-call and reasoning parsers, and the sampling DeepSeek
# reports DeepSWE with (reasoning_effort=max, temperature 1.0, top_p 0.95).
#
# DeepSWE task contexts reach ~660K tokens, so MAX_MODEL_LEN is 1M as on the
# reference GPU run. The 20 GiB KV budget holds ~8.2M tokens, 7.8 requests at
# full length; actual contexts average ~200K.
MODEL="${DEEPSEEK_V41_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
ASYNC_SCHEDULING=true
GPU_MEMORY_UTILIZATION=0.85
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# Required by run_benchmarks.sh for every config; the agent drives the load.
ISL_OSL_CONFIGS="128:16"
CONCURRENCY_OPTIONS="1"
RANDOM_RANGE_RATIO="0.0"
BENCHMARK_TEMPERATURE=0
EVAL_TOLERANCE="0.03"
SERVER_READY_WAIT_MIN=120
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE=vllm
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_AOT_COMPILE=0
export TPU_DEEPSEEK_V41_IMPL="${TPU_DEEPSEEK_V41_IMPL:-v2}"
export TPU_DEEPSEEK_V41_CED="${TPU_DEEPSEEK_V41_CED:-1}"
# Prefix caching stays off (run_benchmarks.sh's default) until CED supports
# it; set ENABLE_PREFIX_CACHING=true to try it. Each agent step then
# re-prefills the whole conversation.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--enable-ep-weight-filter --safetensors-load-strategy=prefetch --safetensors-prefetch-num-threads=8 --block-size=16 --kv-cache-memory-bytes=21474836480 --compilation-config={\"compile_sizes\":[4,8,16,2048]} --enable-auto-tool-choice --tool-call-parser=deepseek_v41 --reasoning-parser=deepseek_v41 --default-chat-template-kwargs={\"reasoning_effort\":\"max\"} --override-generation-config={\"temperature\":1.0,\"top_p\":0.95}"

# Keep the v7x Engram gather on the runtime default path.
if [[ "${LIBTPU_INIT_ARGS:-}" != *"--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"* ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"
fi
