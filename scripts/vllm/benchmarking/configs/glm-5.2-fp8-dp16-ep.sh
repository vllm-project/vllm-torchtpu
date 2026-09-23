#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly perf benchmark for GLM-5.2-FP8 on TPU v7x-16: attention-DP
# layout TP=1 x DP=16 with EP — every device runs its own chip-local
# attention replica; MoE experts stay EP-sharded across all 16 devices.
# At saturated load this serves 868.1 output tok/s (peak 3600), median
# TPOT 769ms — 13x the TP=16 bring-up throughput (dev build 196;
# util 0.90 vs 0.85 trades TPOT for throughput and half the queue-drain
# TTFT, and model-len 9216 + 64 seqs/engine beat the 32768/1024
# geometry by a further +7% throughput / -7% TPOT vs dev build 190).
# Requires the multihost DP support (#342) and the v7x topology-map
# fix (#400). Benchmarked at concurrency 1024: this layout is proven
# under sustained saturation (75 min clean, build 188); light trickle
# loads are the regime of the known SparseCore idle-trap (build 187),
# so the nightly measures the saturated operating point.
# Sized for decode batch 512 (64 / device across 8 devices per host).
# Throughput/chip = throughput / 16 (TPxDP recorded in config.json).
MODEL="zai-org/GLM-5.2-FP8"
# v7x-16 hosts have no attached disk, so weights stream from GCS via
# --load-format runai_streamer.
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/glm/GLM-5.2-FP8}"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=16
ENABLE_EP=true
# No --quantization: the FP8 checkpoint self-describes.
QUANTIZATION=""
ISL_OSL_CONFIGS="8192:1024"
# Single cell at concurrency 64 with 128 prompts while serving perf sits
# at the ~900ms/step kernel floor (~66 tok/s aggregate): a full
# 3-cell x 512-prompt sweep takes ~14h, and the higher-concurrency cells
# only measure queue drain at that speed. Restore "64 256 512" and
# NUM_PROMPTS=512 once kernel perf lands.
CONCURRENCY_OPTIONS="1024"
NUM_PROMPTS=2048

# Disable random length jitter: benchmark requests use exact nominal 8192:1024 lengths.
RANDOM_RANGE_RATIO="0.0"

# Server configuration:
# - max-model-len=9216: exact nominal length for 8192 input + 1024
#   output tokens; vs 32768 it shrinks per-request KV reservations and
#   compiled shapes (+7% throughput / -7% TPOT, dev build 196 vs 190).
MAX_MODEL_LEN=9216
# 1024/engine is required at DP=16: the MoE dispatch dimension
# topk(4) x DP(16) x batched-tokens must stay at 65536 — 2048/engine
# puts it at 131072 where a SparseCore fault halts the first batch
# (dev builds 186/187/188; tracked with the torch_tpu/libtpu teams).
# Aggregate step budget is still 16 x 1024 = 16k tokens.
MAX_NUM_BATCHED_TOKENS=1024
# max-num-seqs is PER DP ENGINE (vllm 0.27 scheduler:
# max_num_running_reqs = max_num_seqs, never divided by
# data_parallel_size), so 64 x DP16 = 1024 concurrent sequences
# fleet-wide, matching the benchmark concurrency.
MAX_NUM_SEQS=64

SERVER_READY_WAIT_MIN=180
GPU_MEMORY_UTILIZATION=0.90
KV_CACHE_DTYPE="fp8"

# GLM 5.2 is an MLA model: tpu_platform.get_attn_backend_cls forces
# FLASH_ATTN_MLA whenever use_mla is set.

# Parser flags for tool call / reasoning compatibility.
# --enable-ep-weight-filter: with EP + the runai gs:// path, each rank
# streams only the byte ranges of its own experts plus dense weights
# (model_loader_patches.py, #373) instead of the full 761GB checkpoint
# (validated: 16/256 experts per rank, dev build 184).
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--enable-ep-weight-filter --safetensors-load-strategy=prefetch --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45"

# MoE headroom analysis note:
# FORCE_MOE_RANDOM_ROUTING=1 can be set in the environment for headroom analysis
# to force balanced token allocation across MoE experts. Never use for serving or accuracy eval.

# Greedy decoding for bench requests.
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
LM_EVAL_TASKS="mmlu_pro"
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
EXTRA_LM_EVAL_MODEL_ARGS="num_concurrent=32,timeout=7200"
LM_EVAL_GEN_KWARGS="until=['<|im_end|>']"
