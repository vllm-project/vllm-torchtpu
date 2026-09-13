#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly perf benchmark for Kimi-K3 on the multihost TPU v7x-32 pod
# (4 hosts x 8 chips): TP=32 with EP. Single-request latency probe at
# 8k in / 1k out. Baseline calibrated on the pod from three repeats in one
# server lifetime, run-to-run spread <= 0.3% on every gated metric — the
# default 5% perf tolerance applies.
# Serve flags select the tuned MLA path for this model on this topology. The v2
# MLA tuning table carries an entry for K3 at TP32 keyed on (max_num_tokens,
# q_heads, kv_dtype, page_size_per_kv_packing, max_num_seqs, pages_per_seq), and
# the settings below are what set those fields:
#
#   max_num_tokens           <- MAX_NUM_BATCHED_TOKENS 8192 + compile size 8192
#   actual_num_q_heads       <- 96 / TP32 = 3
#   kv_dtype                 <- KV_CACHE_DTYPE fp8
#   page_size_per_kv_packing <- block size 256 / kv_packing 4 = 64
#   max_num_seqs             <- MAX_NUM_SEQS 8
#   pages_per_seq            <- ceil(MAX_MODEL_LEN 10240 / block size 256) = 40
#
# Changing max-model-len, block-size, kv-cache-dtype or max-num-seqs drops the
# lookup to the untuned fallback with no warning, so treat those four as one
# setting rather than four. The table has no K3 TP32 entry above max_num_seqs 8,
# which is why this config stays a single-request latency probe.
#
# compile_sizes[0] = 1: a decode step is padded up to the next compiled size, so
# without a 1 in the list a single-sequence step runs a 16-token shape.
MODEL="moonshotai/Kimi-K3"
# v7x-32 hosts have no data disk that fits the 1561 GB checkpoint, so
# weights stream directly from GCS via --load-format runai_streamer.
# --enable-ep-weight-filter (model_loader_patches.py, #373) keeps that
# tractable: each rank streams only the byte ranges of its own experts
# (28/896 at EP32) plus dense weights, instead of the full checkpoint —
# full-checkpoint streaming accumulated >1 TB of host anonymous memory
# and drew Ray OOM kills (dev builds 82/83/85).
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
# No --quantization: the FP8 checkpoint self-describes (compressed-tensors).
QUANTIZATION=""
ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
# 64 sequential bs=1 requests (~42 s each, ~45 min of bench time) give a
# stabler median for the gated metrics; checkpoint stream + compile still
# dominate the step's wall clock. check_regression.py fails a cell when
# completed < 0.95 x baseline completed, so the floor is 60.8 and up to 3
# transient request failures do not fail the nightly on the single
# tpu_v7x_32 agent.
NUM_PROMPTS=64

# Disable random length jitter: benchmark requests use exact nominal
# 8192:1024 lengths, so TTFT/TPOT medians are comparable run to run.
RANDOM_RANGE_RATIO="0.0"

# Server configuration:
# 10240 gives pages_per_seq 40 at block size 256, the value the tuned MLA entry
# is keyed on. Still the exact nominal length for 8192 in + 1024 out.
MAX_MODEL_LEN=10240
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=8

# Startup budget covers the GCS stream plus a cold XLA compile
# (VLLM_DISABLE_COMPILE_CACHE=1 in the multihost containers).
SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.82
# The proven recipe serves with vllm's default KV cache dtype and
# synchronous scheduling; both are recorded in the run's config.json.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"

# Kimi-K3 is an MLA model; ATTENTION_BACKEND default CUSTOM applies.

# Serve flags:
# --trust-remote-code: the HF repo ships custom code (TikTokenTokenizer).
# --language-model-only + --limit-mm-per-prompt: K3 is image-text-to-text;
#   serve just the LM (the vision tower has no TPU path here).
# --model-loader-extra-config memory_limit=16GiB: caps the runai streamer's
#   per-rank host buffer so 32 ranks fit host RAM.
# --compilation-config compile_sizes [16,512]: the only token-count
#   buckets this bench reaches — prefill chunks are exactly 512 (8192 is
#   a multiple of max-num-batched-tokens, no remainder) and bs=1 decode
#   pads to 16. 64/256 were compiled cold every run (~5.6 min on the pod)
#   and never executed.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 --compilation-config {\"compile_sizes\":[1,8,8192]}"

# The bench client tokenizes from the dir the server pulled from GCS
# (run_benchmarks.sh derives --tokenizer from MODEL_URI); the harness
# passes --trust-remote-code to the client unconditionally, which covers
# the custom tokenizer code in that dir.

# Greedy decoding for bench requests.
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
# MMLU-Pro extracts an answer letter from a complete chat response. K3's
# template closes assistant messages, so continuation-based tasks do not fit.
LM_EVAL_TASKS="mmlu_pro"
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
EXTRA_LM_EVAL_MODEL_ARGS="num_concurrent=16"
# K3's tokenizer uses `thinking`; its default enables reasoning.
LM_EVAL_GEN_KWARGS='{"chat_template_kwargs":{"thinking":false}}'
