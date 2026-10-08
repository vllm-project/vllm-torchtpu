#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Kimi-K3 on 32 v7x chips (TP32, one host group): server geometry for the
# InferenceX AgentX trace
# replay (multi-turn agentic-coding sessions, 100k+ token contexts, heavy
# prefix reuse). agentx_sweep.sh starts this server; agentx_client.sh replays
# the traces against it. Derived from kimi-k3-tp32-ep.sh; differences:
#   - long context: MAX_MODEL_LEN 131072; the aiperf client filters traces above
#     it, so that choice decides which traces are replayed at all
#   - prefix caching off by default (the -apc variant turns it on), which on the
#     hybrid KDA+MLA model needs --mamba-cache-mode align
#   - throughput regime: MAX_NUM_SEQS = 2 x AGENTX_CONC_LIST's top concurrency,
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
#     driven by agentx_client.sh against the kept-alive server
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""
# The AgentX cases: one trace replay per concurrency, ascending, against one
# server. agentx_sweep.sh reads this and passes it to agentx_client.sh, and the
# this is the only place that says what the agentx_sweep.sh lane covers.
# MAX_NUM_SEQS below has to cover it. A kube P/D lane serves a different
# deployment and so has its own config and its own list.
AGENTX_CONC_LIST="1 2 4 8"

# What InferenceX files the row under. The client is model-agnostic, so the model
# name and the precision label come from here. "fp4" because the checkpoint's
# experts are MXFP4 with FP8 attention, which is the label the published B300
# vLLM row uses.
AGENTX_MODEL_PREFIX=kimik3
AGENTX_PRECISION=fp4
# Printed under the summary, for reading our numbers against a published row, in
# the same unit the table reports: output tokens per dollar. The dashboard gives
# tok/s per GPU, so each point is that over B300's $2.26/GPU-hour, the
# "owning at large hyperscaler volume" tier of the SemiAnalysis TCO model that
# InferenceX costs its own rows with (inferencex.semianalysis.com/chips/b300).
# Interactivity is the P90 the dashboard reports at that concurrency, in tok/s
# per user, and it is half the comparison: a cheaper token at 8 tok/s/user is not
# the same product as a dearer one at 200.
AGENTX_REFERENCE_ROW="InferenceX dashboard, 8x B300 vLLM, taken 2026-09-07; out tok/\$ @ P90 interactivity tok/s/user, at B300 \$2.26/GPU-hour: c1 24.1k@203, c2 29.6k@133, c4 37.3k@121, c8 64.7k@64, c16 93.0k@32, c24 105.6k@22, c48 120.6k@12.5, c70 139.5k@7.9"

# Not the AgentX cases. run_benchmarks.sh insists on a fixed-sequence cell
# before it will hand back a live server, and these are the smallest one that
# boots; the replay that follows ignores them.
ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=4
RANDOM_RANGE_RATIO="0.0"

# Plain assignments from here down, with no env indirection: agentx_sweep.sh has
# no flag for any of them, so what this file says is what the server is. Another
# arm - prefix caching on, bf16 KV, a different parallelism - is a copy of this
# file with the line changed, and agentx_sweep.sh --config picks it.

# 131072, not the model's native context: at TP32 the MLA latent is replicated
# on every rank, so the pool holds ~1.16M tokens in all. aiperf FILTERS whole
# traces above this (it does not truncate), so a run here replays the subset
# that fits and is a bring-up number rather than a submission.
MAX_MODEL_LEN=131072
MAX_NUM_BATCHED_TOKENS=8192
# Twice AGENTX_CONC_LIST's top concurrency, so the scheduler is never the limit
# in a throughput measurement. agentx_sweep.sh checks the two agree instead of
# deriving this, so that a server too small for the ladder fails at boot rather
# than quietly capping the numbers.
MAX_NUM_SEQS=16

SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.85
# fp8 MLA latent KV. This file used to say "auto" while the driver exported fp8
# on every run, so reading the config told you bf16 - the reason the serve knobs
# live here now and not in a flag. For a bf16 arm, "auto" here.
KV_CACHE_DTYPE=fp8

ASYNC_SCHEDULING=true
# No spec decode: K3's checkpoint ships no MTP tensors, so DSpark is the only
# drafter, and InferenceX pins it with a synthetic acceptance length. To try it,
# set ASYNC_SCHEDULING=false, AGENTX_SPEC_METHOD=mtp, and append to
# EXTRA_SERVE_ARGS below:
#   --speculative-config {"method":"dspark","model":"Inferact/Kimi-K3-DSpark",
#     "num_speculative_tokens":7,"rejection_sample_method":"synthetic",
#     "synthetic_acceptance_length":3.84}
# AGENTX_SPEC_METHOD is what agentx_sweep.sh passes the client to record in the
# row, in InferenceX's naming.
AGENTX_SPEC_METHOD=none

# W4A16 MoE (the default gmm_v2 path). For the W4A8 arm - fp4 weights re-blocked
# at load, fp8 activations in the MoE GEMM, the TPU twin of the GPU b12x row -
# export MOE_REQUANTIZE_WEIGHT_DTYPE=fp4 and MOE_REQUANTIZE_BLOCK_SIZE=512 here,
# as configs/deepseek-v4-flash-dp8-ep.sh does.

# Prefix caching off. On the hybrid it needs the align cache mode (tpu_platform.py
# gate) and prompt-token details for the per-request hit rate, so an arm that
# turns it on sets all three lines together. It needs no env var for the unified
# KV pool: K3 is in POOLED_GDN_ARCHITECTURES (gdn_pool_layout.py), so the pooled
# layout is already its default.
ENABLE_PREFIX_CACHING=false
APC_ARGS=""
# ENABLE_PREFIX_CACHING=true
# APC_ARGS="--mamba-cache-mode align --enable-prompt-tokens-details "

# Decode buckets have to bracket the ladder at BOTH ends: a decode step is
# padded up to the next compiled size, so without a 1 a single-sequence step
# runs a 16-wide shape (kimi-k3-tp32-ep.sh says the same, and the nightly uses
# [1,8,8192]), and without a bucket that holds MAX_NUM_SEQS a full batch is
# padded to the next one up. 1,2,4,8,16 covers AGENTX_CONC_LIST; the loop adds
# 32..512 only when MAX_NUM_SEQS needs them.
K3_COMPILE_SIZES="1,2,4,8,16"
if [ "$MAX_NUM_SEQS" -gt 16 ]; then
    for _s in 32 64 128 256 512; do
        K3_COMPILE_SIZES="$K3_COMPILE_SIZES,$_s"
        [ "$_s" -ge "$MAX_NUM_SEQS" ] && break
    done
fi
K3_COMPILE_SIZES="$K3_COMPILE_SIZES,1024,2048,4096,8192"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 ${APC_ARGS}--enable-auto-tool-choice --tool-call-parser kimi_k3 --reasoning-parser kimi_k3 --compilation-config {\"compile_sizes\":[$K3_COMPILE_SIZES]}"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
