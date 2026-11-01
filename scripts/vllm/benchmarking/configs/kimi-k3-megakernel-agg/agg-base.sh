#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# shellcheck disable=SC2034  # Variables are read by k3_1p1d.sh and k3_agentx_client_pod.sh
# What every Kimi-K3 megakernel aggregate point shares: one engine on one
# 2x2x4 slice runs both megakernel programs (no KV transfer config: the
# aggregate role), TP32 on the TP32 pre-shard, FP8 KV in 512-token blocks,
# DSpark with the MLA draft, behind vLLM's Rust frontend, on
# kimi-k3-agentx-agg.yaml. The engine environment and flags are the Oct 16
# 1P1D set's (../kimi-k3-megakernel/recipe-base.sh) without Raiden and the host
# KV store. A point sets the AGG_* values below and sources this file.
#
#   AGG_CONCURRENCY     the AgentX concurrency
#   AGG_MAX_NUM_SEQS    --max-num-seqs
#   AGG_DRAFTS          DSpark drafts (3 or 7)
#   AGG_ACCEPTANCE      synthetic acceptance length of the replay
#   AGG_KV_BYTES        --kv-cache-memory-bytes; optional, 14 GiB

for _v in AGG_CONCURRENCY AGG_MAX_NUM_SEQS AGG_DRAFTS AGG_ACCEPTANCE; do
  [ -n "${!_v:-}" ] || { echo "ERROR: kimi-k3-megakernel-agg: the point sets no $_v" >&2; return 2; }
done
_eval_job="${AGENTX_EVAL_JOB:-0}"
case "$_eval_job" in 0|1) ;;
  *) echo "ERROR: kimi-k3-megakernel-agg: AGENTX_EVAL_JOB='$_eval_job' must be 0 or 1" >&2
     return 2 ;;
esac

# ---- what the AgentX client reads ----------------------------------------------
K3_1P1D_LAUNCH=nnodes
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
MAX_NUM_SEQS=$AGG_MAX_NUM_SEQS
AGENTX_ALLOW_SEQS_BELOW_CONC=1
AGENTX_CONC_LIST="$AGG_CONCURRENCY"
AGENTX_MODEL_PREFIX=kimik3
AGENTX_PRECISION=fp4
AGENTX_POST_EVAL_SUITES=""
AGENTX_EVAL_JOB_SUITES="gsm8k kimi_tool_call_schema_full"
AGENTX_GSM8K_LIMIT=0
AGENTX_SPEC_METHOD=dspark
[ "$AGG_DRAFTS" != 0 ] || AGENTX_SPEC_METHOD=none
AGENTX_PREFILL_DCP_SIZE=32
AGENTX_DURATION=3600
AGENTX_FAST=0

[ "${K3_1P1D_ENGINE:-0}" = 1 ] || return 0

# ---- engine only ---------------------------------------------------------------
if [ "${K3_1P1D_ROLE:-}" != aggregate ]; then
  echo "ERROR: kimi-k3-megakernel-agg: K3_1P1D_ROLE must be aggregate" >&2; return 2
fi
for _v in INDEX POD_IP DP_ADDRESS; do
  [ -n "${!_v:-}" ] || { echo "ERROR: kimi-k3-megakernel-agg: $_v is unset; k3_1p1d.sh sets it" >&2; return 2; }
done
_role=aggregate
PLUGIN_SRC="${REPO_DIR:-/workspace/vllm-torchtpu}/src"
LOG_DIR="${TMPDIR:-/dev/shm}/k3mk/logs"
HF_HOME_DIR="${TMPDIR:-/dev/shm}/k3mk/hf-home"
CACHE_ROOT="/cache/jax/k3-mk/$(printf '%s' "${K3_ENGINE_IMAGE:-unknown}" | sha1sum | cut -c1-12)"
mkdir -p "$CACHE_ROOT/$_role"/{vllm-runtime,vllm-config,torch_tpu_tier3} "$HF_HOME_DIR" "$LOG_DIR"
MODEL_DIR=/ckpt/k3
HF_HUB_CACHE_DIR=/cache/hf/hub
PRESHARDED_DIR="$HF_HUB_CACHE_DIR/presharded/kimi-k3-tp32"
DSPARK_REVISION=cf6b8244620e7ea4b0651d214f28e89eac75bed6
DSPARK_DIR="$HF_HUB_CACHE_DIR/models--Inferact--Kimi-K3-DSpark/snapshots/$DSPARK_REVISION"

# ---- environment (recipe-base.sh's, without Raiden) ----------------------------
export PYTHONPATH="$PLUGIN_SRC"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TCMALLOC_AGGRESSIVE_DECOMMIT=true
export VLLM_USE_RUST_FRONTEND=1
export VLLM_RUST_FRONTEND_PATH=/usr/local/bin/vllm-rs
export VLLM_ENGINE_READY_TIMEOUT_S=14400
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_CACHE="$HF_HUB_CACHE_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_TARGET_DEVICE=tpu
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP="$POD_IP"
export VLLM_SERVER_DEV_MODE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_HOST_BOUNDS="1,1,4"
export TPU_NODE_ID="$INDEX"
export TPU_SKIP_MDS_QUERY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TPU_MOE_HIERARCHICAL_EP=1
export TPU_LATENT_PROJ_INTRA_CHIP_TP=1
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_LOG_DIR="$LOG_DIR/tpu"
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK=0
export NPROC=2
export TMPDIR=/dev/shm
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false --xla_tpu_dvfs_p_state=7"
export HF_MODULES_CACHE="$HF_HOME_DIR/modules-$_role"
export VLLM_XLA_CACHE_PATH="$CACHE_ROOT/$_role"
export VLLM_CACHE_ROOT="$CACHE_ROOT/$_role/vllm-runtime"
export VLLM_CONFIG_ROOT="$CACHE_ROOT/$_role/vllm-config"
export TORCH_TPU_TIER2_COMPILATION_CACHE="k3_k3mk_$_role"
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="k3_k3mk_$_role"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$CACHE_ROOT/$_role/torch_tpu_tier3"
export TPU_K3_MOE_REDUCE_DTYPE=float32
# The -all points' prefill FP8 cached-prefix attention (the EP16 placement plan
# does not apply to the aggregate's EP8 prefill).
export K3_PREFILL_PREFIX_FP8=1

# ---- arguments -------------------------------------------------------------------
_lengths=$(printf '{"%s":%s}' "$AGG_DRAFTS" "$AGG_ACCEPTANCE")
if [ "$_eval_job" = 1 ] || [ "$AGG_DRAFTS" = 0 ]; then
  # The eval-only job verifies drafts for real and runs the parsers; without
  # drafts there is nothing to accept.
  _loader=$(printf '{"presharded":"%s"}' "$PRESHARDED_DIR")
  _synthetic=""
else
  _loader=$(printf '{"presharded":"%s","synthetic_acceptance_lengths":%s}' "$PRESHARDED_DIR" "$_lengths")
  _synthetic=",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$AGG_ACCEPTANCE"
fi
_spec=$(printf '{"method":"dspark","model":"%s","num_speculative_tokens":%s%s}' "$DSPARK_DIR" "$AGG_DRAFTS" "$_synthetic")
SERVE_ARGS=(
  "$MODEL_DIR"
  --served-model-name moonshotai/Kimi-K3
  --tensor-parallel-size 32 --enable-expert-parallel
  --distributed-timeout-seconds 1800
  --max-model-len 1048576
  --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512
  --trust-remote-code --language-model-only
  --mamba-cache-mode align --enable-prefix-caching
  --enable-prompt-tokens-details
  --decode-context-parallel-size 32
  --load-format megakernel --model-loader-extra-config "$_loader"
  --max-num-seqs "$AGG_MAX_NUM_SEQS"
  --max-num-batched-tokens $((${AGG_PREFILL_WINDOW:-8192} + 8 * AGG_MAX_NUM_SEQS))
  --kv-cache-memory-bytes "${AGG_KV_BYTES:-15032385536}"
)
# AGG_DRAFTS=0: no speculative decoding.
[ "$AGG_DRAFTS" = 0 ] || SERVE_ARGS+=(--speculative-config "$_spec")
if [ "$_eval_job" = 1 ]; then
  SERVE_ARGS+=(--enable-auto-tool-choice --tool-call-parser kimi_k3 --reasoning-parser kimi_k3)
fi

K3_1P1D_INPUTS=("$MODEL_DIR/config.json" "$PRESHARDED_DIR/layout.json" "$PRESHARDED_DIR/prefill-sp-cache/layout.json"
  "$DSPARK_DIR/config.json" "$PRESHARDED_DIR/dspark-${DSPARK_REVISION:0:8}")
K3_1P1D_IMPORTS=(vllm_torchtpu.megakernel vllm_torchtpu.megakernel.kimi_k3.aggregate)
unset _v _role _loader _spec _synthetic _lengths _eval_job
