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
# What every Kimi-K3 megakernel recipe point shares: Kimi-K3 1P1D served as the
# "Kimi-K3 on TPU7x: 1P1D prefill/decode recipes" doc serves it - each role
# TP32 on its own 2x2x4 slice, the megakernel backend on the TP32 pre-shard,
# FP8 KV in 512-token blocks, Raiden prefill-to-decode transfer and a host
# memory KV store, behind vLLM's Rust frontend - on the AgentX 1P1D pair
# (kimi-k3-agentx-1p1d.yaml). A point is one file beside this one, or in
# kimi-k3-megakernel-1013/ for the doc's Oct 13 set (its pruned points a-h),
# that sets the RECIPE_* values below and sources it, and
# K3_PD_CONFIG=<directory>/<point> serves it.
#
#   RECIPE_CONCURRENCY            the AgentX concurrency
#   RECIPE_DECODE                 dspark | dspark_mixed | dp2_dcp16 | dcp32
#   RECIPE_DECODE_MAX_NUM_SEQS    decode's --max-num-seqs
#   RECIPE_DRAFTS                 dspark: drafts
#   RECIPE_ACCEPTANCE             dspark: synthetic acceptance length
#   RECIPE_DRAFTS_PER_BATCH_SIZE  dspark_mixed: drafts by decode batch size
#   RECIPE_OFFLOAD_BYTES          host KV store size, all ranks; optional
#   RECIPE_SET                    oct13 for the doc's Oct 13 set; unset for the first
#   RECIPE_STORE_SERVERS          1: the Oct 13 decode-host store servers
#   RECIPE_STORE_NODE_BYTES       each store server's DRAM; optional
#   RECIPE_PARSERS                0: no K3 reasoning or tool-call parsers and no
#                                 tool-call suite, the doc's command line; default 1
#
# Two readers:
#   k3_agentx_client_pod.sh  sources it with K3_1P1D_ROLE=decode for the shape,
#                            MAX_NUM_SEQS and the AGENTX_* cases and labels, and
#                            stops at the K3_1P1D_ENGINE guard
#   k3_1p1d.sh               sets K3_1P1D_ENGINE=1 and K3_1P1D_ROLE, and from its
#                            own shell INDEX, POD_IP, DP_ADDRESS (host 0 of the
#                            role) and resolve(); gets SERVE_ARGS, the doc's
#                            environment and the store servers' commands, and
#                            launches with --nnodes
#
# Where this departs from the doc, each needed on GKE: distinct Raiden ids per
# role and --headless on hosts 1-3 (which k3_1p1d.sh adds).
#
# Not in the doc: both roles run --enable-auto-tool-choice and the kimi_k3
# tool-call and reasoning parsers, and the Kimi tool-call suite runs after
# the replay, as InferenceX does for every K3 point. The suite's strict tool
# calls decode under a grammar, which the decoder applies from
# inferact/megakernel-integration-1015 on; the replay sends no tools.
# RECIPE_PARSERS=0 drops the parsers and the suite. AGENTX_EVAL_JOB=1, the
# GSM8K eval-only job, has DSpark verify drafts for real.
#
# RECIPE_SET=oct13 serves the doc's Oct 13 set as its "Changes from" list reads:
# TPU P-state 7 and RAIDEN_NUMA_POLICY=bind on both roles, the prefill without
# --prefix-cache-retention-interval, the one-hour benchmark, and for the points
# without speculation (e-h) the prefill's --long-prefill-token-threshold 4096
# and --numa-bind (which needs numactl in the engine image) and the decode's
# default deferred Stage-3 submit. RECIPE_STORE_SERVERS=1 (c-h) adds the global
# registry on the prefill head and a host-store server on each decode host,
# which k3_1p1d.sh starts; the prefill's host store joins their pool and the
# decode drops its own.

for _v in RECIPE_CONCURRENCY RECIPE_DECODE RECIPE_DECODE_MAX_NUM_SEQS; do
  [ -n "${!_v:-}" ] || { echo "ERROR: kimi-k3-megakernel: the point sets no $_v" >&2; return 2; }
done
case "$RECIPE_DECODE" in
  dspark)
    if [ -z "${RECIPE_DRAFTS:-}" ] || [ -z "${RECIPE_ACCEPTANCE:-}" ]; then
      echo "ERROR: kimi-k3-megakernel: dspark needs RECIPE_DRAFTS and RECIPE_ACCEPTANCE" >&2
      return 2
    fi ;;
  dspark_mixed)
    if [ -z "${RECIPE_DRAFTS_PER_BATCH_SIZE:-}" ]; then
      echo "ERROR: kimi-k3-megakernel: dspark_mixed needs RECIPE_DRAFTS_PER_BATCH_SIZE" >&2
      return 2
    fi ;;
  dp2_dcp16|dcp32) ;;
  *) echo "ERROR: kimi-k3-megakernel: RECIPE_DECODE='$RECIPE_DECODE'; expected dspark, dspark_mixed, dp2_dcp16 or dcp32" >&2
     return 2 ;;
esac
# The decodes without speculation: attention DP2 x DCP16, or plain DCP32 (the
# doc's pruned point h: the DP2 points' prefill and decode, with the transfer
# keeping DCP32 and the decode without attention DP).
_no_spec=0
case "$RECIPE_DECODE" in dp2_dcp16|dcp32) _no_spec=1 ;; esac
case "${RECIPE_SET:-}" in
  ''|oct13) ;;
  *) echo "ERROR: kimi-k3-megakernel: RECIPE_SET='$RECIPE_SET'; expected oct13 or unset" >&2
     return 2 ;;
esac
if [ -n "${RECIPE_STORE_SERVERS:-}" ] && [ "${RECIPE_SET:-}" != oct13 ]; then
  echo "ERROR: kimi-k3-megakernel: RECIPE_STORE_SERVERS is the Oct 13 set's (RECIPE_SET=oct13)" >&2
  return 2
fi
# The eval-only job always runs the parsers.
_eval_job="${AGENTX_EVAL_JOB:-0}"
_parsers="${RECIPE_PARSERS:-1}"
case "$_eval_job:$_parsers" in [01]:[01]) ;;
  *) echo "ERROR: kimi-k3-megakernel: AGENTX_EVAL_JOB='$_eval_job' and RECIPE_PARSERS='$_parsers' must be 0 or 1" >&2
     return 2 ;;
esac
[ "$_eval_job" != 1 ] || _parsers=1

# ---- what the AgentX client reads ----------------------------------------------
# Each role is TP32 with EP over all 32 ranks and no DP, launched with --nnodes.
K3_1P1D_LAUNCH=nnodes
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
# The decode engine's limit, which is what bounds the replay.
MAX_NUM_SEQS=$RECIPE_DECODE_MAX_NUM_SEQS
# The doc gives decode fewer slots than the AgentX concurrency on purpose: an
# agentic session sits idle between turns. The client otherwise requires twice
# the concurrency.
AGENTX_ALLOW_SEQS_BELOW_CONC=1
AGENTX_CONC_LIST="$RECIPE_CONCURRENCY"
AGENTX_MODEL_PREFIX=kimik3
AGENTX_PRECISION=fp4
# The Kimi tool-call suite; it needs the tool-call parser.
AGENTX_POST_EVAL_SUITES=""
[ "$_parsers" = 0 ] || AGENTX_POST_EVAL_SUITES=kimi_tool_call_schema_full
# The doc's request path: vllm-router in its prefill/decode mode.
AGENTX_PROXY=router
# No GSM8K before the replay (DSpark answers are synthetic there); the
# eval-only job runs it.
AGENTX_GSM8K_LIMIT=0
if [ "$_no_spec" = 1 ]; then
  AGENTX_SPEC_METHOD=none
else
  AGENTX_SPEC_METHOD=dspark
fi
# The row's topology, in ix_export_topology's terms: both roles DCP32, except
# the DP2 points' decode, DCP16 with attention DP2.
AGENTX_PREFILL_DCP_SIZE=32
if [ "$RECIPE_DECODE" = dp2_dcp16 ]; then
  AGENTX_DECODE_DCP_SIZE=16 AGENTX_DECODE_DP_ATTN=true AGENTX_DECODE_NUM_WORKERS=2
else
  AGENTX_DECODE_DCP_SIZE=32 AGENTX_DECODE_DP_ATTN=false AGENTX_DECODE_NUM_WORKERS=1
fi
# The Oct 13 set's benchmark: one hour of aiperf's agentx scenario at its own
# defaults, which are the doc's options (seed 42, trajectories from 25-75% in,
# ten warmup requests per lane, 393 dataset entries, server token counts). The
# first set keeps the lane's 20-minute fast mode.
if [ "${RECIPE_SET:-}" = oct13 ]; then
  AGENTX_DURATION=3600
  AGENTX_FAST=0
fi

[ "${K3_1P1D_ENGINE:-0}" = 1 ] || return 0

# ---- engine only: the doc's placeholders on the kube fleet ---------------------
_role="${K3_1P1D_ROLE:-}"
case "$_role" in
  prefill|decode) ;;
  *) echo "ERROR: kimi-k3-megakernel: K3_1P1D_ROLE must be prefill or decode" >&2; return 2 ;;
esac
for _v in INDEX POD_IP DP_ADDRESS; do
  [ -n "${!_v:-}" ] || { echo "ERROR: kimi-k3-megakernel: $_v is unset; k3_1p1d.sh sets it" >&2; return 2; }
done
NODE_RANK="$INDEX"
NODE_IP="$POD_IP"
HEAD_IP="$DP_ADDRESS"
# The prefill head, where the Oct 13 store servers' global registry runs: this
# role's head on the prefill, PREFILL_HOST resolved on the decode.
if [ -n "${RECIPE_STORE_SERVERS:-}" ]; then
  if [ "$_role" = prefill ]; then
    PREFILL_HEAD_IP="$HEAD_IP"
  else
    PREFILL_HEAD_IP=$(resolve "${PREFILL_HOST:?the manifest names the prefill head}" 1800) || return 2
  fi
fi
PLUGIN_SRC="${REPO_DIR:-/workspace/vllm-torchtpu}/src"
LOG_DIR="${TMPDIR:-/dev/shm}/k3mk/logs"
HF_HOME_DIR="${TMPDIR:-/dev/shm}/k3mk/hf-home"
# Compilation caches, keyed by the engine image: a cache filled by one toolchain
# must not be read by another. The points run without the fleet's caches
# (common.sh), so /cache/jax is the container's own disk and empty every boot.
CACHE_ROOT="/cache/jax/k3-mk/$(printf '%s' "${K3_ENGINE_IMAGE:-unknown}" | sha1sum | cut -c1-12)"
mkdir -p "$CACHE_ROOT/$_role"/{vllm-runtime,vllm-config,torch_tpu_tier3} "$HF_HOME_DIR" "$LOG_DIR"
# The checkpoint (gs://tpu-commons-ci/moonshootai/kimi/k3, the doc's revision
# 9f62e4e9), which the manifest mounts at the path the pre-shard was written
# from. The pre-shard and the speculator snapshot are in the model bucket
# (tpu-ci-models-*), which the manifest mounts at these two paths under
# /cache/hf/hub; PROVENANCE.txt in the pre-shard records how it was made.
MODEL_DIR=/ckpt/k3
HF_HUB_CACHE_DIR=/cache/hf/hub
PRESHARDED_DIR="$HF_HUB_CACHE_DIR/presharded/kimi-k3-tp32"
DSPARK_REVISION=38a88101e0d46bb22134b9da340f381b954d40d4
DSPARK_DIR="$HF_HUB_CACHE_DIR/models--RedHatAI--Kimi-K3-speculator.dspark/snapshots/$DSPARK_REVISION"

# ---- environment shared by both roles (all points) -----------------------------
export PYTHONPATH="$PLUGIN_SRC"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
# The engine image preloads gperftools tcmalloc, which keeps freed pages in its
# heap rather than returning them to the OS. Without this the weight load's
# buffers stay resident once the weights are in HBM: ~52-54 GiB of private
# memory per worker, ~410-450 GiB a host before any KV store. Returning pages
# as they are freed halves that, ~200 GiB a host back, at no measurable serving
# cost: point d-c24 served 890 against 891 out tok/s with TPOT p50 11.07
# against 11.09 ms (vllm-torchtpu-dev builds 601 and 599). Engine init runs
# ~5% longer.
export TCMALLOC_AGGRESSIVE_DECOMMIT=true
# vLLM's Rust frontend, which the engine image carries from #1884 on.
export VLLM_USE_RUST_FRONTEND=1
export VLLM_RUST_FRONTEND_PATH=/usr/local/bin/vllm-rs
# The doc's 3600 assumes local storage. Here each host reads ~420 GB of
# pre-shard through gcsfuse before it compiles, so the K3 lanes' 4 h bound.
export VLLM_ENGINE_READY_TIMEOUT_S=14400
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_CACHE="$HF_HUB_CACHE_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_TARGET_DEVICE=tpu
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP="$NODE_IP"
export VLLM_SERVER_DEV_MODE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_HOST_BOUNDS="1,1,4"
export TPU_NODE_ID="$NODE_RANK"
export TPU_SKIP_MDS_QUERY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TPU_MOE_HIERARCHICAL_EP=1
export TPU_LATENT_PROJ_INTRA_CHIP_TP=1
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_KIMIK3_ADMISSION=1
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_RAIDEN_TRANSFER_PARALLELISM=32
export TPU_RAIDEN_PREFIX_AWARE_LOAD=1
export TPU_RAIDEN_TRANSFER_NUM_SLOTS=1
export TPU_RAIDEN_DST_SHARDS=32
export TPU_RAIDEN_POOL_STAGING_LEASES=16
export TPU_P2P_WAIT_PULL_TIMEOUT=900
export RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=1800
export TPU_RAIDEN_ADVERTISE_HOST="$HEAD_IP"
# TPU_RAIDEN_JOB_NAME and TPU_RAIDEN_ENGINE_ID are set per role below. The doc
# shares k3-mk-0 / k3-mk-0-engine between the roles, which gives prefill rank k
# and decode rank k the same Raiden work-unit id; the reshard coordinator then
# makes every decode worker a sender and arms no receiver, and every transfer
# fails ("No transfer chunks found", vllm-torchtpu-dev build 563).
export TPU_KV_TRANSFER_NAMESPACE=k3-mk-0
export TPU_LOG_DIR="$LOG_DIR/tpu"
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK=0
export NPROC=2
export TMPDIR=/dev/shm
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false"
if [ "${RECIPE_SET:-}" = oct13 ]; then
  # P-state 7, 2.2 GHz against the default P3's 1.9 GHz; outputs are
  # bit-identical.
  export LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS --xla_tpu_dvfs_p_state=7"
  # The host-store allocations bind strictly to their NUMA node.
  export RAIDEN_NUMA_POLICY=bind
fi
if [ -n "${RECIPE_STORE_SERVERS:-}" ]; then
  export PYTHONHASHSEED=0                 # same block-hash seed on every store of the pool
  export RAIDEN_ENABLE_STORE_MONITOR=1    # store heartbeats to the global registry
  export RAIDEN_ENABLE_EVICT_SWEEP=1      # demote blocks the prefill evicts to the store servers
fi

# ---- per-role environment (all points) -----------------------------------------
# The doc's ${USER} cache names; a pod has no USER.
_cache_user="${USER:-k3}"
export HF_MODULES_CACHE="$HF_HOME_DIR/modules-$_role"
export TPU_RAIDEN_JOB_NAME="k3-mk-0-$_role"
export TPU_RAIDEN_ENGINE_ID="k3-mk-0-$_role-engine"
export VLLM_XLA_CACHE_PATH="$CACHE_ROOT/$_role"
export VLLM_CACHE_ROOT="$CACHE_ROOT/$_role/vllm-runtime"
export VLLM_CONFIG_ROOT="$CACHE_ROOT/$_role/vllm-config"
export TORCH_TPU_TIER2_COMPILATION_CACHE="${_cache_user}_k3mk_$_role"
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="${_cache_user}_k3mk_$_role"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$CACHE_ROOT/$_role/torch_tpu_tier3"
if [ "$_role" = prefill ]; then
  export TPU_K3_MOE_REDUCE_DTYPE=bfloat16
  export TPU_RAIDEN_CONTROLLER_ADDRESS="$HEAD_IP:27340"
  export TPU_RAIDEN_RESHARD_PORT_BASE=27340
  export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27440
  export TPU_KV_TRANSFER_PORT=29340
  export TPU_SIDE_CHANNEL_PORT=29740
else
  export TPU_K3_MOE_REDUCE_DTYPE=float32
  export TPU_RAIDEN_CONTROLLER_ADDRESS="$HEAD_IP:28340"
  export TPU_RAIDEN_RESHARD_PORT_BASE=28340
  export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28440
  export TPU_KV_TRANSFER_PORT=30340
  export TPU_SIDE_CHANNEL_PORT=30740
  # The Oct 13 points without speculation keep the default deferred Stage-3
  # submit.
  if [ "${RECIPE_SET:-}" != oct13 ] || [ "$_no_spec" = 0 ]; then
    export TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT=0
  fi
fi

# ---- arguments shared by both roles (all points) -------------------------------
# k3_1p1d.sh adds --nnodes, --node-rank, --master-addr, and --host and --port on
# host 0 or --headless on the others.
COMMON_ARGS=(
  --served-model-name moonshotai/Kimi-K3
  --tensor-parallel-size 32 --enable-expert-parallel
  --distributed-timeout-seconds 1800
  --max-model-len 1048576
  --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512
  --trust-remote-code --language-model-only
  --mamba-cache-mode align --enable-prefix-caching
  --enable-prompt-tokens-details
)
if [ "$_parsers" = 1 ]; then
  COMMON_ARGS+=(--enable-auto-tool-choice --tool-call-parser kimi_k3 --reasoning-parser kimi_k3)
fi

# ---- per role and point ---------------------------------------------------------
# The host-memory KV store the roles add beside Raiden: the doc's 1.5 TiB
# (1649267441664), split across the 32 ranks, so ~384 GiB on each host, all of
# it allocated when the ranks register. With the fleet's gcsfuse file cache and
# tcmalloc keeping freed pages, that reached 790-850 GiB of 960 and the kubelet
# evicted the prefill head (vllm-torchtpu-dev build 562); without the caches
# (#1950) the prefill hosts peaked at 800-862 GiB (build 599), and with
# TCMALLOC_AGGRESSIVE_DECOMMIT (#1981) at 645-647 GiB of the 917.9 allocatable
# (build 601). The size matters: at 384 GiB the first set's point 4 prefill
# recomputed 3.2x the trace's minimum prompt tokens with P90 TTFT 19.2 s
# (build 588), at 768 GiB 2.6x and 7.5 s (589), at 1.5 TiB 6.2 s with 5% more
# throughput than 768 GiB (597). The doc's points without speculation (e-h)
# give the prefill 2 TiB (2199023255552, 64 GiB per rank, 512 GiB per host). A
# point may set RECIPE_OFFLOAD_BYTES.
if [ "$_no_spec" = 1 ]; then
  _offload_bytes="${RECIPE_OFFLOAD_BYTES:-2199023255552}"
else
  _offload_bytes="${RECIPE_OFFLOAD_BYTES:-1649267441664}"
fi
# The Oct 13 store servers' pool: the prefill's host store registers with the
# global registry and joins the pool group the decode-host store servers serve.
_registry_port=27900
_pool_group=k3-mk-1p1d-host-store
_registry=""
if [ -n "${RECIPE_STORE_SERVERS:-}" ]; then
  _registry=$(printf ',"global_registry_address":"%s:%s","kv_pool_group":"%s"' "$PREFILL_HEAD_IP" "$_registry_port" "$_pool_group")
fi
_offload_connector() {  # _offload_connector <raiden_controller_port>
  printf '{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":%s,"raiden_controller_port":%s,"raiden_job_name":"k3-mk-offload-0","store_server_ip":"%s"%s}}' "$_offload_bytes" "$1" "$HEAD_IP" "$_registry"
}
_multi_connector() {  # _multi_connector <kv_role> <raiden connector json> <offload connector json>
  printf '{"kv_connector":"TPUMultiConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector","kv_role":"%s","kv_connector_extra_config":{"connectors":[%s,%s]}}' "$1" "$2" "$3"
}
_raiden='"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector"'
# The first set's prefill sets --prefix-cache-retention-interval 16384; the Oct
# 13 set leaves vLLM's default, 0.
_retention=(--prefix-cache-retention-interval 16384)
[ "${RECIPE_SET:-}" != oct13 ] || _retention=()

if [ "$_role" = prefill ]; then
  if [ "$RECIPE_DECODE" = dp2_dcp16 ]; then
    # The DP2 points: no draft, and the transfer reshards for decode's DCP16.
    _loader=$(printf '{"presharded":"%s"}' "$PRESHARDED_DIR")
    _kv=$(_multi_connector kv_producer \
      "{${_raiden},\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"destination_decode_context_parallel_size\":16}}" \
      "$(_offload_connector 27800)")
    _kv_cache_bytes=23622320128
  elif [ "$RECIPE_DECODE" = dcp32 ]; then
    # Point h: the DP2 points' prefill, except that the transfer keeps DCP32.
    _loader=$(printf '{"presharded":"%s"}' "$PRESHARDED_DIR")
    _kv=$(_multi_connector kv_producer \
      "{${_raiden},\"kv_role\":\"kv_producer\"}" "$(_offload_connector 27800)")
    _kv_cache_bytes=23622320128
  else
    # Every DSpark point: the same prefill.
    _loader=$(printf '{"presharded":"%s","draft":"%s"}' "$PRESHARDED_DIR" "$DSPARK_DIR")
    _kv=$(_multi_connector kv_producer \
      "{${_raiden},\"kv_role\":\"kv_producer\"}" "$(_offload_connector 27800)")
    _kv_cache_bytes=19327352832
  fi
  ROLE_ARGS=(
    --decode-context-parallel-size 32
    --load-format megakernel --model-loader-extra-config "$_loader"
    --max-num-seqs 4 --max-num-batched-tokens 8192
    --kv-cache-memory-bytes "$_kv_cache_bytes"
    "${_retention[@]}"
    --kv-transfer-config "$_kv"
  )
  # The Oct 13 points without speculation: a long prompt takes at most 4,096
  # tokens of the 8,192-token step, so short requests share it. --numa-bind
  # pins ranks 0-3 to NUMA node 0 and 4-7 to node 1, CPUs and memory, so each
  # rank's host store lands on its own node; vLLM launches the workers under
  # numactl, which k3_1p1d.sh checks the engine image for.
  if [ "${RECIPE_SET:-}" = oct13 ] && [ "$_no_spec" = 1 ]; then
    ROLE_ARGS+=(--long-prefill-token-threshold 4096)
    ROLE_ARGS+=(--numa-bind --numa-bind-nodes 0 0 0 0 1 1 1 1)
  fi
elif [ "$RECIPE_DECODE" = dp2_dcp16 ]; then
  _loader=$(printf '{"presharded":"%s","attention_dp":2}' "$PRESHARDED_DIR")
  ROLE_ARGS=(
    --decode-context-parallel-size 16
    --load-format megakernel --model-loader-extra-config "$_loader"
    --max-num-seqs "$RECIPE_DECODE_MAX_NUM_SEQS" --max-num-batched-tokens 512
    --kv-cache-memory-bytes 30064771072
    --kv-transfer-config "{${_raiden},\"kv_role\":\"kv_consumer\"}"
  )
elif [ "$RECIPE_DECODE" = dcp32 ]; then
  # Point h: the DP2 points' decode on DCP32, without attention DP.
  _loader=$(printf '{"presharded":"%s"}' "$PRESHARDED_DIR")
  ROLE_ARGS=(
    --decode-context-parallel-size 32
    --load-format megakernel --model-loader-extra-config "$_loader"
    --max-num-seqs "$RECIPE_DECODE_MAX_NUM_SEQS" --max-num-batched-tokens 512
    --kv-cache-memory-bytes 30064771072
    --kv-transfer-config "{${_raiden},\"kv_role\":\"kv_consumer\"}"
  )
else
  # The eval-only job verifies drafts against the target; the replay accepts
  # them at the doc's synthetic length.
  if [ "$RECIPE_DECODE" = dspark ]; then
    _acceptance="$RECIPE_ACCEPTANCE"
    _drafts="\"num_speculative_tokens\":$RECIPE_DRAFTS"
  else
    _acceptance=3.78
    _drafts="\"num_speculative_tokens\":7,\"num_speculative_tokens_per_batch_size\":$RECIPE_DRAFTS_PER_BATCH_SIZE"
  fi
  if [ "$_eval_job" = 1 ]; then
    _loader=$(printf '{"presharded":"%s"}' "$PRESHARDED_DIR")
    _synthetic=""
  else
    _loader=$(printf '{"presharded":"%s","synthetic_acceptance_lengths":{"3":2.87,"7":3.78}}' "$PRESHARDED_DIR")
    _synthetic=",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$_acceptance"
  fi
  _spec=$(printf '{"method":"dspark","model":"%s",%s%s}' "$DSPARK_DIR" "$_drafts" "$_synthetic")
  if [ -n "${RECIPE_STORE_SERVERS:-}" ]; then
    # The store servers' pool replaces the decode's own host store.
    _kv="{${_raiden},\"kv_role\":\"kv_consumer\"}"
  else
    _kv=$(_multi_connector kv_consumer \
      "{${_raiden},\"kv_role\":\"kv_consumer\"}" "$(_offload_connector 28800)")
  fi
  ROLE_ARGS=(
    --decode-context-parallel-size 32
    --load-format megakernel --model-loader-extra-config "$_loader"
    --max-num-seqs "$RECIPE_DECODE_MAX_NUM_SEQS" --max-num-batched-tokens 512
    --kv-cache-memory-bytes 19327352832
    --speculative-config "$_spec"
    --kv-transfer-config "$_kv"
  )
fi
SERVE_ARGS=("$MODEL_DIR" "${COMMON_ARGS[@]}" "${ROLE_ARGS[@]}")

# The Oct 13 store servers, which k3_1p1d.sh starts: the global registry on the
# prefill head before its serve, and a host-store server on each decode host
# once the decode head serves, as the doc starts them after the engines are up.
# Each server lends the doc's 512 GiB (549755813888). Before
# TCMALLOC_AGGRESSIVE_DECOMMIT (#1981) a decode host without its own host store
# held ~410-450 GiB of workers, and 512 GiB servers put the hosts at 868-917
# GiB of the 917.9 allocatable and the kubelet evicted two of them (build 598);
# with it the workers hold ~210 GiB and a 384 GiB server left the hosts at
# 599-601 GiB (build 601), so 512 GiB comes to roughly 730 GiB. A point may
# set RECIPE_STORE_NODE_BYTES.
if [ -n "${RECIPE_STORE_SERVERS:-}" ]; then
  K3_1P1D_REGISTRY_ADDRESS="$PREFILL_HEAD_IP:$_registry_port"
  if [ "$_role" = prefill ]; then
    K3_1P1D_REGISTRY_ARGS=(--port="$_registry_port")
  else
    K3_1P1D_STORE_NODE_ARGS=(
      --dram_budget_bytes="${RECIPE_STORE_NODE_BYTES:-549755813888}"
      --global_registry_address="$K3_1P1D_REGISTRY_ADDRESS"
      --kv_pool_group="$_pool_group"
      --store_server_ip="$NODE_IP"
      --job_name=k3-host-store-node --job_replica_id="$NODE_RANK"
      --parallelism=16
    )
  fi
fi

# What k3_1p1d.sh checks before a boot that takes minutes to fail on them.
K3_1P1D_INPUTS=("$MODEL_DIR/config.json" "$PRESHARDED_DIR/layout.json" "$PRESHARDED_DIR/prefill-sp-cache/layout.json")
if [ "$_no_spec" = 0 ]; then
  K3_1P1D_INPUTS+=("$DSPARK_DIR/config.json" "$PRESHARDED_DIR/dspark-${DSPARK_REVISION:0:8}")
fi
K3_1P1D_IMPORTS=(vllm_torchtpu.megakernel)
unset _v _role _cache_user _offload_bytes _raiden _loader _kv _kv_cache_bytes _spec
unset _registry _registry_port _pool_group _retention _no_spec _synthetic _acceptance _drafts _eval_job _parsers
unset -f _offload_connector _multi_connector
