#!/usr/bin/env bash
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

# What every Kimi-K3 TPU7x 1P1D recipe point shares, as the recipe document
# states it: one prefill and one decode engine, each TP32 x EP over its own
# 2x2x4 slice via vLLM's `--nnodes` multi-host mode, the megakernel loader on
# the TP32 pre-shard, fp8 KV. A point is a file beside this one that states
# what differs (its concurrency, decode batch size, DSpark draft schedule) and
# sources this; K3_PD_CONFIG=kimi-k3-megakernel/<point> serves it.
#
# A transcription, not a config in run_benchmarks.sh's sense: the serve
# arguments and the environment are the document's, so that what runs here is
# what the recipe's numbers were measured with. Only the placeholders are
# filled in (the paths the fleet mounts, this pod's addresses) and nothing of
# ours is added; a flag that turns out to be needed on the fleet is a finding
# to report. The one liberty is the API port: both roles serve on k3_1p1d.sh's
# 8000 rather than the document's 18540/19540, each on its own hosts.
#
# Served by the AgentX pair (kimi-k3-agentx-1p1d.yaml), so it speaks to the
# same two readers as kimi-k3-tp8-dp4-ep-agentx.sh:
#   - k3_1p1d.sh, with K3_1P1D_ROLE and HEAD_IP, NODE_IP, NODE_RANK set:
#     K3_1P1D_LAUNCH=nnodes, the role's environment, SERVE_ARGS complete, and
#     K3_1P1D_INPUTS, the files the loader needs.
#   - k3_agentx_client_pod.sh, with K3_1P1D_ROLE=decode and no addresses: the
#     geometry, the point's concurrency and the AGENTX_* labels.
# Not kimi-k3-tp32-ep.sh's: that is main's K3, this is the recipe's.
#
# A point states, before sourcing this:
#   RECIPE_CONCURRENCY           the AgentX concurrency, one per point
#   RECIPE_DECODE_MAX_NUM_SEQS   decode's --max-num-seqs
#   RECIPE_DECODE_DRAFTS, RECIPE_DECODE_ACCEPTANCE
#                                DSpark drafts per step and the synthetic
#                                acceptance length for that width; or
#   RECIPE_DRAFTS_PER_BATCH_SIZE [[min_batch,max_batch,drafts],...], the
#                                width following the batch size; or neither,
#                                for no speculation
# and may override RECIPE_DECODE_DCP (32), RECIPE_ATTENTION_DP (none),
# RECIPE_PREFILL_KV_BYTES and RECIPE_DECODE_KV_BYTES (18 GiB).

# shellcheck disable=SC2034  # read by the scripts that source this file
case "${K3_1P1D_ROLE:-}" in prefill|decode) ;;
    *) echo "recipe-base.sh: K3_1P1D_ROLE must be prefill or decode, got '${K3_1P1D_ROLE:-}'" >&2; return 1 ;;
esac
_point=$(basename "${BASH_SOURCE[1]:-}" .sh)
for _v in RECIPE_CONCURRENCY RECIPE_DECODE_MAX_NUM_SEQS; do
    [ -n "${!_v:-}" ] || { echo "recipe-base.sh: the point ($_point) does not set $_v" >&2; return 1; }
done

# ---- The point ---------------------------------------------------------------
_concurrency="$RECIPE_CONCURRENCY"
_decode_max_num_seqs="$RECIPE_DECODE_MAX_NUM_SEQS"
_decode_drafts="${RECIPE_DECODE_DRAFTS:-}"
_decode_acceptance="${RECIPE_DECODE_ACCEPTANCE:-}"
_drafts_per_batch_size="${RECIPE_DRAFTS_PER_BATCH_SIZE:-}"
_decode_dcp="${RECIPE_DECODE_DCP:-32}"
_attention_dp="${RECIPE_ATTENTION_DP:-}"
_prefill_kv_bytes="${RECIPE_PREFILL_KV_BYTES:-19327352832}"
_decode_kv_bytes="${RECIPE_DECODE_KV_BYTES:-19327352832}"
_spec_method=dspark
[ -z "$_decode_drafts$_drafts_per_batch_size" ] && _spec_method=none
if [ "$_spec_method" = dspark ] && [ -z "$_drafts_per_batch_size" ] && [ -z "$_decode_acceptance" ]; then
    echo "recipe-base.sh: $_point sets RECIPE_DECODE_DRAFTS without RECIPE_DECODE_ACCEPTANCE" >&2; return 1
fi

# ---- Placeholders ------------------------------------------------------------
# The document's $MODEL_DIR, $PRESHARDED_DIR and $DSPARK_DIR.
MODEL="moonshotai/Kimi-K3"
# The checkpoint, for its config, tokenizer and remote code; the weights come
# from the pre-shard. The manifest mounts gs://tpu-commons-ci/moonshootai/kimi/k3.
K3_RECIPE_MODEL_DIR="${K3_RECIPE_MODEL_DIR:-/ckpt/k3}"
# Written by the pre-shard job (k3_preshard.sh) into the fleet's model bucket.
K3_RECIPE_PRESHARDED_DIR="${K3_RECIPE_PRESHARDED_DIR:-/cache/hf/hub/presharded/kimi-k3-tp32}"
K3_RECIPE_DSPARK_DIR="${K3_RECIPE_DSPARK_DIR:-/cache/hf/hub/models--RedHatAI--Kimi-K3-speculator.dspark/snapshots/38a88101e0d46bb22134b9da340f381b954d40d4}"

# ---- What the pair's scripts read --------------------------------------------
# The geometry, in the vocabulary of kimi-k3-tp8-dp4-ep-1p1d.sh: one TP32 rank
# group per role across its four hosts, no DP.
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
KV_CACHE_DTYPE=fp8
if [ "$K3_1P1D_ROLE" = prefill ]; then
    MAX_NUM_SEQS=4
    MAX_NUM_BATCHED_TOKENS=8192
else
    MAX_NUM_SEQS="$_decode_max_num_seqs"
    MAX_NUM_BATCHED_TOKENS=512
fi
# The engines launch as the document does: the same `vllm serve` on every
# host, `--nnodes 4`, not the DP head/headless pair.
K3_1P1D_LAUNCH=nnodes

# One point is one concurrency; the labels InferenceX files the row under.
AGENTX_CONC_LIST="$_concurrency"
AGENTX_MODEL_PREFIX=kimik3
AGENTX_PRECISION=fp4
AGENTX_LABEL="agentx_kimi-k3-megakernel_${_point}"
AGENTX_SPEC_METHOD="$_spec_method"
# The document's router in front of the pair, not the toy proxy.
AGENTX_PROXY=router
# vllm-rs's /v1/models does not report the window; aiperf needs it.
AGENTX_MAX_MODEL_LEN=1048576
# The recipe's run is the replay alone: no GSM8K and no post-replay suite.
AGENTX_GSM8K_LIMIT=0
AGENTX_POST_EVAL_SUITES=""
# A decode that admits fewer sequences than the point's concurrency (points
# 3, 4, 6, 7) queues the rest on purpose.
AGENTX_ALLOW_QUEUEING=1
# The shape the row is filed under: decode's DCP, and on point 5 the two
# attention groups inside the TP32 rank group.
AGENTX_DECODE_DCP_SIZE="$_decode_dcp"
AGENTX_DECODE_ATTN_DP="${_attention_dp:-1}"

# Only the engines go past here: k3_1p1d.sh sets the addresses.
if [ -z "${HEAD_IP:-}" ]; then
    unset _prefill_kv_bytes _decode_kv_bytes _decode_dcp _attention_dp _decode_drafts \
        _decode_acceptance _drafts_per_batch_size _concurrency _decode_max_num_seqs _spec_method _point _v
    return 0
fi
: "${NODE_IP:?the IP of this host}" "${NODE_RANK:?0..3}"

# ---- Environment shared by both roles (all points) ---------------------------
# The document's list. Left out, with the reason:
#   PYTHONPATH=$PLUGIN_SRC            the image sets it
#   VLLM_RUST_FRONTEND_PATH=$VLLM_RS_BIN  the image sets it (docker/Dockerfile)
#   HF_HOME, HF_HUB_CACHE             the fleet's /cache/hf, set by the pod
#   TPU_CHIPS_PER_HOST_BOUNDS, TPU_HOST_BOUNDS, TPU_NODE_ID
#                                     the fleet describes the slice to the pod
#   TMPDIR=/dev/shm                   the manifest sets it
_cache_root="${K3_RECIPE_CACHE_ROOT:-/cache/jax/k3-recipe}/${K3_1P1D_ROLE}"
_log_dir="${ARTIFACTS_DIR:-/tmp/k3-recipe}/${K3_1P1D_ROLE}"
mkdir -p "$_cache_root"/{vllm-runtime,vllm-config,torch_tpu_tier3} "$_log_dir/tpu" "/tmp/k3-recipe/modules-${K3_1P1D_ROLE}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export VLLM_USE_RUST_FRONTEND=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_TARGET_DEVICE=tpu
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP="$NODE_IP"
export VLLM_SERVER_DEV_MODE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800
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
export TPU_RAIDEN_JOB_NAME=k3-mk-0
export TPU_RAIDEN_ENGINE_ID=k3-mk-0-engine
export TPU_KV_TRANSFER_NAMESPACE=k3-mk-0
export TPU_LOG_DIR="$_log_dir/tpu"
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK=0
export NPROC=2
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false"
export HF_MODULES_CACHE="/tmp/k3-recipe/modules-${K3_1P1D_ROLE}"
export VLLM_XLA_CACHE_PATH="$_cache_root"
export VLLM_CACHE_ROOT="$_cache_root/vllm-runtime"
export VLLM_CONFIG_ROOT="$_cache_root/vllm-config"
export TORCH_TPU_TIER2_COMPILATION_CACHE="k3_recipe_${K3_1P1D_ROLE}"
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="k3_recipe_${K3_1P1D_ROLE}"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$_cache_root/torch_tpu_tier3"

# ---- Per-role environment (all points) ---------------------------------------
if [ "$K3_1P1D_ROLE" = prefill ]; then
    export TPU_K3_MOE_REDUCE_DTYPE=bfloat16
    export TPU_RAIDEN_CONTROLLER_ADDRESS="$HEAD_IP:27340"
    export TPU_RAIDEN_RESHARD_PORT_BASE=27340
    export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27440
    export TPU_KV_TRANSFER_PORT=29340
    export TPU_SIDE_CHANNEL_PORT=29740
    _offload_controller_port=27800
else
    export TPU_K3_MOE_REDUCE_DTYPE=float32
    export TPU_RAIDEN_CONTROLLER_ADDRESS="$HEAD_IP:28340"
    export TPU_RAIDEN_RESHARD_PORT_BASE=28340
    export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28440
    export TPU_KV_TRANSFER_PORT=30340
    export TPU_SIDE_CHANNEL_PORT=30740
    export TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT=0
    _offload_controller_port=28800
fi

# ---- Arguments shared by both roles (all points) -----------------------------
# k3_1p1d.sh adds --nnodes, --node-rank, --master-addr, --host and --port.
_common_args=(
    --served-model-name "$MODEL"
    --tensor-parallel-size 32 --enable-expert-parallel
    --distributed-timeout-seconds 1800
    --max-model-len 1048576
    --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512
    --trust-remote-code --language-model-only
    --mamba-cache-mode align --enable-prefix-caching
    --enable-prompt-tokens-details
)

# The KV connectors, as the document's printf templates. Both roles run the
# TPUMultiConnector (Raiden transfer + Raiden offloading store) except point
# 5's decode, which runs the bare TPURaidenConnector.
_raiden_connector() {  # <kv_role> [extra-json]
    printf '{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"%s"%s}' "$1" "${2:+,\"kv_connector_extra_config\":$2}"
}
_multi_connector() {  # <kv_role> <offload controller port> [raiden extra-json]
    printf '{"kv_connector":"TPUMultiConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector","kv_role":"%s","kv_connector_extra_config":{"connectors":[%s,{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":1649267441664,"raiden_controller_port":%s,"raiden_job_name":"k3-mk-offload-0","store_server_ip":"%s"}}]}}' \
        "$1" "$(_raiden_connector "$1" "${3:-}")" "$2" "$HEAD_IP"
}

# The files the loader opens, checked before anything boots.
K3_1P1D_INPUTS=(
    "$K3_RECIPE_MODEL_DIR/config.json"
    "$K3_RECIPE_PRESHARDED_DIR/prefill-sp-cache/layout.json"
)
if [ "$_spec_method" = dspark ]; then
    _dspark_rev=$(basename "$K3_RECIPE_DSPARK_DIR")
    K3_1P1D_INPUTS+=(
        "$K3_RECIPE_PRESHARDED_DIR/dspark-${_dspark_rev:0:8}/index.json"
        "$K3_RECIPE_DSPARK_DIR/config.json"
    )
fi

if [ "$K3_1P1D_ROLE" = prefill ]; then
    if [ "$_spec_method" = dspark ]; then
        _loader_cfg=$(printf '{"presharded":"%s","draft":"%s"}' "$K3_RECIPE_PRESHARDED_DIR" "$K3_RECIPE_DSPARK_DIR")
        _kv_cfg=$(_multi_connector kv_producer "$_offload_controller_port")
    else
        # Point 5: no draft, and the transfer connector is told decode's DCP.
        _loader_cfg=$(printf '{"presharded":"%s"}' "$K3_RECIPE_PRESHARDED_DIR")
        _kv_cfg=$(_multi_connector kv_producer "$_offload_controller_port" \
            "$(printf '{"destination_decode_context_parallel_size":%s}' "$_decode_dcp")")
    fi
    SERVE_ARGS=(
        "$K3_RECIPE_MODEL_DIR" "${_common_args[@]}"
        --decode-context-parallel-size 32
        --load-format megakernel --model-loader-extra-config "$_loader_cfg"
        --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --kv-cache-memory-bytes "$_prefill_kv_bytes"
        --prefix-cache-retention-interval 16384
        --kv-transfer-config "$_kv_cfg"
    )
else
    _spec_args=()
    if [ "$_spec_method" = dspark ]; then
        _loader_cfg=$(printf '{"presharded":"%s","synthetic_acceptance_lengths":{"3":2.87,"7":3.78}}' "$K3_RECIPE_PRESHARDED_DIR")
        if [ -n "$_drafts_per_batch_size" ]; then
            # Points 6/7: the width follows the batch size; the 7-draft
            # acceptance length here must equal the one in the loader config.
            _spec_cfg=$(printf '{"method":"dspark","model":"%s","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":%s,"rejection_sample_method":"synthetic","synthetic_acceptance_length":3.78}' \
                "$K3_RECIPE_DSPARK_DIR" "$_drafts_per_batch_size")
        else
            _spec_cfg=$(printf '{"method":"dspark","model":"%s","num_speculative_tokens":%s,"rejection_sample_method":"synthetic","synthetic_acceptance_length":%s}' \
                "$K3_RECIPE_DSPARK_DIR" "$_decode_drafts" "$_decode_acceptance")
        fi
        _spec_args=(--speculative-config "$_spec_cfg")
        _kv_cfg=$(_multi_connector kv_consumer "$_offload_controller_port")
    else
        _loader_cfg=$(printf '{"presharded":"%s","attention_dp":%s}' "$K3_RECIPE_PRESHARDED_DIR" "$_attention_dp")
        _kv_cfg=$(_raiden_connector kv_consumer)
    fi
    SERVE_ARGS=(
        "$K3_RECIPE_MODEL_DIR" "${_common_args[@]}"
        --decode-context-parallel-size "$_decode_dcp"
        --load-format megakernel --model-loader-extra-config "$_loader_cfg"
        --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --kv-cache-memory-bytes "$_decode_kv_bytes"
        "${_spec_args[@]}"
        --kv-transfer-config "$_kv_cfg"
    )
fi
unset _prefill_kv_bytes _decode_kv_bytes _decode_dcp _attention_dp _decode_drafts \
    _decode_acceptance _drafts_per_batch_size _concurrency _decode_max_num_seqs _spec_method \
    _cache_root _log_dir _offload_controller_port _common_args _loader_cfg _kv_cfg _spec_cfg \
    _spec_args _dspark_rev _point _v
