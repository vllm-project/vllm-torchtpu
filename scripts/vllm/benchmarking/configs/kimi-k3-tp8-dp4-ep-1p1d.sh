#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by k3_1p1d.sh
# One engine of the Kimi-K3 1P1D pair: TP8 x DP4 x EP32 on a 2x2x4 slice, one
# DP replica per host, over vLLM's own multi-node DP. The nightly
# kimi-k3-tp32-ep model and serve flags, with what Raiden Stage 3 and the
# correctness stages (scripts/vllm/integration/run_k3_1p1d_disagg_correctness.sh)
# need on top. K3_1P1D_ROLE, prefill or decode, selects the engine.
# VLLM_HOST_IP and TPU_RAIDEN_ADVERTISE_HOST come from k3_1p1d.sh, which knows
# the pod addresses.
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-tp32-ep.sh"

TENSOR_PARALLELISM=8
DATA_PARALLELISM=4
# What the connector's DP support (#1271) was validated with.
GPU_MEMORY_UTILIZATION=0.80

# ---- serve flags --------------------------------------------------------------
# fp8, as the nightly serves; K3_1P1D_KV_CACHE_DTYPE=auto selects bf16.
KV_CACHE_DTYPE="${K3_1P1D_KV_CACHE_DTYPE:-fp8}"
# The Kimi pool manifest needs KDA state blocks to match attention blocks,
# which only prefix caching gives.
ENABLE_PREFIX_CACHING=true

# One prefill chunk is one KV block, so a request resumed from a prefix hit
# runs the cold run's chunk shapes and the stages can compare tokens. The block
# is the one the engine settles on, four times the TP32 pair's for a TP8 rank's
# KDA state: 648 for bf16 (observed), 1296 for fp8 (derived from TP32's 324);
# the stages verify it against /metrics.
if [ "$KV_CACHE_DTYPE" = "fp8" ]; then
    MAX_NUM_BATCHED_TOKENS=1296
else
    MAX_NUM_BATCHED_TOKENS=648
fi
_before="$EXTRA_SERVE_ARGS"
EXTRA_SERVE_ARGS=$(printf '%s' "$EXTRA_SERVE_ARGS" | sed "s/\"compile_sizes\":\[1,8,8192\]/\"compile_sizes\":[1,8,${MAX_NUM_BATCHED_TOKENS}]/")
if [ "$EXTRA_SERVE_ARGS" = "$_before" ]; then
    echo "ERROR: kimi-k3-tp8-dp4-ep-1p1d.sh: the nightly config no longer sets compile_sizes [1,8,8192]; update the substitution above" >&2
    return 1
fi
# --enable-prompt-tokens-details: the stages read usage.cached_tokens.
# --reasoning-parser: without it a chat response carries K3's channel-closing
# markers as text.
EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS --mamba-cache-mode align --enable-prompt-tokens-details --reasoning-parser kimi_k3 --distributed-executor-backend mp"

# ---- Raiden Stage 3, both engines --------------------------------------------
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_KIMIK3_ADMISSION=1
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
# Kimi admission: the producer's TP and the decode's TP, on both engines.
export TPU_RAIDEN_TRANSFER_PARALLELISM="$TENSOR_PARALLELISM"
export TPU_RAIDEN_DST_SHARDS="$TENSOR_PARALLELISM"
# Decode keeps a local prefix hit and pulls the suffix alone.
export TPU_RAIDEN_PREFIX_AWARE_LOAD=1
export TPU_KV_TRANSFER_PORT=9100
export TPU_SIDE_CHANNEL_PORT=9600
export TPU_KV_SHM_POOL_GB=8
export TPU_PREMAPPED_BUFFER_SIZE=17179869184
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-14400}"
# The rest of what #1271 was validated with.
export TPU_MOE_HIERARCHICAL_EP=1
export TPU_LATENT_PROJ_INTRA_CHIP_TP=1
export TPU_RAIDEN_TRANSFER_NUM_SLOTS=1
export TPU_PARALLEL_PRECOMPILE=1
export TPU_P2P_WAIT_PULL_TIMEOUT=900
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800

# ---- per engine: identity and ports, which must differ between the two -------
case "${K3_1P1D_ROLE:-}" in
    prefill)
        _kv_role=kv_producer
        export TPU_KV_TRANSFER_NAMESPACE=k3_prefill
        export TPU_RAIDEN_ENGINE_ID=k3-prefill-engine
        export TPU_RAIDEN_JOB_NAME=prefill
        export TPU_RAIDEN_RESHARD_PORT_BASE=27000
        export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100
        ;;
    decode)
        _kv_role=kv_consumer
        export TPU_KV_TRANSFER_NAMESPACE=k3_decode
        export TPU_RAIDEN_ENGINE_ID=k3-decode-engine
        export TPU_RAIDEN_JOB_NAME=decode
        export TPU_RAIDEN_RESHARD_PORT_BASE=28000
        export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28100
        ;;
    *)
        echo "ERROR: kimi-k3-tp8-dp4-ep-1p1d.sh: K3_1P1D_ROLE must be prefill or decode, got '${K3_1P1D_ROLE:-}'" >&2
        return 1
        ;;
esac
# No spaces in the JSON: k3_1p1d.sh word-splits the command it builds.
_kv_transfer_config="{\"kv_connector\":\"TPURaidenConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_connector\",\"kv_role\":\"${_kv_role}\",\"kv_port\":14579}"
# Decode also offloads to host memory, behind TPUMultiConnector as in the
# Qwen3.5 p4d2 pair. Every child must accept a decode-local prefix hit that the
# recurrent state lags, or prefix-aware loads turn off for the whole engine,
# so the counters stage (decode local_cache_hit > 0) guards that too.
if [ "$_kv_role" = kv_consumer ]; then
    _offload_config="{\"kv_connector\":\"TPURaidenOffloadingConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.offload.raiden_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${K3_1P1D_DECODE_CPU_BYTES_TO_USE:-68719476736},\"raiden_controller_port\":48800}}"
    _kv_transfer_config="{\"kv_connector\":\"TPUMultiConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"connectors\":[${_kv_transfer_config},${_offload_config}]}}"
fi
EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS --kv-transfer-config ${_kv_transfer_config}"
