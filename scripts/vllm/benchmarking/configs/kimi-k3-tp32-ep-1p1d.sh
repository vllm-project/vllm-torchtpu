#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# One engine of the Kimi-K3 1P1D pair: the nightly kimi-k3-tp32-ep model,
# sharding and serve flags, with what Raiden Stage 3 and the correctness stages
# (scripts/vllm/integration/run_k3_1p1d_disagg_correctness.sh) need on top.
#
# K3_1P1D_ROLE, prefill or decode, selects the engine. Sourced twice per
# engine: by .buildkite/kubernetes/k3_1p1d.sh on every host before Ray starts,
# for the environment below, and by run_benchmarks.sh on the head, for the
# serve command. What only a running pod knows stays in k3_1p1d.sh:
# VLLM_HOST_IP and TPU_RAIDEN_ADVERTISE_HOST.
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-tp32-ep.sh"

# ---- serve flags --------------------------------------------------------------
# Raiden's Kimi pool manifest admits only a bf16 cache whose KDA state blocks
# are the attention block size. The nightly serves fp8 without prefix caching,
# and without it vLLM sizes a state block to the whole max-model-len; with it
# the state blocks align to the attention blocks.
KV_CACHE_DTYPE="${K3_1P1D_KV_CACHE_DTYPE:-auto}"
ENABLE_PREFIX_CACHING=true

# One prefill chunk is one KV block, the nightly's --block-size 256. K3's
# greedy output moves with prefill chunk geometry, and hybrid prefix caching
# cuts chunks at block boundaries, so only with chunk == block does a request
# resumed from a prefix hit run the chunk shapes of the cold one - which is
# what lets the correctness stages compare their tokens.
MAX_NUM_BATCHED_TOKENS=256

_before="$EXTRA_SERVE_ARGS"
EXTRA_SERVE_ARGS=$(printf '%s' "$EXTRA_SERVE_ARGS" | sed 's/"compile_sizes":\[1,8,8192\]/"compile_sizes":[1,8,256]/')
if [ "$EXTRA_SERVE_ARGS" = "$_before" ]; then
    echo "ERROR: kimi-k3-tp32-ep-1p1d.sh: the nightly config no longer sets compile_sizes [1,8,8192]; update the substitution above" >&2
    return 1
fi
# --enable-prompt-tokens-details: the stages read usage.cached_tokens.
# --reasoning-parser: K3 ends an answer by closing its response and message
# channels, and without its parser a chat response carries those markers as
# text ("4 <|close|> response <|sep|> <|close|> message <|sep|>").
EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS --mamba-cache-mode align --enable-prompt-tokens-details --reasoning-parser kimi_k3"

# ---- Raiden Stage 3, both engines --------------------------------------------
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_KIMIK3_ADMISSION=1
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
# Kimi admission requires the producer's TP and the decode's TP here, on both
# engines: TP32 into TP32.
export TPU_RAIDEN_TRANSFER_PARALLELISM="$TENSOR_PARALLELISM"
export TPU_RAIDEN_DST_SHARDS="$TENSOR_PARALLELISM"
# Decode keeps a local prefix hit and pulls the suffix alone, rather than the
# whole payload into every page, the shared ones included.
export TPU_RAIDEN_PREFIX_AWARE_LOAD=1
export TPU_KV_TRANSFER_PORT=9100
export TPU_SIDE_CHANNEL_PORT=9600
export TPU_KV_SHM_POOL_GB=8
export TPU_P2P_WAIT_PULL_TIMEOUT=600
export TPU_PREMAPPED_BUFFER_SIZE=17179869184
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-14400}"

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
        echo "ERROR: kimi-k3-tp32-ep-1p1d.sh: K3_1P1D_ROLE must be prefill or decode, got '${K3_1P1D_ROLE:-}'" >&2
        return 1
        ;;
esac
# No spaces in the JSON: run_benchmarks.sh word-splits the command it builds.
EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS --kv-transfer-config {\"kv_connector\":\"TPURaidenConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_connector\",\"kv_role\":\"${_kv_role}\",\"kv_port\":14579}"
