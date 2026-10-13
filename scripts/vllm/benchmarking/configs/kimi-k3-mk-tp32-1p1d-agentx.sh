#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by k3_mk_engine.sh and k3_agentx_client_pod.sh
# Kimi-K3 1P1D on the megakernel backend, served as the "Kimi-K3 on TPU7x: 1P1D
# prefill/decode recipes" doc serves it - each role TP32 on its own 2x2x4 slice,
# presharded TP32 weights, FP8 KV in 512-token blocks, Raiden prefill-to-decode
# transfer and a host-memory KV store, behind vLLM's Rust frontend - and driven
# by the AgentX lane's own client (k3_agentx_client_pod.sh: the toy proxy and the
# InferenceX replay), not the doc's router and aiperf build.
#
# Two readers:
#   k3_agentx_client_pod.sh  sources it with K3_1P1D_ROLE=decode for the shape,
#                            MAX_NUM_SEQS and the AGENTX_* cases below, and
#                            stops at the K3_MK_ENGINE guard
#   k3_mk_engine.sh          sets K3_MK_ENGINE=1, K3_1P1D_ROLE and the doc's
#                            per-host variables (NODE_RANK, NODE_IP, HEAD_IP,
#                            CACHE_ROOT, LOG_DIR, HF_HOME_DIR, PLUGIN_SRC), and
#                            gets the doc's environment and arguments as well
# K3_MK_POINT (1-5, 6-7 experimental) picks the doc's point; one point per boot,
# since each needs its own decode server.
#
# Not here, as in the doc: --enable-auto-tool-choice, --tool-call-parser and
# --reasoning-parser, and so not the Kimi tool-call post-eval either.

# ---- the point: the doc's table -------------------------------------------------
# DECODE_MODE: dspark (points 1-4), dspark_mixed (6-7, experimental: the number
# of drafts follows the decode batch size), dp2_dcp16 (5: no speculation).
K3_MK_POINT="${K3_MK_POINT:-1}"
case "$K3_MK_POINT" in
  1) DECODE_MODE=dspark DECODE_MAX_NUM_SEQS=1  NUM_DRAFTS=7 ACCEPTANCE_LENGTH=3.78 CONCURRENCY=1 ;;
  2) DECODE_MODE=dspark DECODE_MAX_NUM_SEQS=2  NUM_DRAFTS=7 ACCEPTANCE_LENGTH=3.78 CONCURRENCY=4 ;;
  3) DECODE_MODE=dspark DECODE_MAX_NUM_SEQS=4  NUM_DRAFTS=7 ACCEPTANCE_LENGTH=3.78 CONCURRENCY=12 ;;
  4) DECODE_MODE=dspark DECODE_MAX_NUM_SEQS=16 NUM_DRAFTS=3 ACCEPTANCE_LENGTH=2.87 CONCURRENCY=32 ;;
  5) DECODE_MODE=dp2_dcp16 DECODE_MAX_NUM_SEQS=96 CONCURRENCY=64 ;;
  6) DECODE_MODE=dspark_mixed DECODE_MAX_NUM_SEQS=8 DRAFTS_PER_BATCH_SIZE='[[1,4,7],[5,8,3]]' CONCURRENCY=24 ;;
  7) DECODE_MODE=dspark_mixed DECODE_MAX_NUM_SEQS=4 DRAFTS_PER_BATCH_SIZE='[[1,1,7],[2,4,3]]' CONCURRENCY=16 ;;
  *) echo "ERROR: kimi-k3-mk-tp32-1p1d-agentx.sh: K3_MK_POINT must be 1-7, not '$K3_MK_POINT'" >&2
     return 2 ;;
esac

# ---- what the AgentX client reads ----------------------------------------------
# Each role is TP32 with EP over all 32 ranks and no DP.
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
# The decode engine's limit, which is what bounds the replay.
MAX_NUM_SEQS=$DECODE_MAX_NUM_SEQS
# The doc gives decode fewer slots than the AgentX concurrency on purpose: an
# agentic session sits idle between turns. The client otherwise requires twice
# the concurrency.
AGENTX_ALLOW_SEQS_BELOW_CONC=1
AGENTX_CONC_LIST="$CONCURRENCY"
AGENTX_MODEL_PREFIX=kimik3
AGENTX_PRECISION=fp4
# The Kimi tool-call suite needs the tool-call parser this recipe does not run.
AGENTX_POST_EVAL_SUITES=""

[ "${K3_MK_ENGINE:-0}" = 1 ] || return 0

# ---- engine only: the doc's placeholders on the kube fleet ---------------------
K3_MK_ROLE="${K3_1P1D_ROLE:-}"
case "$K3_MK_ROLE" in
  prefill|decode) ;;
  *) echo "ERROR: kimi-k3-mk-tp32-1p1d-agentx.sh: K3_1P1D_ROLE must be prefill or decode" >&2
     return 2 ;;
esac
for _v in NODE_RANK NODE_IP HEAD_IP CACHE_ROOT LOG_DIR HF_HOME_DIR PLUGIN_SRC; do
  [ -n "${!_v:-}" ] || { echo "ERROR: kimi-k3-mk-tp32-1p1d-agentx.sh: $_v is unset" >&2; return 2; }
done
# The checkpoint (gs://tpu-commons-ci/moonshootai/kimi/k3, the doc's revision
# 9f62e4e9) mounted at the path the pre-shard was written from. The pre-shard
# and the speculator snapshot are in the model bucket every TPU pod mounts at
# /cache/hf/hub (tpu-ci-models-*); PROVENANCE.txt there records how it was made.
MODEL_DIR="${K3_MK_MODEL_DIR:-/ckpt/k3}"
HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-/cache/hf/hub}"
PRESHARDED_DIR="${K3_MK_PRESHARDED_DIR:-$HF_HUB_CACHE_DIR/presharded/kimi-k3-tp32}"
K3_MK_DSPARK_REVISION=38a88101e0d46bb22134b9da340f381b954d40d4
DSPARK_DIR="$HF_HUB_CACHE_DIR/models--RedHatAI--Kimi-K3-speculator.dspark/snapshots/$K3_MK_DSPARK_REVISION"
# Both roles on the port the AgentX client and its proxy address.
K3_MK_PORT=8000

# ---- environment shared by both roles (all points) -----------------------------
export PYTHONPATH="$PLUGIN_SRC"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
# vLLM's Rust frontend, which the engine image carries from #1884 on.
export VLLM_USE_RUST_FRONTEND=1
export VLLM_RUST_FRONTEND_PATH="${VLLM_RS_BIN:-/usr/local/bin/vllm-rs}"
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

# ---- per-role environment (all points) -----------------------------------------
# The doc's ${USER} cache names; a pod has no USER.
_cache_user="${USER:-k3mk}"
if [ "$K3_MK_ROLE" = prefill ]; then
  export HF_MODULES_CACHE="$HF_HOME_DIR/modules-prefill"
  export TPU_RAIDEN_JOB_NAME=k3-mk-0-prefill
  export TPU_RAIDEN_ENGINE_ID=k3-mk-0-prefill-engine
  export TPU_K3_MOE_REDUCE_DTYPE=bfloat16
  export TPU_RAIDEN_CONTROLLER_ADDRESS="$HEAD_IP:27340"
  export TPU_RAIDEN_RESHARD_PORT_BASE=27340
  export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27440
  export TPU_KV_TRANSFER_PORT=29340
  export TPU_SIDE_CHANNEL_PORT=29740
  export VLLM_XLA_CACHE_PATH="$CACHE_ROOT/prefill"
  export VLLM_CACHE_ROOT="$CACHE_ROOT/prefill/vllm-runtime"
  export VLLM_CONFIG_ROOT="$CACHE_ROOT/prefill/vllm-config"
  export TORCH_TPU_TIER2_COMPILATION_CACHE="${_cache_user}_k3_prefill"
  export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="${_cache_user}_k3_prefill"
  export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$CACHE_ROOT/prefill/torch_tpu_tier3"
else
  export HF_MODULES_CACHE="$HF_HOME_DIR/modules-decode"
  export TPU_RAIDEN_JOB_NAME=k3-mk-0-decode
  export TPU_RAIDEN_ENGINE_ID=k3-mk-0-decode-engine
  export TPU_K3_MOE_REDUCE_DTYPE=float32
  export TPU_RAIDEN_CONTROLLER_ADDRESS="$HEAD_IP:28340"
  export TPU_RAIDEN_RESHARD_PORT_BASE=28340
  export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28440
  export TPU_KV_TRANSFER_PORT=30340
  export TPU_SIDE_CHANNEL_PORT=30740
  export VLLM_XLA_CACHE_PATH="$CACHE_ROOT/decode"
  export VLLM_CACHE_ROOT="$CACHE_ROOT/decode/vllm-runtime"
  export VLLM_CONFIG_ROOT="$CACHE_ROOT/decode/vllm-config"
  export TORCH_TPU_TIER2_COMPILATION_CACHE="${_cache_user}_k3_decode"
  export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="${_cache_user}_k3_decode"
  export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$CACHE_ROOT/decode/torch_tpu_tier3"
  export TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT=0
fi

# ---- arguments shared by both roles (all points) -------------------------------
COMMON_ARGS=(
  --served-model-name moonshotai/Kimi-K3
  --tensor-parallel-size 32 --enable-expert-parallel
  --nnodes 4 --node-rank "$NODE_RANK" --master-addr "$HEAD_IP"
  --distributed-timeout-seconds 1800
  --max-model-len 1048576
  --kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512
  --trust-remote-code --language-model-only
  --mamba-cache-mode align --enable-prefix-caching
  --enable-prompt-tokens-details
)

# ---- per role and point ---------------------------------------------------------
# The host-memory KV store the DSpark points' roles add beside Raiden. Its size
# is split across the 32 ranks, so a quarter of it lands on each host, all of it
# allocated when the ranks register. The doc's 1.5 TiB (1649267441664) put
# ~384 GiB on each GKE tpu7x host; on top of the weight loads' page cache and
# the gcsfuse sidecar's ~63 GiB, every host's working set reached 790-850 GiB of
# 960 and the kubelet evicted the prefill head (vllm-torchtpu-dev build 562).
# 384 GiB in all is ~96 GiB per host, which peaked at 500-590 GiB (build 566).
K3_MK_OFFLOAD_BYTES="${K3_MK_OFFLOAD_BYTES:-412316860416}"
_offload_connector() {  # _offload_connector <raiden_controller_port>
  printf '{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":%s,"raiden_controller_port":%s,"raiden_job_name":"k3-mk-offload-0","store_server_ip":"%s"}}' "$K3_MK_OFFLOAD_BYTES" "$1" "$HEAD_IP"
}
_multi_connector() {  # _multi_connector <kv_role> <raiden connector json> <offload connector json>
  printf '{"kv_connector":"TPUMultiConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector","kv_role":"%s","kv_connector_extra_config":{"connectors":[%s,%s]}}' "$1" "$2" "$3"
}
_raiden='"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector"'

if [ "$K3_MK_ROLE" = prefill ]; then
  if [ "$DECODE_MODE" = dp2_dcp16 ]; then
    # Point 5: no draft, and the transfer reshards for decode's DCP16.
    LOADER_CFG=$(printf '{"presharded":"%s"}' "$PRESHARDED_DIR")
    KV_TRANSFER_CFG=$(_multi_connector kv_producer \
      "{${_raiden},\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"destination_decode_context_parallel_size\":16}}" \
      "$(_offload_connector 27800)")
    KV_CACHE_MEMORY_BYTES=23622320128
  else
    # Points 1-4, 6, 7: the same prefill.
    LOADER_CFG=$(printf '{"presharded":"%s","draft":"%s"}' "$PRESHARDED_DIR" "$DSPARK_DIR")
    KV_TRANSFER_CFG=$(_multi_connector kv_producer \
      "{${_raiden},\"kv_role\":\"kv_producer\"}" "$(_offload_connector 27800)")
    KV_CACHE_MEMORY_BYTES=19327352832
  fi
  ROLE_ARGS=(
    --decode-context-parallel-size 32
    --load-format megakernel --model-loader-extra-config "$LOADER_CFG"
    --max-num-seqs 4 --max-num-batched-tokens 8192
    --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"
    --prefix-cache-retention-interval 16384
    --kv-transfer-config "$KV_TRANSFER_CFG"
  )
else
  case "$DECODE_MODE" in
    dspark)
      LOADER_CFG=$(printf '{"presharded":"%s","synthetic_acceptance_lengths":{"3":2.87,"7":3.78}}' "$PRESHARDED_DIR")
      SPEC_CFG=$(printf '{"method":"dspark","model":"%s","num_speculative_tokens":%s,"rejection_sample_method":"synthetic","synthetic_acceptance_length":%s}' \
        "$DSPARK_DIR" "$NUM_DRAFTS" "$ACCEPTANCE_LENGTH")
      ;;
    dspark_mixed)
      LOADER_CFG=$(printf '{"presharded":"%s","synthetic_acceptance_lengths":{"3":2.87,"7":3.78}}' "$PRESHARDED_DIR")
      SPEC_CFG=$(printf '{"method":"dspark","model":"%s","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":%s,"rejection_sample_method":"synthetic","synthetic_acceptance_length":3.78}' \
        "$DSPARK_DIR" "$DRAFTS_PER_BATCH_SIZE")
      ;;
  esac
  if [ "$DECODE_MODE" = dp2_dcp16 ]; then
    LOADER_CFG=$(printf '{"presharded":"%s","attention_dp":2}' "$PRESHARDED_DIR")
    KV_TRANSFER_CFG="{${_raiden},\"kv_role\":\"kv_consumer\"}"
    ROLE_ARGS=(
      --decode-context-parallel-size 16
      --load-format megakernel --model-loader-extra-config "$LOADER_CFG"
      --max-num-seqs "$DECODE_MAX_NUM_SEQS" --max-num-batched-tokens 512
      --kv-cache-memory-bytes 30064771072
      --kv-transfer-config "$KV_TRANSFER_CFG"
    )
  else
    KV_TRANSFER_CFG=$(_multi_connector kv_consumer \
      "{${_raiden},\"kv_role\":\"kv_consumer\"}" "$(_offload_connector 28800)")
    ROLE_ARGS=(
      --decode-context-parallel-size 32
      --load-format megakernel --model-loader-extra-config "$LOADER_CFG"
      --max-num-seqs "$DECODE_MAX_NUM_SEQS" --max-num-batched-tokens 512
      --kv-cache-memory-bytes 19327352832
      --speculative-config "$SPEC_CFG"
      --kv-transfer-config "$KV_TRANSFER_CFG"
    )
  fi
fi
unset _v _cache_user _raiden
