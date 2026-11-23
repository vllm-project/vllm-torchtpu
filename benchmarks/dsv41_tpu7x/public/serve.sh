#!/usr/bin/env bash
# Start one role of the validated 1P1D shape inside its pod (after bootstrap.sh). Usage: RUN_ID=<same on both pods> serve.sh prefill|decode
#   prefill: 4 x (TP2, EP2) 20-layer CED servers on :8400-8403, 8,192-token chunk
#   decode : DP8 / TP1 / EP8, DSpark K=5, max-num-seqs 128, max-num-batched-tokens 768, on :9400
set -euo pipefail
ROLE=${1:?prefill|decode}; : "${RUN_ID:?set the same RUN_ID on both pods}"
ADVERTISE_HOST=${ADVERTISE_HOST:-${POD_IP:?set POD_IP or ADVERTISE_HOST to the routable IP of this host}}
B=/workspace/vllm-torchtpu/benchmarks/dsv41_tpu7x
{
  echo "export ADVERTISE_HOST='$ADVERTISE_HOST' RUN_ID='$RUN_ID'"
  echo "export CED_PREFILL='1' GPU_MEMORY_UTILIZATION='0.85'"
  echo "export PREFILL_MODE='4x_tp2' PREFILL_4X_TP2_GPU_MEM='0.92' PREFILL_4X_TP2_MNBT='8192'"
  # Required on torch_tpu 0.2.0: the chunked stream-index kernel's scheduling groups do not compile there.
  echo "export VLLM_DSV41_STREAMIDX_SCHED_GROUPS='0'"
  if [ "$ROLE" = prefill ]; then
    echo "export MAX_NUM_BATCHED_TOKENS='8192' ENABLE_DSPARK='0'"
  else
    echo "export MAX_NUM_BATCHED_TOKENS='768' ENABLE_DSPARK='1' MAX_NUM_SEQS='128'"
    echo "export COMPILE_SIZES='[16, 40, 48, 64, 96, 192, 384, 512, 768]'"
    echo "export VLLM_DSV41_CSA_FUSED_MAX_TOKENS='768'"
    # Decode-side prefix caching is on so the decode-only sweep (sweep.sh) can share its warm prefix.
    echo "export ENABLE_PREFIX_CACHING='${DECODE_PREFIX_CACHING:-1}' PREFIX_CACHE_RETENTION_INTERVAL='1024'"
  fi
} > /tmp/pd_env.sh
nohup setsid bash $B/scripts/pd/in_pod_serve.sh "$ROLE" > /dev/shm/pd_serve_$ROLE.out 2>&1 < /dev/null &
echo "started $ROLE (ADVERTISE_HOST=$ADVERTISE_HOST RUN_ID=$RUN_ID); log: /dev/shm/pd_$ROLE.log"
