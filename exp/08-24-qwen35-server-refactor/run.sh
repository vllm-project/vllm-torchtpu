#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
script_dir="$repo_root/scripts/vllm/benchmark/inferencex/qwen3.5"
server_config_for() {
  bash "$script_dir/sweep.sh" --point-config "$@"
}

vllm() {
  shift
  printf "%s" "$*"
  printf " USE_MOE_FUSED_EP_KERNEL=%s" "${USE_MOE_FUSED_EP_KERNEL:-}"
  printf " TPU_MOE_FUSED_EP_STEP_MIN_TOKENS=%s" "${TPU_MOE_FUSED_EP_STEP_MIN_TOKENS:-}"
  printf " TPU_MOE_SKIP_PADDED_TOKENS=%s" "${TPU_MOE_SKIP_PADDED_TOKENS:-}"
  printf " TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER=%s\n" "${TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER:-}"
}
export -f vllm

for conc in 4 8 16 32 64 128 256; do
  config="$(server_config_for 8192 1024 "$conc")"
  read -r sharding max_seqs global_max_tokens prefill_interval <<<"$config"
  printf "POINT ISL=8192 OSL=1024 CONC=%s SHARDING=%s\n" "$conc" "$sharding"
  env ISL=8192 OSL=1024 CONC="$conc" SHARDING="$sharding" \
    MAX_NUM_SEQS="$max_seqs" \
    GLOBAL_MAX_NUM_BATCHED_TOKENS="$global_max_tokens" \
    PREFILL_SCHEDULE_INTERVAL="$prefill_interval" \
    bash "$script_dir/server.sh" 2>/dev/null
done
