#!/bin/bash
# Verify that the sweep and standalone launcher resolve each point identically.
# Usage: check_points.sh [concurrency...]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
status=0
for c in ${@:-4 8 16 32 64 128 256 512 1024}; do
  if ! sweep_raw=$(ISL=8192 OSL=1024 CONCS=$c DRY_RUN=1 bash "$HERE/sweep.sh" 2>&1); then
    printf 'ERROR: sweep dry run failed at concurrency %s:\n%s\n' "$c" "$sweep_raw" >&2
    status=1
    continue
  fi
  if ! alone_raw=$(ISL=8192 OSL=1024 CONC=$c DRY_RUN=1 bash "$HERE/server.sh" 2>&1); then
    printf 'ERROR: standalone dry run failed at concurrency %s:\n%s\n' "$c" "$alone_raw" >&2
    status=1
    continue
  fi
  sweep=$(printf '%s\n' "$sweep_raw" | grep -E '^[A-Z_]+=|^vllm serve' | sed "s#$HERE/##g")
  alone=$(printf '%s\n' "$alone_raw" | grep -E '^[A-Z_]+=|^vllm serve' | sed "s#$HERE/##g")
  if [ -z "$sweep" ] || [ -z "$alone" ]; then
    printf 'ERROR: missing resolved settings at concurrency %s\n' "$c" >&2
    status=1
  elif [ "$sweep" != "$alone" ]; then
    echo "DIFFERS at $c users (sweep < > standalone):"
    diff <(echo "$sweep") <(echo "$alone") | grep '^[<>]' | cut -c1-200
    status=1
  else
    echo "$c users: the sweep and a standalone launch resolve alike -- $(echo "$sweep" | grep -E '^(SHARDING|MAX_NUM_SEQS|GLOBAL_MAX_NUM_BATCHED_TOKENS|DP_SCHED_BUFFER_THRESHOLD|TPU_EXPERT_PLACEMENT|TPU_MOE_LOWTOK_FROM_SCORES|TPU_SMALL_ALL_REDUCE_MAX_ROWS|GDN_STATE_DTYPE)=' | tr '\n' ' ')"
  fi
done
exit "$status"
