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

# Sweep: restart the server per concurrency.
# Usage: bash sweep.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-5400}"   # 90 min (covers a cold compile)

# "ISL:OSL" pairs to sweep, and the concurrencies to run for each.
# Both are overridable via env so per-point jobs can each own a slice.
read -r -a WORKLOADS <<< "${WORKLOADS:-8192:1024}"
read -r -a CONCS <<< "${CONCS:-4 8 16 32 64 128 256 512 1024}"
# Per-point sharding, followed by DP8 slots, global token budget and
# release group.
# The points live in points.sh, shared with server.sh.
source "${SCRIPT_DIR}/points.sh"

VLLM_PROCS='vllm serve|VLLM::'

stop_server() {
  pkill -TERM -f "$VLLM_PROCS" 2>/dev/null || true
  for _ in $(seq 1 15); do
    pgrep -f "$VLLM_PROCS" >/dev/null 2>&1 || break
    sleep 2
  done
  pkill -KILL -f "$VLLM_PROCS" 2>/dev/null || true
  # Do not proceed while anything vllm-shaped still exists.
  for _ in $(seq 1 30); do
    pgrep -f "$VLLM_PROCS" >/dev/null 2>&1 || break
    sleep 2
  done
  if pgrep -f "$VLLM_PROCS" >/dev/null 2>&1; then
    echo "WARN: vllm processes survived SIGKILL:" >&2
    pgrep -af "$VLLM_PROCS" >&2 || true
  fi
  for _ in $(seq 1 30); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://0.0.0.0:${PORT}/health" 2>/dev/null)" = "200" ] || break
    sleep 2
  done
  sleep "${TPU_RELEASE_WAIT:-20}"
}

start_server() {
  # server.sh applies the point's settings from points.sh, except any the
  # caller already set in the environment. An override passes its own.
  local point_env=(ISL="$ISL" OSL="$OSL" CONC="$CONC")
  if [ -n "${SHARDING_OVERRIDE:-}" ] || [ -n "${POINT_OVERRIDE:-}" ]; then
    local sharding seqs budget group
    read -r sharding seqs budget group <<< "$1"
    point_env+=(SHARDING="$sharding")
    [ -n "${seqs:-}" ] && point_env+=(MAX_NUM_SEQS="$seqs" GLOBAL_MAX_NUM_BATCHED_TOKENS="$budget" VLLM_TPU_DP_RELEASE_THRESHOLD="$group")
  fi
  SERVER_LOG="/tmp/qwen3.5_sweep_server_isl${ISL}_osl${OSL}_conc${CONC}_$(date +%m%d-%H%M).log"
  echo "--- starting server ${point_env[*]} (log: $SERVER_LOG) ---"
  env "${point_env[@]}" bash "${SCRIPT_DIR}/server.sh" > >(tee -a "$SERVER_LOG") 2>&1 &
  SERVER_PID=$!
  local waited=0
  until grep -q "Application startup complete" "$SERVER_LOG" 2>/dev/null; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: server exited during startup (see $SERVER_LOG)" >&2; return 1; }
    sleep 5; waited=$((waited + 5))
    [ "$waited" -ge "$READY_TIMEOUT" ] && { echo "ERROR: server not ready after ${READY_TIMEOUT}s" >&2; return 1; }
  done
  echo "--- server ready ---"
}

trap stop_server EXIT

for wl in "${WORKLOADS[@]}"; do
  export ISL="${wl%%:*}" OSL="${wl#*:}"
  for CONC in "${CONCS[@]}"; do
    export CONC
    # SHARDING_OVERRIDE keeps its meaning: one sharding for every point.
    point="${POINT_OVERRIDE:-${SHARDING_OVERRIDE:-${POINT_TABLE[$ISL,$OSL,$CONC]:-}}}"
    [ -n "$point" ] || { echo "ERROR: no settings in POINT_TABLE for ISL=$ISL OSL=$OSL CONC=$CONC" >&2; exit 1; }
    stop_server
    started=0
    for attempt in 1 2; do
      if start_server "$point"; then started=1; break; fi
      echo "--- start attempt $attempt failed; cleaning up and retrying ---" >&2
      TPU_RELEASE_WAIT=60 stop_server
    done
    if [ "$started" -ne 1 ]; then
      echo "ERROR: server would not start for ISL=$ISL OSL=$OSL CONC=$CONC; skipping point" >&2
      continue
    fi
    echo "########## ISL=$ISL OSL=$OSL CONC=$CONC POINT=$point ##########"
    BENCH_LOG="/tmp/qwen3.5_sweep_bench_isl${ISL}_osl${OSL}_conc${CONC}_$(date +%m%d-%H%M).log"
    echo "--- benchmark (log: $BENCH_LOG) ---"
    bash "${SCRIPT_DIR}/bench.sh" 2>&1 | tee -a "$BENCH_LOG"
  done
done
echo "########## sweep complete ##########"
