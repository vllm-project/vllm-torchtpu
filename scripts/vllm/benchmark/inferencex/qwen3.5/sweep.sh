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
read -r -a WORKLOADS <<< "${WORKLOADS:-8192:1024 1024:1024}"
read -r -a CONCS <<< "${CONCS:-4 8 16 32 64 128 256}"
# Sharding per swept point, keyed by "ISL,OSL,CONC" (every point listed).
# Low concurrency runs TP8_EP, higher concurrency runs DP8_EP.
declare -A SHARDING_TABLE=(
  [8192,1024,4]=TP8_EP
  [8192,1024,8]=TP8_EP
  [8192,1024,16]=TP8_EP
  [8192,1024,32]=DP4TP2_EP
  [8192,1024,64]=DP4TP2_EP
  [8192,1024,128]=DP8_EP
  [8192,1024,256]=DP8_EP
  [1024,1024,4]=TP8_EP
  [1024,1024,8]=TP8_EP
  [1024,1024,16]=TP8_EP
  [1024,1024,32]=TP8_EP
  [1024,1024,64]=TP8_EP
  [1024,1024,128]=TP8_EP
  [1024,1024,256]=TP8_EP
)

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
  SERVER_LOG="/tmp/qwen3.5_sweep_server_isl${ISL}_osl${OSL}_conc${CONC}_$(date +%m%d-%H%M).log"
  echo "--- starting server SHARDING=$1 ISL=$ISL OSL=$OSL CONC=$CONC (log: $SERVER_LOG) ---"
  SHARDING="$1" bash "${SCRIPT_DIR}/server.sh" > >(tee -a "$SERVER_LOG") 2>&1 &
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
    sharding="${SHARDING_OVERRIDE:-${SHARDING_TABLE[$ISL,$OSL,$CONC]:-}}"
    [ -n "$sharding" ] || { echo "ERROR: no sharding in SHARDING_TABLE for ISL=$ISL OSL=$OSL CONC=$CONC" >&2; exit 1; }
    stop_server
    started=0
    for attempt in 1 2; do
      if start_server "$sharding"; then started=1; break; fi
      echo "--- start attempt $attempt failed; cleaning up and retrying ---" >&2
      TPU_RELEASE_WAIT=60 stop_server
    done
    if [ "$started" -ne 1 ]; then
      echo "ERROR: server would not start for ISL=$ISL OSL=$OSL CONC=$CONC; skipping point" >&2
      continue
    fi
    echo "########## ISL=$ISL OSL=$OSL CONC=$CONC SHARDING=$sharding ##########"
    BENCH_LOG="/tmp/qwen3.5_sweep_bench_isl${ISL}_osl${OSL}_conc${CONC}_$(date +%m%d-%H%M).log"
    echo "--- benchmark (log: $BENCH_LOG) ---"
    bash "${SCRIPT_DIR}/bench.sh" 2>&1 | tee -a "$BENCH_LOG"
  done
done
echo "########## sweep complete ##########"
