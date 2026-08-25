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

# Values: sharding, max sequences, global max tokens, prefill interval.
server_config_for() {
  case "$1,$2,$3" in
    8192,1024,4) printf "%s" "TP8_EP 4 16384 1" ;;
    8192,1024,8) printf "%s" "TP8_EP 8 16384 1" ;;
    8192,1024,16) printf "%s" "TP8_EP 16 16384 1" ;;
    8192,1024,32) printf "%s" "TP8_EP 32 16384 1" ;;
    8192,1024,64) printf "%s" "TP8_EP 64 16384 1" ;;
    8192,1024,128) printf "%s" "DP8_EP 16 2048 1" ;;
    8192,1024,256) printf "%s" "DP8_EP 32 2560 1" ;;
    *) return 1 ;;
  esac
}

if [ "${1:-}" = "--point-config" ]; then
  server_config_for "$2" "$3" "$4"
  exit
fi

# Sweep 8k1k points and reuse servers with identical configurations.
# Usage: bash sweep.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-5400}"   # 90 min (covers a cold compile)

# Select workloads and concurrencies.
read -r -a WORKLOADS <<< "${WORKLOADS:-8192:1024}"
read -r -a CONCS <<< "${CONCS:-4 8 16 32 64 128 256}"
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

# Build the environment used to resolve and start a point.
build_server_env() {
  local sharding="$1"
  local max_seqs="$2"
  local global_max_tokens="$3"
  local prefill_interval="$4"
  SERVER_ENV=(
    SHARDING="$sharding"
    CONC="$CONC"
    MAX_NUM_SEQS="$max_seqs"
    GLOBAL_MAX_NUM_BATCHED_TOKENS="$global_max_tokens"
    PREFILL_SCHEDULE_INTERVAL="$prefill_interval"
  )
}

start_server() {
  local sharding="$1"
  build_server_env "$@"

  SERVER_LOG="/tmp/qwen3.5_sweep_server_isl${ISL}_osl${OSL}_${sharding}_conc${CONC}_$(date +%m%d-%H%M%S)_$$.log"
  echo "--- starting server SHARDING=$sharding ISL=$ISL OSL=$OSL CONC=$CONC (log: $SERVER_LOG) ---"
  env "${SERVER_ENV[@]}" bash "${SCRIPT_DIR}/server.sh" > >(tee "$SERVER_LOG") 2>&1 &
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

current_server_key=""
for wl in "${WORKLOADS[@]}"; do
  export ISL="${wl%%:*}" OSL="${wl#*:}"
  for CONC in "${CONCS[@]}"; do
    export CONC
    point="$ISL,$OSL,$CONC"
    if [ -n "${SHARDING_OVERRIDE:-}" ]; then
      # Restart each point when using an explicit sharding override.
      sharding="$SHARDING_OVERRIDE"
      max_seqs=""
      global_max_tokens=""
      prefill_interval=""
      server_key="override:$sharding:$point"
    else
      if ! point_config="$(server_config_for "$ISL" "$OSL" "$CONC")"; then
        echo "ERROR: no server config for $point" >&2
        exit 1
      fi
      read -r sharding max_seqs global_max_tokens prefill_interval <<<"$point_config"
      # Reuse only identical point configurations.
      server_key="$ISL:$OSL:$point_config"
    fi
    [ -n "$sharding" ] || { echo "ERROR: no sharding for ISL=$ISL OSL=$OSL CONC=$CONC" >&2; exit 1; }
    if [ "$server_key" != "$current_server_key" ]; then
      stop_server
      started=0
      for attempt in 1 2; do
        if start_server "$sharding" "$max_seqs" "$global_max_tokens" "$prefill_interval"; then
          started=1
          break
        fi
        echo "--- start attempt $attempt failed; cleaning up and retrying ---" >&2
        TPU_RELEASE_WAIT=60 stop_server
      done
      if [ "$started" -ne 1 ]; then
        echo "ERROR: server would not start for ISL=$ISL OSL=$OSL CONC=$CONC; skipping point" >&2
        current_server_key=""
        continue
      fi
      current_server_key="$server_key"
    else
      echo "--- reusing server for ISL=$ISL OSL=$OSL CONC=$CONC ---"
    fi
    echo "########## ISL=$ISL OSL=$OSL CONC=$CONC SHARDING=$sharding ##########"
    BENCH_LOG="/tmp/qwen3.5_sweep_bench_isl${ISL}_osl${OSL}_conc${CONC}_$(date +%m%d-%H%M%S)_$$.log"
    echo "--- benchmark (log: $BENCH_LOG) ---"
    if ! bash "${SCRIPT_DIR}/bench.sh" 2>&1 | tee "$BENCH_LOG"; then
      echo "ERROR: benchmark failed for ISL=$ISL OSL=$OSL CONC=$CONC" >&2
    fi
  done
done
echo "########## sweep complete ##########"
