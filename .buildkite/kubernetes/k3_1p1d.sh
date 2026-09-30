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

# One pod of the Kimi-K3 1P1D test. Prefill and decode each get a 2x2x4 slice,
# four hosts, and run one Ray engine across it.
#
#   k3_1p1d.sh prefill|decode          every pod of the role's slice
#   k3_1p1d.sh --head prefill|decode   what multihost_entry.sh runs on index 0
#
# Both engines serve the kimi-k3-tp32-ep-1p1d config, which holds the serve
# flags and the Raiden environment of each. The prefill head then fronts the
# pair with the toy proxy, checks a few greedy completions through it, runs the
# correctness stages and the MMLU-Pro eval through it, and exits; the JobSet's
# success policy tears decode down.
#
# Environment, from manifests/workloads/kimi-k3-1p1d.yaml: POD_IP,
# DECODE_HOST (decode's index 0), and what multihost_entry.sh reads.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/workspace/vllm-torchtpu}"
CONFIG=kimi-k3-tp32-ep-1p1d
PORT=8000
PROXY_PORT=8100

head=0
if [[ "${1:-}" == "--head" ]]; then
  head=1
  shift
fi
role="${1:-}"
case "$role" in
  prefill|decode) ;;
  *) echo "usage: $0 [--head] prefill|decode" >&2; exit 2 ;;
esac

# head_ip: index 0's address, waiting for its DNS record as multihost_entry.sh
# waits for its Ray port.
head_ip() {
  local deadline=$((SECONDS + 1800)) ip
  until ip=$(getent ahostsv4 "$HEAD_HOST" | awk 'NR == 1 { print $1 }') \
      && [[ -n "$ip" ]]; do
    if [[ "$SECONDS" -ge "$deadline" ]]; then
      echo "ERROR: ${HEAD_HOST} did not resolve in 30 minutes" >&2
      return 1
    fi
    sleep 10
  done
  echo "$ip"
}

# warm_remote_code: import the model's remote code once on this host before Ray
# starts the engine's workers here; see warm_remote_code.py for why.
warm_remote_code() {
  local uri
  # The subshell keeps the config's variables, KV_CACHE_DTYPE among them, out
  # of this script. Followed, shellcheck reports them as lost (SC2031).
  # shellcheck source=/dev/null
  uri=$(source "$REPO_DIR/scripts/vllm/benchmarking/configs/kimi-k3-tp32-ep.sh" \
    >/dev/null && echo "$MODEL_URI") || return 1
  timeout 600 python3 "$HERE/warm_remote_code.py" "$uri"
}

if [[ "$head" == "0" ]]; then
  # Set before Ray starts on every host: the engine's workers are Ray actors on
  # all four, and they inherit the raylet's environment, not the head's.
  export VLLM_HOST_IP="${POD_IP}"
  # Best effort: without it the workers pull and import as they always have.
  if [[ "${JOB_COMPLETION_INDEX:-0}" != "0" ]]; then
    warm_remote_code || echo "WARNING: could not warm the model's remote code"
  fi
  # The engine's Raiden store runs in its rank-0 worker, on index 0, and every
  # worker dials it at this host - so it is index 0's address on all four.
  if [[ "${JOB_COMPLETION_INDEX:-0}" == "0" ]]; then
    TPU_RAIDEN_ADVERTISE_HOST="${POD_IP}"
  else
    TPU_RAIDEN_ADVERTISE_HOST="$(head_ip)" || exit 1
  fi
  export TPU_RAIDEN_ADVERTISE_HOST
  # The rest of the Raiden environment is the config's.
  export K3_1P1D_ROLE="$role"
  # shellcheck source=/dev/null
  source "$HERE/../../scripts/vllm/benchmarking/configs/${CONFIG}.sh"
  exec bash "$HERE/multihost_entry.sh" bash "$HERE/k3_1p1d.sh" --head "$role"
fi

cd "$REPO_DIR"
run_dir="/tmp/k3_1p1d/${role}"
mkdir -p "$run_dir"

serve() {
  bash scripts/vllm/benchmarking/run_benchmarks.sh \
    --config "$CONFIG" --serve-only --port "$PORT" --results-dir "$run_dir"
}

# The server writes to a file; follow it here so the pod's log carries it.
touch "$run_dir/server.log"
tail -n +1 -F "$run_dir/server.log" 2>/dev/null | sed -u "s/^/[${role} server] /" &
tail_pid=$!

if [[ "$role" == "decode" ]]; then
  serve
  exit $?
fi

serve &
serve_pid=$!
proxy_pid=""
# shellcheck disable=SC2317  # invoked via the EXIT trap
cleanup() {
  if [[ -n "$proxy_pid" ]]; then
    kill "$proxy_pid" 2>/dev/null || true
  fi
  kill "$serve_pid" 2>/dev/null || true
  wait "$serve_pid" 2>/dev/null || true
  kill "$tail_pid" 2>/dev/null || true
}
trap cleanup EXIT

# wait_ready <url> <what> [pid]: poll until the URL answers 200, failing early
# when the named process has died.
wait_ready() {
  local url="$1" what="$2" pid="${3:-}"
  local deadline=$((SECONDS + 250 * 60))
  echo "--- Waiting for ${what} at ${url}"
  until [[ "$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 "$url" || true)" == "200" ]]; do
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: ${what} exited before it was ready"
      return 1
    fi
    if [[ "$SECONDS" -ge "$deadline" ]]; then
      echo "ERROR: ${what} not ready after 250 minutes"
      return 1
    fi
    sleep 30
  done
  echo "${what} is ready"
}

wait_ready "http://127.0.0.1:${PORT}/health" "prefill engine" "$serve_pid"
wait_ready "http://${DECODE_HOST}:${PORT}/health" "decode engine"

echo "--- Starting the proxy on port ${PROXY_PORT}"
python3 examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 --port "$PROXY_PORT" \
  --prefiller-hosts 127.0.0.1 --prefiller-ports "$PORT" \
  --decoder-hosts "$DECODE_HOST" --decoder-ports "$PORT" \
  >"$run_dir/proxy.log" 2>&1 &
proxy_pid=$!
if ! timeout 120 bash -c "until curl -sf -o /dev/null http://127.0.0.1:${PROXY_PORT}/healthcheck; do sleep 2; done"; then
  echo "ERROR: proxy not ready"
  tail -50 "$run_dir/proxy.log" || true
  exit 1
fi

echo "--- Completions through the proxy"
python3 "$HERE/check_completions.py" \
  "http://127.0.0.1:${PROXY_PORT}/v1/completions" moonshotai/Kimi-K3

# served <key>: what the prefill engine was started with, from the config.json
# run_benchmarks.sh wrote beside its log.
served() {
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
    "$run_dir/config.json" "$1"
}

# The eval runs even if a stage fails: the pair holds 32 chips, and there is no
# sense spending them to learn one fact. Not if the stages could not start,
# though, which is this script's fault or the config's and not the pair's.
stages_rc=0

echo "--- Correctness stages through the proxy"
PROXY_PORT="$PROXY_PORT" \
PREFILL_URL="http://127.0.0.1:${PORT}" \
DECODE_URL="http://${DECODE_HOST}:${PORT}" \
PREFILL_CHUNK_TOKENS="$(served max_num_batched_tokens)" \
LOG_DIR="${ARTIFACTS_DIR:-$run_dir}/k3_1p1d_correctness" \
  bash scripts/vllm/integration/run_k3_1p1d_disagg_correctness.sh || stages_rc=$?
if [[ "$stages_rc" -gt 1 ]]; then
  echo "ERROR: the correctness stages did not start (exit ${stages_rc})"
  exit 1
fi
rc="$stages_rc"

# No benchmark: what the pair serves is set by the stages above, one KV block
# to a prefill chunk, and its latency describes nothing a reader tracks. The
# eval is gated on the aggregated config's baseline, since splitting prefill
# from decode must not change the answers, and kept out of the shared results
# table, where its rows would read as that config's.
echo "--- lm_eval through the proxy"
bash scripts/vllm/benchmarking/run_eval_flow.sh \
  --config "$CONFIG" --baseline-config kimi-k3-tp32-ep --run-lm-eval \
  --host 127.0.0.1 --port "$PROXY_PORT" \
  --skip-benchmark --skip-db-upload || rc=1

exit "$rc"
