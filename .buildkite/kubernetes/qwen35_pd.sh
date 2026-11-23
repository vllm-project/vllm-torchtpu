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

# One engine pod of a Qwen3.5-397B prefill/decode pair served by the C3 recipe
# (scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-pd.sh). Every engine is
# one tpu7x host of four chips; no host joins another headless.
#
#   qwen35_pd.sh prefill|decode
#
# Each pod sources the recipe instantiation QWEN35_PD_CONFIG for its role and
# serves its SERVE_ARGS, keeping its own log. Each pod also serves a small
# status file over HTTP (STATUS_PORT): what its log says about the kernel page
# its workers settled on and any Raiden layout-fingerprint mismatch. The
# prefill head then fronts the pair with the recipe's proxy (vllm-router by
# default), runs scripts/vllm/integration/run_qwen35_397b_pd_correctness.sh
# through it, uploads its results and exits; the JobSet's success policy tears
# the rest down.
#
# Environment, from the manifest: POD_IP; PREFILL_HOSTS and DECODE_HOSTS, the
# roles' pods in replica order (space-separated names); QWEN35_PD_CONFIG;
# PD_REPLICA_INDEX, this pod's replica within its role (0 when unset); and
# ARTIFACTS_DIR.
#
# QWEN35_PD_DRY_RUN=1 prints what this pod would run - the serve arguments, the
# environment the recipe sets, the proxy and the correctness command - and
# exits; tests/models/qwen3_5/test_qwen3_5_recipe.py checks it against the
# validation entries.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${QWEN35_PD_CONFIG:-qwen3.5-397b-fp8-pd-1p1d}"
PROXY_PORT=8100
STATUS_PORT=8300

role="${1:-}"
case "$role" in
  prefill|decode) ;;
  *) echo "usage: $0 prefill|decode" >&2; exit 2 ;;
esac
: "${POD_IP:?}" "${PREFILL_HOSTS:?}" "${DECODE_HOSTS:?}"
INDEX="${PD_REPLICA_INDEX:-0}"
read -r -a prefill_hosts <<<"$PREFILL_HOSTS"
read -r -a decode_hosts <<<"$DECODE_HOSTS"

REPO_DIR="${REPO_DIR:-/workspace/vllm-torchtpu}"
cd "$REPO_DIR" || { echo "$0: no such directory: $REPO_DIR" >&2; exit 2; }
# Core dumps are gigabytes each and fill the pod's ephemeral storage.
ulimit -c 0
if [[ -n "${ARTIFACTS_DIR:-}" ]]; then
  mkdir -p "$ARTIFACTS_DIR"
fi
# shellcheck source=.buildkite/kubernetes/multihost_lib.sh
# shellcheck disable=SC1091
source "${HERE}/multihost_lib.sh"

# What the environment was before the recipe, so a dry run can print only what
# the recipe and this script set.
env_before="$(python3 -c 'import json, os; print(json.dumps(dict(os.environ)))')"

export PD_ROLE="$role"
SERVE_ARGS=()
# shellcheck source=/dev/null
source "scripts/vllm/benchmarking/configs/${CONFIG}.sh" || exit 2
if [[ "${#prefill_hosts[@]}" -ne "$PD_PREFILL_REPLICAS" || "${#decode_hosts[@]}" -ne "$PD_DECODE_REPLICAS" ]]; then
  echo "ERROR: ${CONFIG} serves ${PD_PREFILL_REPLICAS}P${PD_DECODE_REPLICAS}D, but the manifest names" \
    "${#prefill_hosts[@]} prefill and ${#decode_hosts[@]} decode pods" >&2
  exit 2
fi
read -r -a decode_ports <<<"$PD_DECODE_API_PORTS"

# Per pod: the address the engine and its Raiden store advertise, and a Raiden
# identity per replica (replica 0 keeps the role's).
export VLLM_HOST_IP="$POD_IP"
export TPU_RAIDEN_ADVERTISE_HOST="$POD_IP"
if [[ "$INDEX" != "0" ]]; then
  export TPU_RAIDEN_ENGINE_ID="${role}-${INDEX}-engine"
fi
export PYTHONUNBUFFERED=1

# The proxy in front of the pair, on the prefill head: every engine's API
# server is one endpoint (DP4 x TP2 decode runs four per host).
prefill_ep=("127.0.0.1:${PD_SERVE_PORT}")
for host in "${prefill_hosts[@]:1}"; do
  prefill_ep+=("${host}:${PD_SERVE_PORT}")
done
decode_ep=()
for host in "${decode_hosts[@]}"; do
  for port in "${decode_ports[@]}"; do
    decode_ep+=("${host}:${port}")
  done
done
prefill_urls=("${prefill_ep[@]/#/http://}")
decode_urls=("${decode_ep[@]/#/http://}")
if [[ "$PD_PROXY" == router ]]; then
  # As the Kimi-K3 AgentX pair runs it: the request timeout covers a full
  # decode, and retries are off so a failed hop fails the request instead of
  # prefilling it twice.
  proxy_cmd=(vllm-router --host 0.0.0.0 --port "$PROXY_PORT" --vllm-pd-disaggregation)
  for url in "${prefill_urls[@]}"; do proxy_cmd+=(--prefill "$url"); done
  for url in "${decode_urls[@]}"; do proxy_cmd+=(--decode "$url"); done
  proxy_cmd+=(--policy consistent_hash --request-timeout-secs 5000
    --worker-startup-timeout-secs 300 --health-check-interval-secs 30 --disable-retries
    --prometheus-host 0.0.0.0 --prometheus-port "$((PROXY_PORT + 1))")
  proxy_health=health
else
  proxy_cmd=(python3 examples/disagg/toy_proxy_server.py --host 0.0.0.0 --port "$PROXY_PORT"
    --prefiller-hosts "${prefill_ep[@]%:*}" --prefiller-ports "${prefill_ep[@]##*:}"
    --decoder-hosts "${decode_ep[@]%:*}" --decoder-ports "${decode_ep[@]##*:}")
  proxy_health=healthcheck
fi
status_urls() {
  local host
  for host in "$@"; do printf 'http://%s:%s/leg_status.json ' "$host" "$STATUS_PORT"; done
}
run_dir="/tmp/qwen35_pd/${role}"
correctness_env=(
  QWEN35_PD_CONFIG="$CONFIG" PROXY_HOST=127.0.0.1 PROXY_PORT="$PROXY_PORT"
  PREFILL_URL="${prefill_urls[0]}" DECODE_URLS="${decode_urls[*]}"
  PREFILL_LEG_STATUS="${run_dir}/status/leg_status.json $(status_urls "${prefill_hosts[@]:1}")"
  DECODE_LEG_STATUS="$(status_urls "${decode_hosts[@]}")"
  LOG_DIR="${ARTIFACTS_DIR:-$run_dir}/qwen35_397b_pd_correctness"
)

if [[ "${QWEN35_PD_DRY_RUN:-0}" == "1" ]]; then
  printf '[cmd] vllm serve'; printf ' %q' "${SERVE_ARGS[@]}"; echo
  python3 - "$env_before" "${SERVE_ARGS[@]}" <<'PY'
import json, os, sys
before = json.loads(sys.argv[1])
# The shell's own bookkeeping, which bash versions set differently.
shell = {"SHLVL", "_", "PWD", "OLDPWD"}
env = {k: v for k, v in os.environ.items() if before.get(k) != v and k not in shell}
print("[serve-args] " + json.dumps(sys.argv[2:]))
print("[env] " + json.dumps(env, sort_keys=True))
PY
  if [[ "$role" == prefill && "$INDEX" == "0" ]]; then
    printf '[proxy]'; printf ' %q' "${proxy_cmd[@]}"; echo
    printf '[correctness]'; printf ' %q' "${correctness_env[@]}"
    echo " bash scripts/vllm/integration/run_qwen35_397b_pd_correctness.sh"
  fi
  exit 0
fi

mkdir -p "$run_dir/status"
serve_log="$run_dir/serve.log"
echo "--- Starting the ${role} engine (replica ${INDEX}) from ${CONFIG}"
vllm serve "${SERVE_ARGS[@]}" > >(tee "$serve_log") 2>&1 &
serve_pid=$!

# This pod's leg status, refreshed while the engine runs, served for the
# prefill head (python's http.server: the image has nothing else to serve
# files with).
write_status() {
  python3 scripts/vllm/integration/qwen35_pd_checks.py leg-status \
    --log "$serve_log" --out "$run_dir/status/leg_status.json" >/dev/null
}
write_status
(while kill -0 "$serve_pid" 2>/dev/null; do sleep 10; write_status; done) &
python3 -m http.server "$STATUS_PORT" --bind 0.0.0.0 --directory "$run_dir/status" \
  >/dev/null 2>&1 &

if [[ "$role" == decode || "$INDEX" != "0" ]]; then
  wait "$serve_pid"
  exit $?
fi

proxy_pid=""
# shellcheck disable=SC2317  # invoked via the EXIT trap
cleanup() {
  if [[ -n "$proxy_pid" ]]; then
    kill "$proxy_pid" 2>/dev/null || true
  fi
  kill "$serve_pid" 2>/dev/null || true
  wait "$serve_pid" 2>/dev/null || true
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

# Boot: every engine of both legs answers /health.
wait_ready "${prefill_urls[0]}/health" "prefill engine" "$serve_pid" || exit 1
for url in "${prefill_urls[@]:1}" "${decode_urls[@]}"; do
  wait_ready "${url}/health" "engine ${url}" || exit 1
done
for url in "${prefill_urls[@]}" "${decode_urls[@]}"; do
  echo "cache config at ${url}:"
  curl -sf --max-time 30 "${url}/metrics" | grep -m1 '^vllm:cache_config_info' || true
done

echo "--- Starting the ${PD_PROXY} proxy on port ${PROXY_PORT}"
"${proxy_cmd[@]}" >"$run_dir/proxy.log" 2>&1 &
proxy_pid=$!
if ! timeout 300 bash -c "until curl -sf -o /dev/null http://127.0.0.1:${PROXY_PORT}/${proxy_health}; do sleep 2; done"; then
  echo "ERROR: the ${PD_PROXY} proxy is not ready"
  tail -50 "$run_dir/proxy.log" || true
  exit 1
fi

write_status
rc=0
env "${correctness_env[@]}" bash scripts/vllm/integration/run_qwen35_397b_pd_correctness.sh || rc=$?

if [[ -n "${ARTIFACTS_DIR:-}" ]]; then
  cp "$run_dir/proxy.log" "$run_dir/status/leg_status.json" "$ARTIFACTS_DIR/" 2>/dev/null || true
fi
# A failed upload decides the step only when the work itself passed. Outside
# Buildkite (no agent: a deployment running this launcher itself) the results
# stay in ARTIFACTS_DIR for the caller to collect.
if command -v buildkite-agent >/dev/null; then
  if ! publish_results && [[ "$rc" -eq 0 ]]; then
    rc=1
  fi
fi
exit "$rc"
