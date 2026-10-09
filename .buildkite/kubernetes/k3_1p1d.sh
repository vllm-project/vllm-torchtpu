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
# four hosts, and serve TP8 x DP4 x EP32 over vLLM's own multi-node DP, one DP
# replica per host.
#
#   k3_1p1d.sh prefill|decode          every pod of the role's slice
#
# The launch is mp_multihost_entry.sh's, which serves a config under
# run_eval_flow.sh and so cannot front a pair: index 0 serves the API at the
# role's --data-parallel-address, the other hosts join it --headless at their
# start rank. Both engines serve the kimi-k3-tp8-dp4-ep-1p1d config, which
# holds the serve flags and the Raiden environment of each. The prefill head
# then fronts the pair with the toy proxy, checks a few greedy completions
# through it, runs the correctness stages and the MMLU-Pro eval through it, and
# exits; the JobSet's success policy tears decode down.
#
# Environment, from manifests/workloads/kimi-k3-1p1d-4dp8tp.yaml: POD_IP,
# ROLE_HOST_PREFIX and ROLE_HOST_DOMAIN (host i of this role is
# <prefix>i.<domain>), DECODE_HOST (decode's index 0), NUM_HOSTS.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The pair's serve flags and Raiden environment. The AgentX lane serves the
# same pair with a longer context and its own ladder, so it passes its own.
CONFIG="${K3_PD_CONFIG:-kimi-k3-tp8-dp4-ep-1p1d}"
PORT=8000
PROXY_PORT=8100
DP_RPC_PORT=29500
# DP replicas per host.
DP_LOCAL=1

role="${1:-}"
case "$role" in
  prefill|decode) ;;
  *) echo "usage: $0 prefill|decode" >&2; exit 2 ;;
esac
: "${POD_IP:?}" "${ROLE_HOST_PREFIX:?}" "${ROLE_HOST_DOMAIN:?}" "${DECODE_HOST:?}"
: "${NUM_HOSTS:?}"
INDEX="${JOB_COMPLETION_INDEX:-0}"

# shellcheck source=.buildkite/kubernetes/multihost_lib.sh
# shellcheck disable=SC1091
source "${HERE}/multihost_lib.sh"
pod_setup
use_mp_backend
export VLLM_HOST_IP="$POD_IP"

# The role's hosts in DP-rank order: host i owns DP rank i, and that rank's
# Raiden store runs in its rank-0 worker, on host i. The connector indexes this
# list by global DP rank, so it is the same list on every host of the role.
hosts=()
for ((i = 0; i < NUM_HOSTS; i++)); do
  if [[ "$i" == "$INDEX" ]]; then
    hosts+=("$POD_IP")
  else
    hosts+=("$(resolve "${ROLE_HOST_PREFIX}${i}.${ROLE_HOST_DOMAIN}" 1800)") || exit 1
  fi
done
TPU_RAIDEN_ADVERTISE_HOST="$(IFS=,; echo "${hosts[*]}")"
export TPU_RAIDEN_ADVERTISE_HOST
DP_ADDRESS="${hosts[0]}"
echo "--- ${role} host ${INDEX}/${NUM_HOSTS}: data-parallel address ${DP_ADDRESS}," \
  "advertise hosts ${TPU_RAIDEN_ADVERTISE_HOST}"

# Best effort, as k3_1p1d.sh: index 0's API server imports the model's remote
# code before any worker exists; elsewhere the workers would race to write it.
# The subshell keeps that config's variables out of this script.
if [[ "$INDEX" != "0" ]]; then
  # shellcheck source=/dev/null
  uri=$(source scripts/vllm/benchmarking/configs/kimi-k3-tp32-ep.sh >/dev/null \
    && echo "$MODEL_URI") \
    && timeout 600 python3 "$HERE/warm_remote_code.py" "$uri" \
    || echo "WARNING: could not warm the model's remote code"
fi

# The serve flags and the Raiden environment of this role, over the defaults
# run_benchmarks.sh applies before sourcing a config.
export K3_1P1D_ROLE="$role"
MODEL_URI="${MODEL_URI:-}"
ENABLE_EP=false
QUANTIZATION=""
ATTENTION_BACKEND="CUSTOM"
ASYNC_SCHEDULING=true
EXTRA_SERVE_ARGS=""
# shellcheck source=/dev/null
source "scripts/vllm/benchmarking/configs/${CONFIG}.sh" || exit 2
mp_serve_args || exit 2
export PYTHONUNBUFFERED=1

# K3_1P1D_DRY_RUN=1: print the serve command and environment and stop.
if [[ "${K3_1P1D_DRY_RUN:-0}" == "1" ]]; then
  printf '[cmd] vllm serve'; printf ' %q' "${SERVE_ARGS[@]}"; echo
  env | grep -E '^(TPU_|VLLM_|K3_)' | sort
  exit 0
fi

if [[ "$INDEX" != "0" ]]; then
  mp_headless_worker
  exit $?
fi

run_dir="/tmp/k3_1p1d/${role}"
mkdir -p "$run_dir"

echo "--- Starting the ${role} head server (DP ${DP_LOCAL} of ${DATA_PARALLELISM})"
# One API frontend for all DP ranks, head only; see run_benchmarks.sh.
vllm serve "${SERVE_ARGS[@]}" --api-server-count=1 --host 0.0.0.0 --port "$PORT" &
serve_pid=$!

if [[ "$role" == "decode" ]]; then
  wait "$serve_pid"
  exit $?
fi

# K3_PD_EXTERNAL_CLIENT=1: something outside this role drives the pair, so this
# head only serves. Unset, this head IS the driver and its finishing ends the
# run, so falling through here would tear its own engine down mid-benchmark.
if [[ "${K3_PD_EXTERNAL_CLIENT:-0}" == "1" ]]; then
  echo "--- K3_PD_EXTERNAL_CLIENT: serving until the external client finishes"
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

wait_ready "http://127.0.0.1:${PORT}/health" "prefill engine" "$serve_pid" || exit 1
wait_ready "http://${DECODE_HOST}:${PORT}/health" "decode engine" || exit 1

# The workers log the block they settled on, on their own hosts; this is it.
for url in "http://127.0.0.1:${PORT}" "http://${DECODE_HOST}:${PORT}"; do
  echo "cache config at ${url}:"
  curl -sf --max-time 30 "${url}/metrics" | grep -m1 '^vllm:cache_config_info' || true
done

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
  "http://127.0.0.1:${PROXY_PORT}/v1/completions" moonshotai/Kimi-K3 || {
  tail -50 "$run_dir/proxy.log" || true
  exit 1
}

# The eval runs even if a stage fails: the pair holds 32 chips, and there is no
# sense spending them to learn one fact. Not if the stages could not start,
# though, which is this script's fault or the config's and not the pair's. The
# token-comparing stages hold to the last DP rank, whose store is on the role's
# last host: a cross-host handoff, which is what DP adds to the pair.
stages_rc=0
echo "--- Correctness stages through the proxy"
PROXY_PORT="$PROXY_PORT" \
PREFILL_URL="http://127.0.0.1:${PORT}" \
DECODE_URL="http://${DECODE_HOST}:${PORT}" \
PREFILL_CHUNK_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
K3_PD_DP_RANK="$((DATA_PARALLELISM - 1))" \
LOG_DIR="${ARTIFACTS_DIR:-$run_dir}/k3_1p1d_correctness" \
  bash scripts/vllm/integration/run_k3_1p1d_disagg_correctness.sh || stages_rc=$?
if [[ "$stages_rc" -gt 1 ]]; then
  echo "ERROR: the correctness stages did not start (exit ${stages_rc})"
  publish_results
  exit 1
fi
rc="$stages_rc"

echo "--- lm_eval through the proxy"
bash scripts/vllm/benchmarking/run_eval_flow.sh \
  --config "$CONFIG" --baseline-config kimi-k3-tp32-ep --run-lm-eval \
  --host 127.0.0.1 --port "$PROXY_PORT" \
  --skip-benchmark --skip-db-upload || rc=1

# A failed upload decides the step only when the work itself passed.
if ! publish_results && [[ "$rc" -eq 0 ]]; then
  rc=1
fi
exit "$rc"
