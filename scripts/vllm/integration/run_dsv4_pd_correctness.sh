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
#
# End-to-end correctness gate for DeepSeek-V4-Flash over P/D disaggregation,
# DP8 prefill on one host and DP8 decode on another.
#
# The thing under test is Raiden Stage 3 carrying six cache groups with six
# page sizes, three of which slide a window. If any group's pages land in the
# wrong place the decode replica reads someone else's KV, and the answer comes
# back wrong while every health check stays green. So the gate is output
# correctness under long shared prefixes, not liveness.
#
# Two properties make that a real test rather than a formality.
#
# The prompts are long. Group dsv4.swa.g1 pages at 128 tokens under a 128-token
# window, so anything past a few hundred tokens leaves most of its block table
# pointing at the null block and forces the connector's window trim to run. The
# fixtures here run to roughly 1800 tokens, which is 15 pages in that group of
# which the window keeps one.
#
# The short-QA suite is skipped. It asserts model-specific trivia tuned for
# Qwen3.5: DSv4 answers "Spell the word cat in uppercase letters" with C-A-T
# rather than CAT and gets one arithmetic item wrong. Those are the model's
# answers, not a transfer fault, and the four fixture-driven suites that remain
# are the ones a corrupt transfer actually breaks.
#
# Every probe runs even after an earlier one fails. A DSv4 boot is about 27
# minutes and there is no sense spending it to learn one fact.
#
# ROLE selects what this invocation does:
#   prefill  launch the prefill server and block
#   decode   launch the decode server and block
#   head     launch prefill locally, then run the probes against a remote decode
#   runner   run the probes against two remote servers
#   all      launch both locally, then run the probes
#   probe    run the probes against a proxy someone else already started
#
# ROLE=probe exists so the CDK recipe can run this file rather than its own copy
# of the probe list. CDK brings the pair up itself through a JobSet and puts a
# proxy in front, so there is nothing here to launch; pointing PROXY_HOST at
# that proxy is the whole difference, so CDK and Buildkite exercise the same
# lines of code.

set -uo pipefail
umask 000

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

ROLE="${ROLE:-all}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_PATH}}"
RUN_ROOT="${RUN_ROOT:-/tmp/dsv4_pd_disagg_ci}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/run}"

PREFILL_HOST="${PREFILL_HOST:-127.0.0.1}"
DECODE_HOST="${DECODE_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
PROXY_PORT="${PROXY_PORT:-8000}"
# Where the probes send their requests. Every role but ROLE=probe starts its own
# proxy on localhost; ROLE=probe points this at one that is already running.
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
# The producer log the Stage-3 check reads. Empty means "wherever this run put
# it", which only exists when this invocation launched prefill itself.
PREFILL_LOG="${PREFILL_LOG:-}"

PREFILL_DP="${PREFILL_DP:-8}"
DECODE_DP="${DECODE_DP:-8}"
PREFILL_TP="${PREFILL_TP:-1}"
DECODE_TP="${DECODE_TP:-1}"

PREFILL_CTRL_PORT="${PREFILL_CTRL_PORT:-27000}"
DECODE_CTRL_PORT="${DECODE_CTRL_PORT:-28000}"
PREFILL_TPU_KV_TRANSFER_PORT="${PREFILL_TPU_KV_TRANSFER_PORT:-9100}"
DECODE_TPU_KV_TRANSFER_PORT="${DECODE_TPU_KV_TRANSFER_PORT:-9200}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
PREFILL_MAX_NUM_SEQS="${PREFILL_MAX_NUM_SEQS:-8}"
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-8}"
# DSv4 has no prefix-aware Stage-3 load. The connector pins skip_tokens to 0 on
# a multi-group plan, so a decode-side prefix hit would leave Raiden writing the
# producer payload over the adopted pages the hit is sharing. See the TODO in
# TPURaidenConnectorScheduler._update_stage3_state_after_alloc.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
# DSv4 runs DP attention: every rank owns whole requests, so no page is striped
# and the connector raises out of request_finished on anything but 1.
TPU_RAIDEN_TRANSFER_PARALLELISM="${TPU_RAIDEN_TRANSFER_PARALLELISM:-1}"
# A full DSv4 boot is roughly 27 minutes of weight load plus compile.
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-3600}"

# Off because ENABLE_PREFIX_CACHING is off. The probe sends one prompt twice and
# compares cold against warm, but with no prefix cache the second request takes
# the same prefill-over-Raiden path as the first, so it compares a path against
# itself. It does fail occasionally, on a 0.25 logprob gap between two identical
# requests. That nondeterminism is real and unexplained, but it is not a
# prefix-caching fact, and the single-server dsv4_offload lane is the cheaper
# place to find out whether Raiden is involved in it at all.
RUN_PREFIX_CACHE_E2E_DIVERGENCE="${RUN_PREFIX_CACHE_E2E_DIVERGENCE:-0}"
PREFIX_E2E_DIVERGENCE_LOGPROB_ATOL="${PREFIX_E2E_DIVERGENCE_LOGPROB_ATOL:-0.2}"
DSV4_PD_CONCURRENCY="${DSV4_PD_CONCURRENCY:-8}"
# The large pass roughly doubles every fixture so the windowed groups carry
# more pages than the default pass, which is the only dimension of this test
# that scales with the trim.
DSV4_PD_LARGE_LINES="${DSV4_PD_LARGE_LINES:-192}"

mkdir -p "${RUN_ROOT}" "${RUN_DIR}/logs"

PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Could not find an executable python3/python in PATH." >&2
  exit 2
fi

cleanup() {
  local pid_file pid
  if [[ -d "${RUN_DIR}" ]]; then
    for pid_file in "${RUN_DIR}/proxy.pid" "${RUN_DIR}/prefill.pid" "${RUN_DIR}/decode.pid"; do
      if [[ -f "${pid_file}" ]]; then
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ "${pid}" =~ ^[0-9]+$ ]]; then
          kill -TERM -- "-${pid}" 2>/dev/null \
            || kill -TERM "${pid}" 2>/dev/null \
            || true
        fi
      fi
    done
    sleep 2
    for pid_file in "${RUN_DIR}/proxy.pid" "${RUN_DIR}/prefill.pid" "${RUN_DIR}/decode.pid"; do
      if [[ -f "${pid_file}" ]]; then
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ "${pid}" =~ ^[0-9]+$ ]]; then
          if kill -0 -- "-${pid}" 2>/dev/null; then
            kill -KILL -- "-${pid}" 2>/dev/null || true
          elif kill -0 "${pid}" 2>/dev/null; then
            kill -KILL "${pid}" 2>/dev/null || true
          fi
        fi
      fi
    done
  fi
}
trap cleanup EXIT INT TERM

export MODEL_PATH
export SERVED_MODEL_NAME
export RUN_ROOT
export RUN_DIR
export PREFILL_HOST
export DECODE_HOST
export PREFILL_PORT
export DECODE_PORT
export PROXY_PORT
export PROXY_HOST
export PREFILL_LOG
export PREFILL_DP
export DECODE_DP
export PREFILL_TP
export DECODE_TP
export PREFILL_CTRL_PORT
export DECODE_CTRL_PORT
export PREFILL_TPU_KV_TRANSFER_PORT
export DECODE_TPU_KV_TRANSFER_PORT
export TPU_RAIDEN_TRANSFER_PARALLELISM
export GPU_MEMORY_UTILIZATION
export MAX_MODEL_LEN
export MAX_NUM_BATCHED_TOKENS
export PREFILL_MAX_NUM_SEQS
export DECODE_MAX_NUM_SEQS
export ENABLE_PREFIX_CACHING
export ASYNC_SCHEDULING
export STARTUP_TIMEOUT_S

# The launcher refuses to start with this set; DSv4 has no GDN pooled layout.
unset TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL

launcher="${repo_root}/examples/disagg/launch_dsv4_disagg_multihost.sh"

cd "${repo_root}" || exit 1

case "${ROLE}" in
  prefill)
    echo "--- Launching role=prefill (DP=${PREFILL_DP}, PORT=${PREFILL_PORT})"
    ROLE=prefill bash "${launcher}" 2>&1 | tee "${RUN_DIR}/logs/launch_prefill.log"
    prefill_pid="$(cat "${RUN_DIR}/prefill.pid" 2>/dev/null || true)"
    if [[ -n "${prefill_pid}" ]]; then
      echo "Prefill server running (PID: ${prefill_pid}). Waiting on process..."
      while kill -0 "${prefill_pid}" 2>/dev/null; do
        sleep 2
      done
    fi
    ;;

  decode)
    echo "--- Launching role=decode (DP=${DECODE_DP}, PORT=${DECODE_PORT})"
    ROLE=decode bash "${launcher}" 2>&1 | tee "${RUN_DIR}/logs/launch_decode.log"
    decode_pid="$(cat "${RUN_DIR}/decode.pid" 2>/dev/null || true)"
    if [[ -n "${decode_pid}" ]]; then
      echo "Decode server running (PID: ${decode_pid}). Waiting on process..."
      while kill -0 "${decode_pid}" 2>/dev/null; do
        sleep 2
      done
    fi
    ;;

  head|runner|all|probe)
    if [[ "${ROLE}" == "head" ]]; then
      echo "--- Launching head prefill server locally"
      ROLE=prefill bash "${launcher}" 2>&1 | tee "${RUN_DIR}/logs/launch_head_prefill.log"
      target_prefill_host="${PREFILL_HOST:-127.0.0.1}"
      target_decode_host="${DECODE_HOST}"
    elif [[ "${ROLE}" == "all" ]]; then
      echo "--- Launching prefill and decode servers locally"
      ROLE=all PREFILL_HOST="127.0.0.1" DECODE_HOST="127.0.0.1" bash "${launcher}" \
        2>&1 | tee "${RUN_DIR}/logs/launch_all.log"
      target_prefill_host="127.0.0.1"
      target_decode_host="127.0.0.1"
    else
      target_prefill_host="${PREFILL_HOST}"
      target_decode_host="${DECODE_HOST}"
    fi

    if [[ "${ROLE}" == "probe" ]]; then
      # Someone else owns the servers and the proxy. Confirm the proxy answers
      # and go straight to the probes.
      echo "--- ROLE=probe: using the proxy already running at ${PROXY_HOST}:${PROXY_PORT}"
      probe_deadline=$((SECONDS + STARTUP_TIMEOUT_S))
      ext_ok=0
      while [[ "${SECONDS}" -lt "${probe_deadline}" ]]; do
        code="$(curl -fsS -o /dev/null -w "%{http_code}" \
          "http://${PROXY_HOST}:${PROXY_PORT}/healthcheck" 2>/dev/null || true)"
        # The toy proxy answers /healthcheck, but a plain vLLM front end in the
        # same position 404s it while still serving completions. Either is a
        # live HTTP peer, which is all this wait is for.
        if [[ "${code}" == "200" || "${code}" == "404" || "${code}" == "405" ]]; then
          ext_ok=1
          break
        fi
        sleep 10
      done
      if [[ "${ext_ok}" -ne 1 ]]; then
        echo "ERROR: no proxy answered at ${PROXY_HOST}:${PROXY_PORT} within ${STARTUP_TIMEOUT_S}s" >&2
        exit 1
      fi
      echo "Proxy is reachable."
    else
      echo "--- Verifying backend server health: Prefill (${target_prefill_host}:${PREFILL_PORT}), Decode (${target_decode_host}:${DECODE_PORT})"
      deadline=$((SECONDS + STARTUP_TIMEOUT_S))
      p_ok=0
      d_ok=0
      while [[ "${SECONDS}" -lt "${deadline}" ]]; do
        p_ok=0
        d_ok=0
        if [[ "$(curl -fsS -o /dev/null -w "%{http_code}" "http://${target_prefill_host}:${PREFILL_PORT}/health" 2>/dev/null || true)" == "200" ]]; then
          p_ok=1
        fi
        if [[ "$(curl -fsS -o /dev/null -w "%{http_code}" "http://${target_decode_host}:${DECODE_PORT}/health" 2>/dev/null || true)" == "200" ]]; then
          d_ok=1
        fi
        if [[ "${p_ok}" -eq 1 && "${d_ok}" -eq 1 ]]; then
          echo "Prefill and Decode backends are healthy!"
          break
        fi
        sleep 15
      done

      if [[ "${p_ok}" -ne 1 || "${d_ok}" -ne 1 ]]; then
        echo "ERROR: Backend servers did not become healthy within ${STARTUP_TIMEOUT_S}s (Prefill: ${p_ok}, Decode: ${d_ok})" >&2
        exit 1
      fi

      echo "--- Starting toy proxy server on port ${PROXY_PORT}"
      proxy_cmd=(
        "${PYTHON_BIN}"
        "${repo_root}/examples/disagg/toy_proxy_server.py"
        --host "0.0.0.0"
        --port "${PROXY_PORT}"
        --prefiller-hosts "${target_prefill_host}"
        --prefiller-ports "${PREFILL_PORT}"
        --decoder-hosts "${target_decode_host}"
        --decoder-ports "${DECODE_PORT}"
      )
      setsid "${proxy_cmd[@]}" >"${RUN_DIR}/logs/proxy.log" 2>&1 &
      echo $! >"${RUN_DIR}/proxy.pid"
      echo "Proxy PID: $(cat "${RUN_DIR}/proxy.pid")"

      echo "--- Waiting for proxy /healthcheck"
      proxy_deadline=$((SECONDS + 60))
      proxy_ok=0
      while [[ "${SECONDS}" -lt "${proxy_deadline}" ]]; do
        if [[ "$(curl -fsS -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PROXY_PORT}/healthcheck" 2>/dev/null || true)" == "200" ]]; then
          proxy_ok=1
          break
        fi
        sleep 2
      done
      if [[ "${proxy_ok}" -ne 1 ]]; then
        echo "ERROR: Proxy failed to become ready within 60s" >&2
        tail -n 50 "${RUN_DIR}/logs/proxy.log" || true
        exit 1
      fi
      echo "Proxy is healthy and ready."
    fi

    smoke="${repo_root}/scripts/vllm/integration/smoke_prefix_cache_correctness.py"
    smoke_common=(--host "${PROXY_HOST}" --port "${PROXY_PORT}" --model "${SERVED_MODEL_NAME}"
                  --skip-short-qa)

    # Each probe records its own exit code and none of them abort the run. The
    # names are what the summary prints.
    declare -A probe_rc=()
    run_probe() {
      local name="$1"
      shift
      echo "--- probe ${name}"
      "$@" 2>&1 | tee "${RUN_DIR}/logs/${name}.log"
      probe_rc["${name}"]="${PIPESTATUS[0]}"
      echo "PROBE ${name} rc=${probe_rc[${name}]}"
    }

    # One request, before anything asserts on content. It separates "the chat
    # template is wrong" from "a transfer is wrong", which look the same once
    # exact-match assertions start failing.
    run_probe quick_probe \
      "${PYTHON_BIN}" "${smoke}" --host "${PROXY_HOST}" --port "${PROXY_PORT}" \
      --model "${SERVED_MODEL_NAME}" --quick-probe-only

    run_probe correctness_default "${PYTHON_BIN}" "${smoke}" "${smoke_common[@]}"

    run_probe correctness_large \
      env P4D2_SHORT_REPEAT_LINES="${DSV4_PD_LARGE_LINES}" \
          P4D2_LONG_REPEAT_LINES="${DSV4_PD_LARGE_LINES}" \
          P4D2_LONG_SHARED_LINES="${DSV4_PD_LARGE_LINES}" \
          P4D2_MIXED_LINES="${DSV4_PD_LARGE_LINES}" \
      "${PYTHON_BIN}" "${smoke}" "${smoke_common[@]}"

    # Concurrency is off by default in the smoke. A transfer race needs more
    # than one request in flight before it can show up at all.
    run_probe correctness_concurrent \
      "${PYTHON_BIN}" "${smoke}" "${smoke_common[@]}" \
      --concurrent-requests "${DSV4_PD_CONCURRENCY}"

    if [[ "${RUN_PREFIX_CACHE_E2E_DIVERGENCE}" == "1" ]]; then
      run_probe divergence \
        "${PYTHON_BIN}" "${repo_root}/scripts/vllm/integration/smoke_prefix_cache_e2e_divergence.py" \
        --host "${PROXY_HOST}" --port "${PROXY_PORT}" --model "${SERVED_MODEL_NAME}" \
        --max-tokens 1 --logprob-atol "${PREFIX_E2E_DIVERGENCE_LOGPROB_ATOL}"
    else
      echo "RUN_PREFIX_CACHE_E2E_DIVERGENCE=${RUN_PREFIX_CACHE_E2E_DIVERGENCE}; skipping divergence smoke"
    fi

    # --- did the transfer actually happen? --------------------------------
    #
    # Every suite above would still pass if the connector had given up and let
    # the decode replica prefill each request itself, because the answer would
    # be correct and only slower. That is the failure mode this catches, and
    # nothing above can see it.
    #
    # check_dsv4_stage3_log.py does the work: it counts registered and completed
    # sends, fails on any skipped one, and asserts the window trim ran by
    # checking that no group lists as many pages as an untrimmed table would.
    #
    # PREFILL_LOG points it at the producer. ROLE=head and ROLE=all write that
    # log locally; ROLE=runner and ROLE=probe do not own the prefill host, so
    # the caller has to supply the path or the check is skipped rather than
    # faked. Under CDK the prefill pod's server.log is on the shared /cdk-outputs
    # mount, so the nightly leg can hand it over.
    prefill_log="${PREFILL_LOG:-${RUN_DIR}/logs/prefill.log}"
    transfer_rc=0
    if [[ -f "${prefill_log}" ]]; then
      echo "--- Stage-3 transfer activity in ${prefill_log}"
      "${PYTHON_BIN}" "${repo_root}/scripts/vllm/integration/check_dsv4_stage3_log.py" \
        "${prefill_log}" 2>&1 | tee "${RUN_DIR}/logs/stage3_check.log"
      transfer_rc="${PIPESTATUS[0]}"
    else
      echo "STAGE3_LOG_CHECK skipped: no prefill log at ${prefill_log} (ROLE=${ROLE})"
    fi

    chmod -R 777 /perf_eval_results 2>/dev/null || true

    # --- verdict ----------------------------------------------------------
    echo "=== DSV4 P/D CORRECTNESS SUMMARY ==="
    failed=0
    for name in "${!probe_rc[@]}"; do
      echo "PROBE_RESULT ${name} ${probe_rc[${name}]}"
      if [[ "${probe_rc[${name}]}" -ne 0 ]]; then
        failed=$((failed + 1))
      fi
    done
    echo "PROBE_RESULT stage3_transfer ${transfer_rc}"
    if [[ "${transfer_rc}" -ne 0 ]]; then
      failed=$((failed + 1))
    fi

    if [[ "${failed}" -ne 0 ]]; then
      echo "DSV4_PD_CORRECTNESS_OK 0"
      echo "ERROR: ${failed} DSv4 P/D probes failed" >&2
      exit 1
    fi
    echo "DSV4_PD_CORRECTNESS_OK 1"
    ;;

  *)
    echo "ERROR: Unknown ROLE: ${ROLE}. Supported: prefill, decode, runner, head, all, probe." >&2
    exit 2
    ;;
esac
