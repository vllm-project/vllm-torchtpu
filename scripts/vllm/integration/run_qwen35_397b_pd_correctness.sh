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

# Correctness of a Qwen3.5-397B prefill/decode pair served by the C3 recipe
# (scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-pd.sh), run against a
# pair and a proxy that are already up. .buildkite/kubernetes/qwen35_pd.sh
# runs it on the prefill head; a deployment that brings the pair up some other
# way runs this same file against its own proxy, so both exercise the same
# checks. Every stage runs even after an earlier one fails: the pair's boot is
# the expensive part.
#
# Stage 1  completions  .buildkite/kubernetes/check_completions.py
#   Greedy completions through the proxy whose answers are not in doubt.
# Stage 2  prefix_hit   qwen35_pd_checks.py prefix-hit
#   One ~8k-token prompt, sent once more than decode has API servers: decode
#   serves the repeat from its own prefix cache, prefill hits its own.
# Stage 3  gsm8k        run_eval_flow.sh --run-lm-eval, the config's
#                       PD_EVAL_FLOW_ARGS, then qwen35_pd_checks.py eval-ran
#   The config's LM_EVAL_TASKS (GSM8K) through the proxy against its
#   baselines, into a results directory of this run's own. A regression or a
#   pending baseline fails unless EVAL_SOFT_FAIL=1; a missing baseline always
#   fails. The stage also fails when lm-eval wrote no result, ran less than
#   the whole split, logged a failed request, or (baseline measured) the gate
#   printed no row for the baseline's metric.
# Stage 4  transfer     qwen35_pd_checks.py transfer
#   Over the whole run, decode received prompt KV from prefill and computed
#   at most PD_MAX_DECODE_COMPUTE_PCT percent of its prompt tokens itself.
# Stage 5  mtp          qwen35_pd_checks.py mtp
#   Decode's MTP acceptance rate is at least PD_MIN_MTP_ACCEPTANCE_RATE.
# Stage 6  legs         qwen35_pd_checks.py leg-check
#   Each leg logged at least one kernel page list, every one of them the
#   PD_KERNEL_PAGE_TOKENS page alone, and no Raiden layout-fingerprint
#   mismatch. Waits
#   LEG_STATUS_SETTLE_S (15 s) first: the engine pods refresh their status
#   every 10 s.
#
# Exit status: 0 every stage passed, 1 a stage failed or could not measure, 2
# no stage was run.
#
# Environment:
#   QWEN35_PD_CONFIG     the recipe instantiation the pair serves
#                        (default qwen3.5-397b-fp8-pd-1p1d)
#   PROXY_HOST, PROXY_PORT
#                        the proxy in front of the pair
#   PREFILL_URL          the prefill engine, for /metrics
#   DECODE_URLS          the decode leg's API servers, space-separated
#   PREFILL_LEG_STATUS, DECODE_LEG_STATUS
#                        each leg's qwen35_pd_checks.py leg-status output,
#                        space-separated files or URLs (one per engine)
#   LOG_DIR              where each stage's output is kept
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${repo_root}" || exit 2

CONFIG="${QWEN35_PD_CONFIG:-qwen3.5-397b-fp8-pd-1p1d}"
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-8100}"
for name in PREFILL_URL DECODE_URLS PREFILL_LEG_STATUS DECODE_LEG_STATUS; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} must be set" >&2
    exit 2
  fi
done
LOG_DIR="${LOG_DIR:-${ARTIFACTS_DIR:-/tmp}/qwen35_397b_pd_correctness}"
mkdir -p "${LOG_DIR}" || exit 2

# The config's model and thresholds, read in a subshell: sourcing it also
# exports an engine's environment, which this script has no use for.
config_values="$(
  # shellcheck disable=SC2034  # read by the config sourced below
  PD_ROLE=prefill
  # shellcheck source=/dev/null
  source "scripts/vllm/benchmarking/configs/${CONFIG}.sh" >&2 || exit 1
  printf '%s\n' "${MODEL}" "${PD_KERNEL_PAGE_TOKENS}" "${PD_MIN_MTP_ACCEPTANCE_RATE}" \
    "${PD_MAX_DECODE_COMPUTE_PCT}" "${PD_EVAL_FLOW_ARGS[*]}" "${LM_EVAL_TASKS}"
)" || { echo "ERROR: ${CONFIG} did not load" >&2; exit 2; }
{
  read -r MODEL
  read -r KERNEL_PAGE
  read -r MIN_MTP_RATE
  read -r MAX_DECODE_COMPUTE_PCT
  read -r -a EVAL_FLOW_ARGS
  read -r -a EVAL_TASKS
} <<<"${config_values}"

read -r -a decode_servers <<<"${DECODE_URLS}"
decode_metrics=()
for url in "${decode_servers[@]}"; do
  decode_metrics+=("${url}/metrics")
done
proxy="http://${PROXY_HOST}:${PROXY_PORT}"
checks=(python3 "${script_dir}/qwen35_pd_checks.py")

declare -A stage_rc=()
stage() {
  local name="$1" start="${SECONDS}"
  shift
  echo "--- stage ${name}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  stage_rc["${name}"]="${PIPESTATUS[0]}"
  echo "STAGE ${name} rc=${stage_rc[${name}]} in $((SECONDS - start))s"
}

# shellcheck disable=SC2317,SC2329  # run through stage
eval_stage() {
  local rc=0 task eval_dir flow_log="${LOG_DIR}/eval_flow.log"
  # A results directory of this run's own: lm-eval and the gate read every
  # results file under it, so one left by an earlier run into the same
  # LOG_DIR would stand in for a run that wrote none.
  eval_dir="$(mktemp -d "${LOG_DIR}/eval.XXXXXX")" || return 2
  # --run-lm-eval here, not from the config: without it run_eval_flow.sh runs
  # nothing and exits 0. run_benchmarks.sh sources the config to read the
  # model and the eval settings, and the config wants a role.
  PD_ROLE=prefill bash scripts/vllm/benchmarking/run_eval_flow.sh \
    --config "${CONFIG}" --run-lm-eval "${EVAL_FLOW_ARGS[@]}" --skip-benchmark \
    --skip-db-upload --results-dir "${eval_dir}" --host "${PROXY_HOST}" --port "${PROXY_PORT}" \
    2>&1 | tee "${flow_log}"
  rc="${PIPESTATUS[0]}"
  # eval-ran reads each task's metric from its baseline (the result's own
  # when the baseline is pending).
  for task in "${EVAL_TASKS[@]}"; do
    "${checks[@]}" eval-ran --results-dir "${eval_dir}" --task "${task}" \
      --flow-log "${flow_log}" \
      --baseline "scripts/vllm/benchmarking/baselines/eval/${CONFIG}.${task}.baseline.json" || rc=1
  done
  return "${rc}"
}

# shellcheck disable=SC2317,SC2329  # run through stage
leg_checks() {
  local rc=0 status
  sleep "${LEG_STATUS_SETTLE_S:-15}"
  for status in ${PREFILL_LEG_STATUS}; do
    "${checks[@]}" leg-check --status "${status}" --name prefill --page "${KERNEL_PAGE}" || rc=1
  done
  for status in ${DECODE_LEG_STATUS}; do
    "${checks[@]}" leg-check --status "${status}" --name decode --page "${KERNEL_PAGE}" || rc=1
  done
  return "${rc}"
}

stage completions python3 .buildkite/kubernetes/check_completions.py \
  "${proxy}/v1/completions" "${MODEL}"
stage prefix_hit "${checks[@]}" prefix-hit --proxy "${proxy}" --model "${MODEL}" \
  --repeats "$((${#decode_servers[@]} + 1))" \
  --prefill-metrics "${PREFILL_URL}/metrics" --decode-metrics "${decode_metrics[@]}"
stage gsm8k eval_stage
stage transfer "${checks[@]}" transfer --decode-metrics "${decode_metrics[@]}" \
  --max-compute-pct "${MAX_DECODE_COMPUTE_PCT}"
stage mtp "${checks[@]}" mtp --metrics "${decode_metrics[@]}" --min-rate "${MIN_MTP_RATE}"
stage legs leg_checks

echo "=== QWEN3.5-397B P/D CORRECTNESS SUMMARY (${CONFIG}) ==="
failed=0
for name in completions prefix_hit gsm8k transfer mtp legs; do
  echo "STAGE_RESULT ${name} ${stage_rc[${name}]}"
  [[ "${stage_rc[${name}]}" -eq 0 ]] || failed=1
done
exit "${failed}"
