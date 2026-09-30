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

# Kimi-K3 1P1D correctness, run by .buildkite/kubernetes/k3_1p1d.sh on the
# prefill head once the pair is serving. Every request goes through the proxy:
# prefill computes the prompt's KV, Raiden Stage 3 moves it, decode generates.
# All stages run even if one fails.
#
# Stage 1  quick_probe             smoke_prefix_cache_correctness.py
#   One chat request ("What is 2+2?"). Pass: the answer is "4".
#
# Stage 2  correctness             smoke_prefix_cache_correctness.py
#   Long prompts that carry an answer code, repeated and sharing prefixes, one
#   request at a time. Pass: every answer is exactly the code.
#
# Stage 3  correctness_concurrent  smoke_prefix_cache_correctness.py
#   Stage 2 again, then the shared-prefix prompts K3_PD_CONCURRENCY at a time.
#   Pass: as stage 2.
#
# Stage 4  divergence              smoke_prefix_cache_e2e_divergence.py
#   One ~4.2k-token prompt sent twice, for three prompts. Pass: the repeat
#   returns the first run's token ids, logprobs within DIVERGENCE_LOGPROB_ATOL.
#
# Stage 5  state_contamination     smoke_mamba_state_contamination.py
#   Two different prefixes X and Y, each sent once for a reference, then
#   X, Y, X, Y. Pass: every X output equals X's reference, same for Y, so a
#   request never resumes from the other's KDA state.
#
# Stage 6  counters                /metrics of both engines
#   Stages 1-5 also pass if decode prefilled every request itself, or if no
#   prefix was ever reused: a response's cached_tokens counts transferred and
#   reused tokens alike. Pass, over the whole run:
#     * decode received prompt tokens from prefill (external_kv_transfer > 0)
#     * decode computed at most MAX_DECODE_COMPUTE_PCT percent of its prompt
#       tokens itself (local_compute): it owes one token per request, and
#       more means requests fell back to prefilling on decode
#     * decode served prompt tokens from its own prefix cache, with only the
#       rest of the prompt transferred (local_cache_hit > 0)
#     * prefill hit its own prefix cache (prefix_cache_hits > 0)
#     * neither engine preempted a request (num_preemptions == 0)
#
# Stages 4 and 5 compare tokens, which needs a prefill chunk to be one KV block
# (configs/kimi-k3-tp32-ep-1p1d.sh); this refuses to start otherwise.
#
# Exit status: 0 every stage passed, 1 a stage failed, anything else no stage
# was run.
#
# Environment:
#   PROXY_HOST, PROXY_PORT   the proxy in front of the pair
#   PREFILL_URL, DECODE_URL  the engines, for /metrics
#   PREFILL_CHUNK_TOKENS     the pair's --max-num-batched-tokens
#   LOG_DIR                  where each stage's output is kept
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3}"
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-8100}"
for name in PREFILL_URL DECODE_URL PREFILL_CHUNK_TOKENS; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} must be set" >&2
    exit 2
  fi
done
LOG_DIR="${LOG_DIR:-${TMPDIR:-/tmp}/k3_1p1d_disagg_ci}"

K3_PD_CONCURRENCY="${K3_PD_CONCURRENCY:-8}"
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-{\"thinking\":false\}}"
DIVERGENCE_LOGPROB_ATOL="${DIVERGENCE_LOGPROB_ATOL:-0.2}"
MAX_DECODE_COMPUTE_PCT="${MAX_DECODE_COMPUTE_PCT:-5}"

mkdir -p "${LOG_DIR}" || exit 2

# The block size the engine settled on, which may not be the one asked for.
# From /metrics: the workers log it, and their logs stay on their own hosts.
# Anchored on the label's start: mamba_block_size sits on the same line.
block_size="$(curl -sf --max-time 30 "${PREFILL_URL}/metrics" \
  | grep -m1 '^vllm:cache_config_info' | grep -oE '[{,]block_size="[0-9]+"' \
  | grep -oE '[0-9]+')"
if [[ "${block_size}" != "${PREFILL_CHUNK_TOKENS}" ]]; then
  echo "ERROR: the engine chose a '${block_size}'-token block but prefill" \
    "chunks are ${PREFILL_CHUNK_TOKENS} tokens; the token-equality stages" \
    "need them equal"
  exit 2
fi
echo "KV block and prefill chunk: ${block_size} tokens"

# metric <base url> <name> [label]: a counter summed over its series. The label
# is a substring of the series' label set. -1 when the engine does not answer
# or has no such series, so that a counter nobody could read is never taken for
# one that reads 0.
metric() {
  local body
  if ! body="$(curl -sf --max-time 30 "$1/metrics")"; then
    echo -1
    return
  fi
  awk -v name="$2" -v label="${3:-}" '
    index($0, name) == 1 && (label == "" || index($0, label)) {s += $NF; n++}
    END {if (n) printf "%d", s; else printf "-1"}' <<<"${body}"
}

declare -A stage_rc=()
stage() {
  local name="$1" start="${SECONDS}"
  shift
  echo "--- stage ${name}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  stage_rc["${name}"]="${PIPESTATUS[0]}"
  echo "STAGE ${name} rc=${stage_rc[${name}]} in $((SECONDS - start))s"
}

check_counters() {
  local rc=0 name value unread=0
  local by_source=vllm:prompt_tokens_by_source_total
  local -A must_be_positive=(
    [decode_external_kv_transfer_tokens]="$(metric "${DECODE_URL}" \
      "${by_source}" 'source="external_kv_transfer"')"
    [decode_local_cache_hit_tokens]="$(metric "${DECODE_URL}" \
      "${by_source}" 'source="local_cache_hit"')"
    [prefill_prefix_cache_hit_tokens]="$(metric "${PREFILL_URL}" \
      vllm:prefix_cache_hits_total)"
  )
  local -A must_be_zero=(
    [prefill_preemptions]="$(metric "${PREFILL_URL}" vllm:num_preemptions_total)"
    [decode_preemptions]="$(metric "${DECODE_URL}" vllm:num_preemptions_total)"
  )
  local computed
  computed="$(metric "${DECODE_URL}" "${by_source}" 'source="local_compute"')"

  for name in "${!must_be_positive[@]}" "${!must_be_zero[@]}"; do
    value="${must_be_positive[${name}]:-${must_be_zero[${name}]:-}}"
    echo "COUNTER ${name} ${value}"
    if [[ "${value}" -lt 0 ]]; then
      echo "COUNTER_FAIL ${name} could not be read"
      unread=1
    fi
  done
  echo "COUNTER decode_local_compute_tokens ${computed}"
  if [[ "${computed}" -lt 0 ]]; then
    echo "COUNTER_FAIL decode_local_compute_tokens could not be read"
    unread=1
  fi
  # What follows compares counters, which means nothing if one is missing.
  [[ "${unread}" -eq 0 ]] || return 1

  for name in "${!must_be_positive[@]}"; do
    if [[ "${must_be_positive[${name}]}" -eq 0 ]]; then
      echo "COUNTER_FAIL ${name} is 0: no request took that path"
      rc=1
    fi
  done
  for name in "${!must_be_zero[@]}"; do
    if [[ "${must_be_zero[${name}]}" -ne 0 ]]; then
      echo "COUNTER_FAIL ${name} is ${must_be_zero[${name}]}, expected 0"
      rc=1
    fi
  done
  local transferred reused share
  transferred="${must_be_positive[decode_external_kv_transfer_tokens]}"
  reused="${must_be_positive[decode_local_cache_hit_tokens]}"
  share="$(awk -v c="${computed}" -v t="$((computed + transferred + reused))" \
    'BEGIN {printf "%d", (t > 0) ? 100 * c / t : 100}')"
  echo "decode computed ${share}% of its prompt tokens itself"
  if [[ "${share}" -gt "${MAX_DECODE_COMPUTE_PCT}" ]]; then
    echo "COUNTER_FAIL that is over ${MAX_DECODE_COMPUTE_PCT}%"
    rc=1
  fi
  return "${rc}"
}

run_stages() {
  local smoke="${script_dir}/smoke_prefix_cache_correctness.py"
  local target=(--host "${PROXY_HOST}" --port "${PROXY_PORT}"
                --model "${SERVED_MODEL_NAME}")
  local chat=("${target[@]}" --chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}")
  # --min-max-tokens: room for the six tokens K3 closes its channels with.
  # --skip-short-qa: that suite asserts trivia in Qwen3.5's phrasing.
  local fixtures=("${chat[@]}" --min-max-tokens 32 --skip-short-qa)

  stage quick_probe python3 "${smoke}" "${chat[@]}" --quick-probe-only
  stage correctness python3 "${smoke}" "${fixtures[@]}"
  stage correctness_concurrent python3 "${smoke}" "${fixtures[@]}" \
    --concurrent-requests "${K3_PD_CONCURRENCY}"
  # 120 lines is ~4.2k tokens.
  stage divergence python3 "${script_dir}/smoke_prefix_cache_e2e_divergence.py" \
    "${chat[@]}" --attempts 3 --line-count 120 \
    --logprob-atol "${DIVERGENCE_LOGPROB_ATOL}"
  stage state_contamination python3 \
    "${script_dir}/smoke_mamba_state_contamination.py" "${target[@]}" --rounds 3

  stage counters check_counters

  echo "=== K3 1P1D CORRECTNESS SUMMARY ==="
  local name failed=0
  for name in "${!stage_rc[@]}"; do
    echo "STAGE_RESULT ${name} ${stage_rc[${name}]}"
    [[ "${stage_rc[${name}]}" -eq 0 ]] || failed=1
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "K3_1P1D_CORRECTNESS_FAIL (logs in ${LOG_DIR})"
    return 1
  fi
  echo "K3_1P1D_CORRECTNESS_OK"
}

run_stages
