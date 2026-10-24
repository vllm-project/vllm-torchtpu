#!/bin/bash
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
# Expert placement parity rig: does TPU_EXPERT_PLACEMENT leave what the server
# generates unchanged?
#
# Serves a benchmark config twice through run_benchmarks.sh --serve-only, once
# without a placement and once with --placement, and drives each server with
# the same greedy prompts, one request at a time, pinned to one DP rank, so the
# two see identical batch geometry (smoke_expert_placement_parity.py). A
# placement moves experts between EP ranks and nothing else, so the two must
# produce the same tokens, and with a DSpark drafter, whose MoE layers are
# placed too, the same speculative-decoding acceptance. Unlike the unit tests,
# this exercises the fetch-time expert filter on the real checkpoint, for the
# target's layers.<L> and the drafter's mtp.<k> tensors, on every EP rank.
# MOE_FUSED_EP_SHARED_BEFORE_KERNEL stays 0 in both: it may round the shared +
# routed sum differently on its own.
#
# Run it on the head of a Ray cluster spanning the slice, the way every v7x-16
# step runs, or on one host for a single-host config:
#
#   .buildkite/kubernetes/run_multihost.sh --topology 2x2x2 \
#       bash scripts/vllm/integration/run_dsv4_expert_placement_correctness.sh
#
#   bash scripts/vllm/integration/run_dsv4_expert_placement_correctness.sh \
#       [--config NAME] [--placement NAME|PATH|gs://URI] \
#       [--spec-config JSON|none] [--variants baseline,placed] \
#       [--out-dir DIR] [--max-tokens N] [--repeats N] \
#       [--prompts-file FILE] [--dp-rank R] [--port P]
#
# Each bring-up is a full weight load and compile, about an hour cold on
# DSv4-Pro. The two programs differ (the placement is in the compile cache
# key), so each compiles once; point VLLM_CACHE_ROOT at persistent storage to
# reuse the compiles on a rerun. --variants placed alone reruns just that arm;
# compare it with an earlier baseline record afterwards:
#
#   python3 scripts/vllm/integration/smoke_expert_placement_parity.py \
#       compare <baseline>/outputs.json <placed>/outputs.json
#
# Outputs, under --out-dir: <variant>/{server.log,launcher.log,outputs.json,
# collect.log} and compare.log. Exit status: 0 parity, 1 mismatch or a failed
# check, 2 inconclusive (a server disagreed with itself), 3 setup or launch
# failure.
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../../.." && pwd)"
bench_dir="${repo_dir}/scripts/vllm/benchmarking"

config="deepseek-v4-pro-dp16-ep"
placement="dsv4_pro_agentx_ep16"
# The drafter the PR's A/B served. num_speculative_tokens has to be the
# checkpoint's DSpark block size (the proposer refuses any other); "none"
# serves without a drafter, which leaves the drafter's layers unchecked.
spec_config='{"method":"dspark","num_speculative_tokens":4}'
variants="baseline,placed"
out_dir="${ARTIFACTS_DIR:-${TMPDIR:-/tmp}}/dsv4_expert_placement"
max_tokens=0
repeats=2
prompts_file=""
dp_rank=0
port="${PORT:-8000}"
python_bin="${PYTHON_BIN:-python3}"

usage() {
  sed -n '/^#   bash/,/^#  *\[--prompts-file/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,3\}//'
  exit 3
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:?}"; shift 2 ;;
    --placement) placement="${2:?}"; shift 2 ;;
    --spec-config) spec_config="${2:?}"; shift 2 ;;
    --variants) variants="${2:?}"; shift 2 ;;
    --out-dir) out_dir="${2:?}"; shift 2 ;;
    --max-tokens) max_tokens="${2:?}"; shift 2 ;;
    --repeats) repeats="${2:?}"; shift 2 ;;
    --prompts-file) prompts_file="${2:?}"; shift 2 ;;
    --dp-rank) dp_rank="${2:?}"; shift 2 ;;
    --port) port="${2:?}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1"; usage ;;
  esac
done
[[ "${spec_config}" == "none" ]] && spec_config=""

# Print to stdout: Buildkite surfaces stdout, and the reason has to be readable
# there without opening the artifacts.
fail() {
  echo "DSV4_EXPERT_PLACEMENT_RIG_FAIL: $1"
  exit "${2:-1}"
}
setup_fail() {
  echo "DSV4_EXPERT_PLACEMENT_RIG_FAIL: $1"
  exit 3
}

config_file="${bench_dir}/configs/${config}.sh"
[[ -f "${config_file}" ]] || setup_fail "no benchmark config ${config_file}"
# run_benchmarks.sh hands EXTRA_SERVE_ARGS to the shell unquoted, one word per
# space, so a JSON argument cannot carry one.
[[ "${spec_config}" != *" "* ]] || setup_fail "--spec-config must not contain spaces"
case ",${variants}," in
  *,baseline,*|*,placed,*) ;;
  *) setup_fail "--variants takes baseline, placed or both" ;;
esac

# What the config serves, read in a subshell so its exports stay there.
read -r served_model ep_size ready_wait_min < <(
  set +u
  # shellcheck disable=SC1090
  source "${config_file}" >/dev/null 2>&1
  echo "${MODEL:-?} $(( ${TENSOR_PARALLELISM:-1} * ${DATA_PARALLELISM:-1} )) \
${SERVER_READY_WAIT_MIN:-90}"
)
startup_timeout_s=$(( ready_wait_min * 60 ))
base="http://127.0.0.1:${port}"

# Refuse a table that cannot serve this config before an hour of bring-up.
if [[ ",${variants}," == *,placed,* ]]; then
  "${python_bin}" - "${placement}" "${ep_size}" <<'PY' || setup_fail "placement check failed"
import json
import sys

from vllm_torchtpu import expert_placement

value, ep_size = sys.argv[1], int(sys.argv[2])
path = expert_placement.resolve(value)
if path.startswith("gs://"):
    print(f"placement {value}: {path}, read by the server")
    sys.exit(0)
with open(path) as f:
    table = expert_placement.parse(json.load(f), path)
if table.shards != ep_size:
    sys.exit(f"placement {value} is for {table.shards} EP ranks; the config runs {ep_size}")
print(f"placement {value}: {table.experts} experts over {table.shards} ranks, "
      f"{len(table.layers)} placed layers")
PY
fi

mkdir -p "${out_dir}" || setup_fail "cannot create ${out_dir}"
echo "===== ${config}: ${served_model}, EP=${ep_size}, variants=${variants}, \
spec=${spec_config:-none}, out=${out_dir} ====="

LAUNCHER_PID=""

# Wait until Ray reports every TPU free again, so the next bring-up does not
# race the last one's actors on the other hosts. Not on Ray: nothing to wait
# for. The answer comes back through the exit status: Ray floods stdout once a
# driver attaches.
wait_tpus_free() {
  "${python_bin}" - >/dev/null 2>&1 <<'PY'
import sys
import time

try:
    import ray

    ray.init(address="auto", logging_level="error")
except Exception:
    sys.exit(0)
deadline = time.time() + 900
while time.time() < deadline:
    total = ray.cluster_resources().get("TPU", 0)
    if total and ray.available_resources().get("TPU", 0) >= total:
        sys.exit(0)
    time.sleep(10)
sys.exit(1)
PY
}

# shellcheck disable=SC2317  # also invoked via the EXIT trap
stop_server() {
  if [[ -n "${LAUNCHER_PID}" ]] && kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
    # run_benchmarks.sh traps TERM and stops the server it started.
    kill -TERM "${LAUNCHER_PID}" 2>/dev/null || true
    wait "${LAUNCHER_PID}" 2>/dev/null || true
  fi
  LAUNCHER_PID=""
  # Only for what the harness's own stop left behind: cleanup_server.sh also
  # kills whatever else holds this host's TPU devices.
  if pgrep -f "vllm serve|VLLM::|vllm\.entrypoints" >/dev/null; then
    bash "${bench_dir}/cleanup_server.sh" >/dev/null 2>&1 || true
  fi
  wait_tpus_free || echo "WARNING: Ray still reports TPUs in use after 15 minutes"
}
trap stop_server EXIT

show_tail() {
  local dir=$1
  echo "--- last lines of ${dir}/server.log"
  tail -60 "${dir}/server.log" 2>/dev/null || true
  echo "--- last lines of ${dir}/launcher.log"
  tail -20 "${dir}/launcher.log" 2>/dev/null || true
}

# Serve one arm and leave it ready to answer on ${base}.
serve() {
  local variant=$1 value=$2 dir=$3
  mkdir -p "${dir}" || setup_fail "cannot create ${dir}"
  (
    # Empty in the baseline, not unset, so a value the raylets inherited is
    # overridden on every engine too.
    export TPU_EXPERT_PLACEMENT="${value}"
    export MOE_FUSED_EP_SHARED_BEFORE_KERNEL=0
    # Ray's DP engine actors receive only VLLM_-prefixed variables and the
    # names a config lists (vllm.ray.ray_env.get_env_vars_to_copy). Without
    # this the placement would reach no engine, and the rig would compare the
    # baseline with itself.
    export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="${VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY:+${VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY},}TPU_EXPERT_PLACEMENT,MOE_FUSED_EP_SHARED_BEFORE_KERNEL"
    # Every rank's "Expert placement ... applied" line has to reach the log.
    export RAY_DEDUP_LOGS=0
    export CAPTURE_PROFILE=0
    export ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"
    if [[ -n "${spec_config}" ]]; then
      export EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+${EXTRA_SERVE_ARGS} }--speculative-config=${spec_config}"
    fi
    env | grep -E '^(TPU_EXPERT_PLACEMENT|MOE_FUSED_EP_[A-Z_]*|VLLM_RAY_EXTRA_[A-Z_]*|ASYNC_SCHEDULING|EXTRA_SERVE_ARGS|MODEL_URI)=' \
      | sort >"${dir}/env.txt"
    exec bash "${bench_dir}/run_benchmarks.sh" --config "${config}" --serve-only \
      --results-dir "${dir}" --port "${port}"
  ) >"${dir}/launcher.log" 2>&1 &
  LAUNCHER_PID=$!
  echo "[${variant}] launching (TPU_EXPERT_PLACEMENT='${value}'), logs in ${dir}"

  local deadline=$((SECONDS + startup_timeout_s))
  until curl -sf --max-time 10 "${base}/health" >/dev/null; do
    if ! kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
      show_tail "${dir}"
      setup_fail "[${variant}] server exited during startup (see ${dir})"
    fi
    [[ "${SECONDS}" -lt "${deadline}" ]] \
      || setup_fail "[${variant}] server not healthy in ${startup_timeout_s}s"
    sleep 15
  done
  # The same warm-up in both arms, on the pinned rank, so the engines are
  # ready and the speculative-decoding counters start from the same place.
  until curl -sf --max-time 300 -X POST "${base}/v1/completions" \
      -H "Content-Type: application/json" -H "X-data-parallel-rank: ${dp_rank}" \
      -d "{\"model\": \"${served_model}\", \"prompt\": \"Hello\", \"max_tokens\": 2, \"temperature\": 0}" \
      >/dev/null; do
    if ! kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
      show_tail "${dir}"
      setup_fail "[${variant}] server exited during warm up (see ${dir})"
    fi
    [[ "${SECONDS}" -lt "${deadline}" ]] \
      || setup_fail "[${variant}] engine not ready in ${startup_timeout_s}s"
    echo "[${variant}] engine not ready yet, waiting 30s..."
    sleep 30
  done
  echo "[${variant}] ready"
}

# The log is where a placement that never reached the engines, or one that
# leaked into the baseline, shows. Ray forwards every engine's lines to the
# driver's log, prefixed with the actor.
check_log() {
  local variant=$1 log=$2 ranks
  ranks=$(grep -oE "Expert placement [^ ]+ applied: rank [0-9]+ of [0-9]+" "${log}" \
    | sed -E 's/.*rank ([0-9]+) of.*/\1/' | sort -un | wc -l)
  grep -q "Fused EP MoE armed" "${log}" \
    || echo "WARNING: [${variant}] no 'Fused EP MoE armed' line in ${log}"
  if [[ "${variant}" == "baseline" ]]; then
    [[ "${ranks}" -eq 0 ]] \
      || fail "[baseline] ${ranks} ranks applied a placement: TPU_EXPERT_PLACEMENT leaked in"
    return
  fi
  [[ "${ranks}" -gt 0 ]] \
    || fail "[placed] no rank logged 'Expert placement ... applied': the placement never reached an engine"
  if [[ "${ranks}" -lt "${ep_size}" ]]; then
    echo "WARNING: [placed] ${ranks}/${ep_size} ranks logged the placement" \
      "(engine logs not forwarded to the driver?)"
  else
    echo "[placed] all ${ep_size} ranks applied the placement"
  fi
  if grep -q "TPU_EXPERT_PLACEMENT lists" "${log}"; then
    fail "[placed] a placed layer's MoE method does not apply placements (see ${log})"
  fi
  if grep -q "MOE_FUSED_EP_SORTED_GATHER does not cover" "${log}"; then
    fail "[placed] placed layers fell back from the sorted gather (see ${log})"
  fi
}

run_variant() {
  local variant=$1 value=$2
  local dir="${out_dir}/${variant}"
  local smoke_args=(collect --base "${base}" --model "${served_model}"
    --label "${variant}" --dp-rank "${dp_rank}" --repeats "${repeats}"
    --max-tokens "${max_tokens}" --out "${dir}/outputs.json")
  [[ -n "${prompts_file}" ]] && smoke_args+=(--prompts-file "${prompts_file}")
  serve "${variant}" "${value}" "${dir}"
  "${python_bin}" "${script_dir}/smoke_expert_placement_parity.py" \
    "${smoke_args[@]}" 2>&1 | tee "${dir}/collect.log"
  local rc="${PIPESTATUS[0]}"
  stop_server
  check_log "${variant}" "${dir}/server.log"
  [[ "${rc}" -eq 0 ]] || fail "[${variant}] collect exited ${rc} (see ${dir}/collect.log)" 3
}

[[ ",${variants}," == *,baseline,* ]] && run_variant baseline ""
[[ ",${variants}," == *,placed,* ]] && run_variant placed "${placement}"

if [[ ",${variants}," != *,baseline,* || ",${variants}," != *,placed,* ]]; then
  echo "one arm only; compare it with the other arm's record:"
  echo "  ${python_bin} ${script_dir}/smoke_expert_placement_parity.py compare" \
    "<baseline>/outputs.json <placed>/outputs.json"
  exit 0
fi

"${python_bin}" "${script_dir}/smoke_expert_placement_parity.py" compare \
  "${out_dir}/baseline/outputs.json" "${out_dir}/placed/outputs.json" \
  2>&1 | tee "${out_dir}/compare.log"
compare_rc="${PIPESTATUS[0]}"
case "${compare_rc}" in
  0) echo "DSV4_EXPERT_PLACEMENT_RIG_OK" ;;
  2) fail "a server disagreed with itself; see ${out_dir}/compare.log" 2 ;;
  *) fail "placed and baseline outputs differ; see ${out_dir}/compare.log" "${compare_rc}" ;;
esac
