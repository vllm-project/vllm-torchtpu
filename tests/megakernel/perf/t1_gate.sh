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
# T1 Perf Gate script for kernel/stream/vmem/pool/module/profile changes.
#
# Usage:
#   tests/megakernel/perf/t1_gate.sh [gate] <base_ref> <head_ref> [buckets] [options]
#
# Examples:
#   tests/megakernel/perf/t1_gate.sh origin/nanwei/mk-dev HEAD
#   tests/megakernel/perf/t1_gate.sh gate origin/nanwei/mk-dev origin/nanwei/mk-b7-verify "1:8,1:32,4:4,4:16,4:32,4:64"

set -euo pipefail

if [[ $# -ge 1 && "$1" == "gate" ]]; then
  shift
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 [gate] <base_ref> <head_ref> [buckets] [--profiles PROFILES] [--mode auto|refactor|intentional] [--workers N] [--force]" >&2
  exit 2
fi

BASE_REF="$1"
HEAD_REF="$2"
shift 2

BUCKETS="1:8,1:32,4:4,4:16,4:32,4:64"
if [[ $# -ge 1 && "$1" != --* ]]; then
  BUCKETS="$1"
  shift
fi

PROFILES="${MK_T1_PROFILES:-default,qwen35-candidate30-fused}"
MODE="${MK_T1_MODE:-auto}"
WORKERS="${MK_T1_WORKERS:-4}"
FORCE=0
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profiles)
      PROFILES="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${MK_PYTHON_BIN:-/usr/local/google/home/nanwei/mk/venv312/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

cd "${REPO_ROOT}"
BASE_SHA="$(git rev-parse "${BASE_REF}")"
HEAD_SHA="$(git rev-parse "${HEAD_REF}")"
SHORT_BASE="${BASE_SHA:0:9}"
SHORT_HEAD="${HEAD_SHA:0:9}"

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="/tmp/mk_t1_gate_${SHORT_BASE}_${SHORT_HEAD}_$$"
fi
mkdir -p "${OUT_DIR}"

echo "================================================================================"
echo "[T1-GATE] Base: ${BASE_REF} (${SHORT_BASE}) -> Head: ${HEAD_REF} (${SHORT_HEAD})"
echo "[T1-GATE] Buckets: ${BUCKETS} | Profiles: ${PROFILES} | Mode: ${MODE}"
echo "[T1-GATE] Output dir: ${OUT_DIR}"
echo "================================================================================"

# Stage 0: Hot-path git diff scope check
HOTPATH_FILES="$(git diff --name-only "${BASE_SHA}" "${HEAD_SHA}" -- \
  src/vllm_torchtpu/megakernel/ \
  src/vllm_torchtpu/megakernel_runtime/ || true)"

if [[ -z "${HOTPATH_FILES}" && "${FORCE}" -eq 0 ]]; then
  echo "[T1-GATE] Stage 0: 0 files changed under megakernel/ or megakernel_runtime/."
  echo "[T1-GATE] Verdict: PASS (SKIP_NO_HOTPATH_DIFF)"
  exit 0
fi

echo "[T1-GATE] Stage 0: Hot-path files touched:"
echo "${HOTPATH_FILES}" | sed 's/^/  - /'

WT_BASE="${OUT_DIR}/wt_base"
WT_HEAD="${OUT_DIR}/wt_head"

cleanup() {
  git -C "${REPO_ROOT}" worktree remove --force "${WT_BASE}" >/dev/null 2>&1 || true
  git -C "${REPO_ROOT}" worktree remove --force "${WT_HEAD}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git -C "${REPO_ROOT}" worktree add --detach "${WT_BASE}" "${BASE_SHA}" >/dev/null 2>&1
git -C "${REPO_ROOT}" worktree add --detach "${WT_HEAD}" "${HEAD_SHA}" >/dev/null 2>&1

# Stage 1: Cloudtop AOT TPU7x full normalized LLO + Mosaic diff per subkernel
echo "[T1-GATE] Stage 1: Running Cloudtop TPU7x AOT compilation and full LLO + Mosaic diff..."
set +e
PYTHONPATH="/usr/local/google/home/nanwei/mk/tpu_aot_pkgs:${REPO_ROOT}/src:${REPO_ROOT}" \
  "${PYTHON_BIN}" -m tests.megakernel.perf.t1_gate \
  --base-wt "${WT_BASE}" \
  --head-wt "${WT_HEAD}" \
  --base-sha "${SHORT_BASE}" \
  --head-sha "${SHORT_HEAD}" \
  --profiles "${PROFILES}" \
  --buckets "${BUCKETS}" \
  --workers "${WORKERS}" \
  --mode "${MODE}" \
  --out-dir "${OUT_DIR}"
STAGE1_RC=$?
set -e

if [[ "${STAGE1_RC}" -eq 2 ]]; then
  echo "[T1-GATE] Verdict: INVALID_TOOLCHAIN (toolchain mismatch; invalid gate, not fail)"
  exit 2
elif [[ "${STAGE1_RC}" -ne 0 ]]; then
  echo "[T1-GATE] Verdict: FAIL (Stage 1 LLO/Mosaic diff detected in refactor mode)"
  exit 1
fi

OVERALL_STATUS="$("${PYTHON_BIN}" -c "import json, sys; print(json.load(open(sys.argv[1]))['overall_status'])" "${OUT_DIR}/stage1_summary.json")"
if [[ "${OVERALL_STATUS}" == "EQUAL" ]]; then
  echo "[T1-GATE] Verdict: PASS (Stage 1 bit-identical LLO + Mosaic across all subkernels; Stage 2 TPU runs skipped)"
  exit 0
fi

echo "[T1-GATE] Stage 1 detected intentional LLO/Mosaic changes (${OUT_DIR}/diffs)."
echo "[T1-GATE] Stage 2 on-TPU interleaved A/B required before landing intentional kernel/schedule changes."
exit 0
