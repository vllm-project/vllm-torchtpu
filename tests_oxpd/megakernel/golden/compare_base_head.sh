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

# Compare CPU-interpret golden hashes between <base-ref> and the current tree
# on the current host (machine-independent base-vs-head verification).
#
# Usage:
#   tests/megakernel/golden/compare_base_head.sh <base-ref> [-n N] [pytest-args...]
#
# Note: <base-ref> must already include MK_GOLDEN_JSON support in
# tests/megakernel/test_golden_bitexact.py. Cases that exist only on one side
# (base-only or head-only) are reported, not failed.

set -euo pipefail

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 <base-ref> [-n N] [pytest-args...]" >&2
  echo "Note: <base-ref> must already support MK_GOLDEN_JSON in test_golden_bitexact.py." >&2
  exit 2
fi

BASE_REF="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ -n "${PYTEST:-}" ]]; then
  PYTEST_BIN="$PYTEST"
elif command -v pytest >/dev/null 2>&1; then
  PYTEST_BIN="$(command -v pytest)"
elif [[ -x "$HOME/mkvenv/bin/pytest" ]]; then
  PYTEST_BIN="$HOME/mkvenv/bin/pytest"
else
  echo "Error: pytest not found (set PYTEST=/path/to/pytest or activate venv)." >&2
  exit 2
fi

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$(dirname "$PYTEST_BIN")/python" ]]; then
  PYTHON_BIN="$(dirname "$PYTEST_BIN")/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  PYTHON_BIN="$(command -v python)"
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mk-golden-cmp.XXXXXX")"
WT_DIR="$TMP_DIR/base-worktree"
BASE_JSON="$TMP_DIR/base.json"
HEAD_XML="$TMP_DIR/head_junit.xml"

# shellcheck disable=SC2317
cleanup() {
  if [[ -d "$WT_DIR" ]]; then
    git -C "$REPO_ROOT" worktree remove --force "$WT_DIR" >/dev/null 2>&1 || rm -rf "$WT_DIR"
    git -C "$REPO_ROOT" worktree prune >/dev/null 2>&1 || true
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

BASE_SHA="$(git -C "$REPO_ROOT" rev-parse --verify "$BASE_REF")"
echo "=== [1/2] Generating base golden at $BASE_REF ($BASE_SHA) ==="
git -C "$REPO_ROOT" worktree add --detach "$WT_DIR" "$BASE_SHA" >/dev/null

BASE_TEST="$WT_DIR/tests/megakernel/test_golden_bitexact.py"
if [[ ! -f "$BASE_TEST" ]] || ! grep -q "MK_GOLDEN_JSON" "$BASE_TEST"; then
  echo "Error: <base-ref> ($BASE_REF) does not support MK_GOLDEN_JSON in tests/megakernel/test_golden_bitexact.py." >&2
  exit 2
fi

(
  cd "$WT_DIR"
  JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" \
    PYTHONPATH="$WT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    MK_REGEN_GOLDEN=1 \
    MK_GOLDEN_JSON="$BASE_JSON" \
    "$PYTEST_BIN" "$BASE_TEST" "$@" || [[ $? -eq 5 ]]
  if grep -q "ref_tiny_" "$REPO_ROOT/tests/megakernel/test_golden_bitexact.py" && ! grep -q "ref_tiny_" "$BASE_TEST"; then
    REF_ARGS=("$@")
    if [[ " $* " != *" -k "* ]]; then
      REF_ARGS+=("-k" "ref_tiny_")
    fi
    if [[ " ${REF_ARGS[*]} " == *"ref_tiny_"* ]]; then
      JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" \
        PYTHONPATH="$WT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
        MK_REGEN_GOLDEN=1 \
        MK_GOLDEN_JSON="$BASE_JSON" \
        "$PYTEST_BIN" "$REPO_ROOT/tests/megakernel/test_golden_bitexact.py" "${REF_ARGS[@]}" || [[ $? -eq 5 ]]
    fi
  fi
)

echo "=== [2/2] Comparing current tree against $BASE_REF ($BASE_SHA) ==="
CMP_RC=0
(
  cd "$REPO_ROOT"
  JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" \
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    MK_GOLDEN_JSON="$BASE_JSON" \
    "$PYTEST_BIN" "$REPO_ROOT/tests/megakernel/test_golden_bitexact.py" \
    -v --junitxml="$HEAD_XML" "$@"
) || CMP_RC=$?

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" - "$BASE_JSON" "$HEAD_XML" "$REPO_ROOT" <<'PY'
import json
import pathlib
import sys
import xml.etree.ElementTree as ET

base_json = pathlib.Path(sys.argv[1])
head_xml = pathlib.Path(sys.argv[2])
repo_root = pathlib.Path(sys.argv[3])
sys.path.insert(0, str(repo_root))

from tests.megakernel import test_golden_bitexact as tgb  # noqa: E402

base_data = json.loads(base_json.read_text())
base_cases = set(base_data) - {tgb._META_KEY}
head_cases = set(tgb.CASES)

failed_cases: set[str] = set()
if head_xml.exists():
    root = ET.parse(head_xml).getroot()
    for tc in root.iter("testcase"):
        name = tc.attrib.get("name", "")
        if name.startswith("test_golden_bitexact[") and name.endswith("]"):
            case_name = name[len("test_golden_bitexact[") : -1]
            if tc.find("failure") is not None or tc.find("error") is not None:
                failed_cases.add(case_name)

print("\n=== Per-case base-vs-head summary ===")
for case in sorted(base_cases | head_cases):
    if case not in head_cases:
        status = "BASE_ONLY (reported, not failed)"
    elif case not in base_cases:
        status = "HEAD_ONLY (reported, not failed)"
    elif case in failed_cases:
        status = "MISMATCH"
    else:
        status = "EQUAL"
    print(f"  {case:24s} {status}")
PY

exit "$CMP_RC"
