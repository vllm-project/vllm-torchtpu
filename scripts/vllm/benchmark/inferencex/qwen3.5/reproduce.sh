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

# The Qwen3.5-397B-A17B-FP8 InferenceX sweep (8,192 in / 1,024 out) on one
# TPU7x host from nothing: a checkout, a virtual environment with the
# repository's pinned install, InferenceX's client, the sweep, a table.
#
#   bash reproduce.sh              from inside a checkout: uses it
#   bash reproduce.sh WORK_DIR     clones REPO at REF into WORK_DIR first
#
# Needs: python3.12, gcloud logged in with access to the Torch TPU registry
# (README.md), and the model under HF_HOME (or MODEL=gs://... for streaming).
# INFERENCEX_REPO optionally selects an existing clean checkout of the pinned client.
# Work-directory mode also requires REF as a full release commit SHA.
# Optional: REPO, CONCS (default: every point), RESULT_DIR, VENV, HF_HOME, MODEL.
set -euo pipefail

REPO="${REPO:-https://github.com/vllm-project/vllm-torchtpu.git}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -n "${1:-}" ]; then
  if [[ ! "${REF:-}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: set REF to the full release commit SHA for work-directory mode" >&2
    exit 1
  fi
  # A work directory: check out the exact release commit in a clean clone.
  WORK="$(mkdir -p "$1" && cd "$1" && pwd)"
  ROOT="$WORK/vllm-torchtpu"
  if [ ! -d "$ROOT/.git" ]; then
    git clone "$REPO" "$ROOT"
  elif [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
    echo "ERROR: $ROOT has local changes; move them aside or use another work directory" >&2
    exit 1
  fi
  git -C "$ROOT" fetch -q "$REPO" "$REF"
  git -C "$ROOT" checkout -q --detach FETCH_HEAD
  VENV="${VENV:-$WORK/venv}"
else
  ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
  VENV="${VENV:-$ROOT/.venv}"
fi
RECIPE="$ROOT/scripts/vllm/benchmark/inferencex/qwen3.5"
if [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
  echo "ERROR: $ROOT has local changes; use a clean release checkout" >&2
  exit 1
fi
if [ -n "${CLIENT:-}" ]; then
  echo "ERROR: reproduce.sh uses bench.sh; unset CLIENT to reproduce the serving sweep" >&2
  exit 1
fi
source "$RECIPE/check_client.sh"
prepare_inferencex_client || exit 1
echo "--- tree: $(git -C "$ROOT" rev-parse HEAD) ($(git -C "$ROOT" status --porcelain | wc -l) local changes) ---"

RESULT_DIR="${RESULT_DIR:-$(mktemp -d /tmp/qwen3.5-inferencex-repro_XXXXXX)}"
mkdir -p "$RESULT_DIR"
if [ -n "$(find "$RESULT_DIR" -mindepth 1 -print -quit)" ]; then
  echo "ERROR: RESULT_DIR must be empty so the table contains only this sweep" >&2
  exit 1
fi

# The install as README.md describes it: vLLM built for no device on top of
# the Torch TPU registry's wheels, with PyPI for the rest. The package is
# installed in place, so the checkout's code is what runs; the install runs
# on every invocation and is a no-op when the pins are already satisfied.
if [ ! -x "$VENV/bin/python" ]; then
  python3.12 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
fi
echo "--- installing $ROOT into $VENV ---"
streaming_deps=()
if [[ "${MODEL:-}" == gs://* || "${LOAD_FORMAT:-}" == runai_streamer ]]; then
  streaming_deps+=("runai-model-streamer[gcs]")
fi
PIP_INDEX_URL="https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" \
  PIP_EXTRA_INDEX_URL="https://pypi.org/simple" VLLM_TARGET_DEVICE="empty" \
  "$VENV/bin/pip" install --quiet --pre -e "$ROOT" "${streaming_deps[@]}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "--- $(vllm --version 2>/dev/null | tail -1); $(pip list 2>/dev/null | grep -iE '^(torch-tpu|torch_tpu|libtpu|jax) ' | tr -s ' ' | tr '\n' ';') ---"

export RESULT_DIR
export CONCS="${CONCS:-4 8 16 32 64 128 256 512 1024}"
echo "--- sweep: CONCS=\"$CONCS\" RESULT_DIR=$RESULT_DIR ---"
bash "$RECIPE/sweep.sh"

# The table: total tokens per second per chip (four chips), the client's
# median first-token time and median time per output token (the
# interactivity is 1000 over the median), and the run's provenance beside
# the client files.
python3 - "$RESULT_DIR" "$ROOT" <<'PY'
import glob, json, os, subprocess, sys
result_dir, root = sys.argv[1:3]
rows = []
for f in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
    d = json.load(open(f))
    if "max_concurrency" not in d:
        continue
    rows.append((d["max_concurrency"], d["total_token_throughput"] / 4,
                 d["median_ttft_ms"] / 1000, d["median_tpot_ms"],
                 1000 / d["median_tpot_ms"], d["completed"], d["num_prompts"]))
lines = ["| users | tokens/s per chip | median TTFT (s) | median TPOT (ms) | tokens/s per user | requests |",
         "|---:|---:|---:|---:|---:|---:|"]
for c, t, ttft, tpot, itl, done, n in sorted(rows):
    lines.append(f"| {c} | {t:,.0f} | {ttft:.2f} | {tpot:.1f} | {itl:.1f} | {done}/{n} |")
table = "\n".join(lines)
print(table)
git = lambda *a: subprocess.run(["git", "-C", root, *a], capture_output=True, text=True).stdout.strip()
infx = os.environ["INFERENCEX_REPO"]
pins = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
json.dump({"commit": git("rev-parse", "HEAD"), "local_changes": git("status", "--porcelain"),
           "inferencex_commit": subprocess.run(["git", "-C", infx, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "inferencex_expected_commit": os.environ["INFERENCEX_COMMIT"],
           "inferencex_local_changes": subprocess.run(["git", "-C", infx, "status", "--porcelain"], capture_output=True, text=True).stdout.strip(),
           "pins": pins,
           "table": table}, open(os.path.join(result_dir, "run.json"), "w"), indent=1)
open(os.path.join(result_dir, "table.md"), "w").write(table + "\n")
PY
echo "--- results: $RESULT_DIR ---"
