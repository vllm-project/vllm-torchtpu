#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
ulimit -c 0

scenario="${1:?scenario is required: encoder_extend or decode}"
model_uri="${2:?model GCS URI is required}"
results="${3:?results directory is required}"
case "$scenario" in
  encoder_extend|decode) ;;
  *) echo "Unsupported DP8 benchmark scenario: $scenario" >&2; exit 2 ;;
esac
if [[ "$model_uri" != gs://*/* ]]; then
  echo "Expected gs://bucket/model-prefix, got: $model_uri" >&2
  exit 2
fi

export TMPDIR=/tmp/tpu_logs/tmp
export PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache UV_CACHE_DIR=/tmp/tpu_logs/uv-cache
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
# The shared CI image preloads tcmalloc; heap trimming requires glibc malloc.
export TPU_RECLAIM_COMPILE_MEMORY=0
mkdir -p "$TMPDIR" "$results" /local_hf_cache/dsv41-ci
exec > >(tee "$results/task.log") 2>&1
scratch=$(mktemp -d "$TMPDIR/dsv41-dp8.XXXXXX")
cache_dir=$(mktemp -d /local_hf_cache/dsv41-ci/"$scenario".XXXXXX)
model_mount="$scratch/model"
mkdir -p "$model_mount"
cleanup() {
  if mountpoint -q "$model_mount"; then
    umount "$model_mount" || true
  fi
  rm -rf "$cache_dir"
}
trap cleanup EXIT

uv pip freeze --system > "$results/pip-freeze.txt"

# Mount weights instead of copying the checkpoint onto the CI agent's disk.
model_path="${model_uri#gs://}"
bucket="${model_path%%/*}"
prefix="${model_path#*/}"
gcsfuse --implicit-dirs --only-dir "$prefix" -o ro "$bucket" "$model_mount"
export MODEL_PATH="$model_mount"
test -f "$MODEL_PATH/config.json"

curl --fail --location --retry 3 \
  https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/main/data.json \
  --output "$scratch/longbench.json"
# prepare checks the dataset SHA and retains the scenario's frozen selection.
python scripts/dsv41_bench/server_steps/bench_cached_extend.py prepare \
  --scenario "$scenario" --dataset "$scratch/longbench.json" \
  --tokenizer "$MODEL_PATH" --out "$results/inputs.json"
sha256sum "$results/inputs.json" > "$results/input.sha256"

python scripts/dsv41_bench/server_steps/run.py \
  --scenario "$scenario" --input "$results/inputs.json" \
  --out "$results/benchmark" --cache-dir "$cache_dir" --cache-mode cold \
  --startup-timeout 7200

# Publish the existing summary verbatim, including its token accounting and
# full-DP checks. Profiling stays separate from the untraced CI measurements.
python - "$results/benchmark/fixed" <<'PY_SUMMARY'
import json
import sys
from pathlib import Path

summaries = sorted(Path(sys.argv[1]).glob("round-*-summary.json"))
if not summaries:
    raise RuntimeError("DP8 benchmark completed without a formal round summary")
for path in summaries:
    print(f"DSV41_DP8_BENCHMARK_SUMMARY {path.name}")
    print(json.dumps(json.loads(path.read_text()), indent=2))
PY_SUMMARY
