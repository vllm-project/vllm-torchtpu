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
mkdir -p "$TMPDIR" "$results"
exec > >(tee "$results/task.log") 2>&1
scratch=$(mktemp -d "$TMPDIR/dsv41-dp8.XXXXXX")
model_mount="$scratch/model"
mkdir -p "$model_mount"
cleanup() {
  if mountpoint -q "$model_mount"; then
    umount "$model_mount" || true
  fi
}
trap cleanup EXIT

uv pip freeze --system > "$results/pip-freeze.txt"

# Each pytest process releases the TPU before the serving process starts.
if [[ "$scenario" == decode ]]; then
  python -m pytest -q tests/kernels/test_dsv41_decode_index_scores.py \
    -k 'production_shapes or selection_is_identical' \
    --junitxml="$results/kernel-scores.xml"
  python -m pytest -q tests/models/vllm/test_dsv41_decode_indexer_route.py \
    -k 'candidate_cache_matches_existing_with_fallback or basic_config or long_candidate_cache' \
    --junitxml="$results/kernel-routes.xml"
  python -m pytest -q tests/kernels/fused_moe/test_fused_ep_moe_v2.py \
    -k row_padded_flat_scales --junitxml="$results/kernel-scales.xml"
fi

# Input preparation uses the mount; serving streams weights from the GCS URI.
model_path="${model_uri#gs://}"
bucket="${model_path%%/*}"
prefix="${model_path#*/}"
gcsfuse --implicit-dirs --only-dir "$prefix" -o ro "$bucket" "$model_mount"
export MODEL_PATH="$model_mount" MODEL_URI="$model_uri"
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
  --out "$results/benchmark" \
  --startup-timeout 7200

# Raw observations stay in artifacts; print the throughput metrics in the log.
python - "$results/benchmark/fixed" <<'PY_SUMMARY'
import json
import sys
from pathlib import Path

summaries = sorted(Path(sys.argv[1]).glob("round-*-summary.json"))
if not summaries:
    raise RuntimeError("DP8 benchmark completed without a formal round summary")
for path in summaries:
    report = json.loads(path.read_text())
    metrics = {}
    if "decode_throughput" in report:
        metrics["decode_throughput"] = report["decode_throughput"]
    metrics["categories"] = {
        category: {
            key: value
            for key, value in values.items()
            if key not in ("samples", "first_20", "last_20")
        }
        for category, values in report["categories"].items()
    }
    print(f"DSV41_DP8_BENCHMARK_SUMMARY {path.name}")
    print(json.dumps(metrics, indent=2))
PY_SUMMARY
