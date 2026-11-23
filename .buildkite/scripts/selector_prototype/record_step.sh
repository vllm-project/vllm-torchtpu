#!/usr/bin/env bash
# PROTOTYPE recording step. Runs one shard of a pytest step with fnrec, the
# function recorder adapted from vllm-project/ci-infra, and leaves the
# recordings under $ARTIFACTS_DIR for the pod to upload.
#
#   record_step.sh STEP SHARD SHARDS COMMIT MARKER [EXCLUDED_FILE...]
#
# Runs inside the CI pod via run.sh, from the repo root. Not for merging.
set -uo pipefail

step="$1" shard="$2" shards="$3" commit="$4" marker="$5"
shift 5

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
recorder="$here/../fnrec_experiment"
out="${ARTIFACTS_DIR:-selector_recordings}/fnrec/$step/shard$shard"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

echo "--- Recording $step, shard $shard of $shards, at ${commit:0:10}"
export FNREC_PACKAGE=vllm_torchtpu
export FNREC_OUT="$out"
export FNREC_TESTS_ROOT="$PWD/tests"
FNREC_ROOT="$(python3 "$recorder/install.py")"
export FNREC_ROOT
echo "FNREC_ROOT=$FNREC_ROOT"

# The same files CI's step would collect, dealt round-robin across shards.
mapfile -t files < <(python3 "$here/shard_files.py" "$shard" "$shards" "$@")
echo "${#files[@]} test files in this shard"

start=$(python3 -c "import time; print(time.time())")
python3 -m pytest -q -p no:cacheprovider --timeout=600 -m "$marker" \
  "${files[@]}" >"$out/pytest.log" 2>&1
rc=$?
seconds=$(python3 -c "import time; print(round(time.time() - $start, 1))")
tail -n 5 "$out/pytest.log"

python3 - "$out/meta.json" "$step" "$shard" "$shards" "$commit" "$rc" "$seconds" "$marker" <<'EOF'
import json, sys
path, step, shard, shards, commit, rc, seconds, marker = sys.argv[1:]
json.dump({"step": step, "shard": int(shard), "shards": int(shards),
           "commit": commit, "exit_code": int(rc), "seconds": float(seconds),
           "marker": marker}, open(path, "w"), indent=1)
EOF
echo "exit=$rc seconds=$seconds files=$(ls "$out" | wc -l)"

# 5 is pytest's "no tests collected", which a shard can reach once the marker
# filter empties it.
[ "$rc" -eq 0 ] || [ "$rc" -eq 5 ]
