#!/usr/bin/env bash
# PROTOTYPE recording step for a script-based CI step (an integration test):
# runs the step's own command with fnrec armed, so every Python process it
# starts (servers, engine workers, clients) records what it ran.
#
#   record_script.sh STEP COMMIT COMMAND [ARGS...]
#
# Runs inside the CI pod via run.sh, from the repo root. Not for merging.
set -uo pipefail

step="$1" commit="$2"
shift 2

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
recorder="$here/../fnrec_experiment"
out="${ARTIFACTS_DIR:-selector_recordings}/fnrec/$step/shard0"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

echo "--- Recording $step at ${commit:0:10}"
export FNREC_PACKAGE=vllm_torchtpu
export FNREC_OUT="$out"
export FNREC_TESTS_ROOT="$PWD/tests"
FNREC_ROOT="$(python3 "$recorder/install.py")"
export FNREC_ROOT
echo "FNREC_ROOT=$FNREC_ROOT"

start=$(python3 -c "import time; print(time.time())")
"$@" 2>&1 | tee "$out/run.log"
rc=${PIPESTATUS[0]}
seconds=$(python3 -c "import time; print(round(time.time() - $start, 1))")

python3 - "$out/meta.json" "$step" "$commit" "$rc" "$seconds" "$*" <<'EOF'
import json, sys
path, step, commit, rc, seconds, command = sys.argv[1:]
json.dump({"step": step, "shard": 0, "shards": 1, "kind": "script",
           "commit": commit, "exit_code": int(rc), "seconds": float(seconds),
           "command": command}, open(path, "w"), indent=1)
EOF
echo "--- exit=$rc seconds=$seconds recordings=$(ls "$out" | grep -c '^fn\.')"
exit "$rc"
