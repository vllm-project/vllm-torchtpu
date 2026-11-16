#!/usr/bin/env bash
# Experiment: does upstream vLLM's function recorder (fnrec) work on
# vllm-torchtpu? Runs a few CPU unit tests twice, without and then with the
# recorder, and prints a summary. Runs inside the CI pod via run.sh, from the
# repo root (/workspace/vllm-torchtpu). Not for merging.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${ARTIFACTS_DIR:-fnrec_experiment}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

# CPU tests that import vllm_torchtpu. The worker test starts a subprocess,
# which checks that child processes are recorded too.
tests=(
  tests/megakernel/core/test_planner.py
  tests/megakernel/core/test_pool_layout.py
  tests/test_envs.py
  tests/test_env_validation.py
  "tests/worker/test_tpu_worker.py::test_real_patch_entrypoints_in_spawned_processes"
  "tests/worker/test_tpu_worker.py::TestProfilerDir"
)
pytest_args=(-q -p no:cacheprovider --timeout=600
  -m "not tpu_test and not multichip and not nightly")

echo "--- Environment"
python3 --version
python3 -c "import sys; print('sys.monitoring available:', hasattr(sys, 'monitoring'))"

echo "--- 1. Baseline run (no recorder)"
start=$(python3 -c "import time; print(time.time())")
python3 -m pytest "${pytest_args[@]}" "${tests[@]}" >"$out/baseline.log" 2>&1
baseline_rc=$?
baseline_s=$(python3 -c "import time; print(round(time.time() - $start, 1))")
tail -n 3 "$out/baseline.log"
echo "exit=$baseline_rc seconds=$baseline_s"

echo "--- 2. Install the recorder (this pod only)"
export FNREC_PACKAGE=vllm_torchtpu
export FNREC_OUT="$out/fnrec"
export FNREC_TESTS_ROOT="$PWD/tests"
mkdir -p "$FNREC_OUT"
FNREC_ROOT="$(python3 "$here/install.py")"
export FNREC_ROOT
echo "FNREC_ROOT=$FNREC_ROOT"

echo "--- 3. Recorded run"
start=$(python3 -c "import time; print(time.time())")
python3 -m pytest "${pytest_args[@]}" "${tests[@]}" >"$out/recorded.log" 2>&1
recorded_rc=$?
recorded_s=$(python3 -c "import time; print(round(time.time() - $start, 1))")
tail -n 3 "$out/recorded.log"
echo "exit=$recorded_rc seconds=$recorded_s"

echo "--- 4. Summary"
python3 "$here/summarize.py" "$FNREC_OUT" \
  --baseline-seconds "$baseline_s" --baseline-exit "$baseline_rc" \
  --recorded-seconds "$recorded_s" --recorded-exit "$recorded_rc" \
  --json "$out/summary.json" | tee "$out/summary.txt"

# Pass the step only if both test runs passed, so a red step means
# "look at the logs", not "the recorder failed".
[ "$baseline_rc" -eq 0 ] && [ "$recorded_rc" -eq 0 ]
