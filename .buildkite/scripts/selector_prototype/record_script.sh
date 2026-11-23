#!/usr/bin/env bash
# PROTOTYPE recording step for a script-based CI step (an integration test):
# runs the step's own command with fnrec armed, so every Python process it
# starts (servers, engine workers, clients) records what it ran.
#
#   record_script.sh STEP COMMIT
#
# The command is looked up from STEP rather than passed in, and its output is
# not piped through tee: integration scripts kill -9 every process whose
# command line mentions python or vllm (except themselves and their parent)
# to free the TPU, and both would otherwise put "vllm" into ours.
#
# Runs inside the CI pod via run.sh, from the repo root. Not for merging.
set -uo pipefail

step="$1" commit="$2"

case "$step" in
  integration_compilation_cache_e2e)
    cmd=(bash ./scripts/vllm/integration/run_compilation_cache_e2e.sh) ;;
  integration_k3_pruned75_apc_offload_correctness)
    cmd=(bash ./scripts/vllm/integration/run_k3_pruned75_apc_offload_correctness.sh) ;;
  integration_qwen35_apc_correctness)
    cmd=(bash ./scripts/vllm/integration/run_qwen35_apc_correctness.sh) ;;
  integration_qwen35_pcp4_dp4_disagg_correctness)
    cmd=(bash ./scripts/vllm/integration/run_qwen35_p4d2_disagg_correctness.sh) ;;
  integration_qwen35_pcp4_2tp2dp_disagg_correctness)
    cmd=(bash ./scripts/vllm/integration/run_qwen35_pcp4_2tp2dp_disagg_correctness.sh) ;;
  integration_qwen35_pcp4tp1_tp4dcp2_disagg_correctness)
    cmd=(bash ./scripts/vllm/integration/run_qwen35_pcp_tp_dcp_disagg_correctness.sh) ;;
  integration_qwen35_gpc_global_dram_pool)
    cmd=(bash ./scripts/vllm/integration/run_qwen35_gpc_global_dram_pool.sh) ;;
  presubmit_disagg_single_host)
    cmd=(env TPU_KV_RESHARD_TRANSPORT=zmq
         LOG_DIR=perf_eval_results/disagg_single_host
         "PREFILL_EXTRA_ARGS=--max-num-seqs 8"
         "DECODE_EXTRA_ARGS=--max-num-batched-tokens 128 --max-num-seqs 128"
         bash ./examples/disagg/run_disagg_single_host.sh) ;;
  *)
    echo "record_script.sh: no command known for step '$step'" >&2
    exit 2 ;;
esac

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
"${cmd[@]}"
rc=$?
seconds=$(python3 -c "import time; print(round(time.time() - $start, 1))")

python3 - "$out/meta.json" "$step" "$commit" "$rc" "$seconds" "${cmd[*]}" <<'EOF'
import json, sys
path, step, commit, rc, seconds, command = sys.argv[1:]
json.dump({"step": step, "shard": 0, "shards": 1, "kind": "script",
           "commit": commit, "exit_code": int(rc), "seconds": float(seconds),
           "command": command}, open(path, "w"), indent=1)
EOF
echo "--- exit=$rc seconds=$seconds recordings=$(ls "$out" | grep -c '^fn\.')"
exit "$rc"
