#!/usr/bin/env bash
# PROTOTYPE recording wrapper for one pod of a multi-pod (JobSet) CI step.
# The manifests on this branch exec it in place of their own command, so every
# pod of the workload records, and each pod uploads its own recordings: these
# workloads do not go through run.sh, which is what uploads for the rest.
#
#   record_pod.sh MODE
#
# MODE names both the step and the pod's role; the command is looked up here
# rather than passed in, so this process's command line never mentions python
# or vllm (the integration scripts kill -9 every process that does, except
# themselves and their parent). Not for merging.
set -uo pipefail

mode="$1"
judged=1  # whether this pod's exit code says anything about the step
case "$mode" in
  gpc_head)
    step=integration_qwen35_gpc_correctness shard=0 shards=2
    cmd=(bash .buildkite/kubernetes/single_host_engine.sh
         bash ./scripts/vllm/integration/run_qwen35_gpc_correctness.sh) ;;
  gpc_worker)
    step=integration_qwen35_gpc_correctness shard=1 shards=2 judged=0
    cmd=(bash .buildkite/kubernetes/single_host_engine.sh
         bash ./scripts/vllm/integration/run_qwen35_gpc_correctness.sh) ;;
  p8d8_head)
    step=integration_qwen35_pcp8_2tp4dp_disagg_correctness shard=0 shards=2
    cmd=(bash .buildkite/kubernetes/single_host_engine.sh
         bash ./scripts/vllm/integration/run_qwen35_p8d8_disagg_correctness.sh) ;;
  p8d8_decode)
    step=integration_qwen35_pcp8_2tp4dp_disagg_correctness shard=1 shards=2 judged=0
    cmd=(bash .buildkite/kubernetes/single_host_engine.sh
         bash ./scripts/vllm/integration/run_qwen35_p8d8_disagg_correctness.sh) ;;
  ray)
    # One host of a Ray slice; index 0 is the head, the others are torn down
    # once it succeeds.
    step=vllm_torchtpu_multihost_presubmit
    shard="${JOB_COMPLETION_INDEX:-0}" shards="${NUM_HOSTS:-1}"
    [ "$shard" = 0 ] || judged=0
    readarray -d '' args < <(printf '%s' "${MULTIHOST_ARGS_B64}" | base64 -d)
    cmd=(bash .buildkite/kubernetes/multihost_entry.sh "${args[@]}") ;;
  *)
    echo "record_pod.sh: unknown mode '$mode'" >&2
    exit 2 ;;
esac

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rel="selector_recordings/fnrec/$step/shard$shard"
mkdir -p "$rel"
out="$(cd "$rel" && pwd)"
commit="${BUILDKITE_COMMIT:-}"

echo "--- Recording $step ($mode, shard $shard of $shards) at ${commit:0:10}"
export FNREC_PACKAGE=vllm_torchtpu
export FNREC_OUT="$out"
export FNREC_TESTS_ROOT="$PWD/tests"
FNREC_ROOT="$(python3 "$here/../fnrec_experiment/install.py")"
export FNREC_ROOT

start=$(date +%s)
finish() {
  local rc="$1"
  printf '{"step": "%s", "shard": %s, "shards": %s, "kind": "script", "commit": "%s", "exit_code": %s, "judged": %s, "seconds": %s, "role": "%s"}\n' \
    "$step" "$shard" "$shards" "$commit" "$rc" "$judged" "$(( $(date +%s) - start ))" "$mode" \
    >"$out/meta.json"
  echo "--- $mode exit=$rc recordings=$(ls "$out" | grep -c '^fn\.')"
  if command -v buildkite-agent >/dev/null; then
    buildkite-agent artifact upload "$rel/*" \
      || echo "WARNING: could not upload this pod's recordings" >&2
  else
    echo "WARNING: no buildkite-agent in this pod; recordings not uploaded" >&2
  fi
  exit "$rc"
}

# Backgrounded and waited on, so a SIGTERM (JobSet tearing a peer down once the
# head is done) runs the trap at once and the recordings still get uploaded.
"${cmd[@]}" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null; wait "$child"; finish 143' TERM
wait "$child"
finish "$?"
