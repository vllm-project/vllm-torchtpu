#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
scenario=${1:?Usage: serve.sh encoder_extend}
case "$scenario" in
    encoder_extend) sequences=1; budget=2048; buckets='[128,256,2048]' ;;
    *) echo "Unknown scenario: $scenario" >&2; exit 2 ;;
esac
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
# Auto-size KV for the cached contexts.
# The client rejects preempted or partially cached measurements.
export TP=1 DP=8 CHUNK=$budget MAX_SEQS=$sequences KV_BYTES=${KV_BYTES:-auto}
profile_args=()
if [[ -n "${ARTIFACT_LOCAL_DIR:-}" ]]; then
    profile_args+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$ARTIFACT_LOCAL_DIR/$scenario/profile\"}")
fi
exec bash "$here/../serve_cached_prefill.sh" \
    "${profile_args[@]}" \
    --worker-extension-cls server_step_worker.ServerStepWorker \
    --compilation-config "{\"compile_sizes\":$buckets}"
