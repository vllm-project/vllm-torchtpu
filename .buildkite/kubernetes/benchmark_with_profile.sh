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

# Runs a benchmark config and ships its xprof traces from inside the pod.
#
#   benchmark_with_profile.sh <config> [run_eval_flow args...]
#
# Profiling used to be a second step against a separate `<config>-profile`,
# until #1061 folded it into the same server: the config now carries
# CAPTURE_PROFILE and PROFILE_GCS_BASE, and one run produces both the
# benchmark numbers and the traces. That change rewrote pipeline_perf.yml and
# deleted the -profile configs; this is the same shape for the pods.
#
# The upload cannot be a second command the way it is on an agent. The traces
# are written in the pod, and the pod is deleted when the step's command
# returns - so the copy to GCS has to happen before that, here.
set -uo pipefail

CONFIG=${1:?usage: benchmark_with_profile.sh <config> [run_eval_flow args...]}
shift

# Mon/Wed/Fri, as on bare metal: profiling costs about +11% TTFT, so a nightly
# that captured every run would report a benchmark it had slowed down. A step
# that sets CAPTURE_PROFILE itself wins, which is how a one-off build asks for
# traces on a Tuesday.
if [[ -z "${CAPTURE_PROFILE:-}" ]]; then
  case "$(LC_ALL=C TZ=America/Los_Angeles date +%a)" in
    Mon|Wed|Fri) CAPTURE_PROFILE=1 ;;
    *)           CAPTURE_PROFILE=0 ;;
  esac
fi
export CAPTURE_PROFILE

# The same path run_eval_flow.sh would pick, stated because upload_profile.sh
# has to be handed it.
RESULTS_DIR="${ARTIFACTS_DIR:-/tmp/perf_eval}/${CONFIG}"

rc=0
bash ./scripts/vllm/benchmarking/run_eval_flow.sh \
  --config "$CONFIG" --results-dir "$RESULTS_DIR" "$@" || rc=$?

if [[ "$CAPTURE_PROFILE" == "1" ]]; then
  # A lost upload fails the step, but never masks the benchmark's own failure:
  # traces are the whole output of a capture run, and a run that already failed
  # keeps the exit code that says why.
  bash .buildkite/scripts/upload_profile.sh "$RESULTS_DIR" \
    || { [[ "$rc" -ne 0 ]] || rc=1; }
fi

exit "$rc"
