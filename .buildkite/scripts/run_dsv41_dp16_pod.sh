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

#
#   run_dsv41_dp16_pod.sh <gs://bucket/model-prefix> <results-dir>
#
# One pod of the DP16 decode benchmark, from
# .buildkite/kubernetes/manifests/workloads/dsv41-dp16-decode.yaml. Both pods
# run this with the same arguments; JOB_COMPLETION_INDEX picks the role.
#
# Each runs run_dsv41_benchmark.sh decode_dp16 for its own host under
# coordinate_multihost.py, which gates the launch and the shutdown between
# them over HTTP on the head and carries the worker's logs to it. The head
# (index 0) then audits both hosts, publishes the GitHub check and uploads the
# artifacts; the worker uploads nothing of its own.
#
# Environment, from the manifest:
#   HEAD_HOST            DNS name of index 0.
#   JOB_COMPLETION_INDEX this pod's index: 0 head, 1 worker.
#   ARTIFACTS_DIR        what the head uploads; <results-dir> lies under it.
set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <gs://bucket/model-prefix> <results-dir>" >&2
  exit 2
fi
model_uri="$1"
results="$2"
: "${HEAD_HOST:?HEAD_HOST must name index 0 of the slice}"
index="${JOB_COMPLETION_INDEX:?JOB_COMPLETION_INDEX must be set}"

scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${scripts}/../.." && pwd)"
# shellcheck source=.buildkite/kubernetes/multihost_lib.sh
# shellcheck disable=SC1091
source "${REPO_DIR}/.buildkite/kubernetes/multihost_lib.sh"
pod_setup

# An address, not a name: run.py binds the DP master on the head at this
# address, and both hosts must be handed the same value.
head_ip="$(resolve "$HEAD_HOST" 600)" || exit 1
export DSV41_HEAD_ADDRESS="$head_ip" DSV41_HOST_RANK="$index"
echo "--- DP16 host ${index}: head ${HEAD_HOST} at ${head_ip}"

bench=(bash "${scripts}/run_dsv41_benchmark.sh" decode_dp16 "$model_uri" "$results")
coordinator=(python3 "${REPO_DIR}/scripts/dsv41_bench/server_steps/coordinate_multihost.py")

case "$index" in
  0)
    status=0
    bash "${scripts}/dsv41_publish_check.sh" "$results/benchmark" \
      "${coordinator[@]}" head --results "$results" -- "${bench[@]}" || status=$?
    # A failed upload decides the step only when the work itself passed.
    if ! publish_results && [[ "$status" == 0 ]]; then
      status=1
    fi
    exit "$status"
    ;;
  1)
    exec "${coordinator[@]}" worker --results "$results" --head "$head_ip" \
      -- "${bench[@]}"
    ;;
  *)
    echo "ERROR: DP16 decode runs on two hosts, not index ${index}" >&2
    exit 2
    ;;
esac
