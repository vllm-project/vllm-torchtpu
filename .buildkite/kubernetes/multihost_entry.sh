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

# Starts Ray across a slice and runs a command on its head.
#
#   multihost_entry.sh <command> [args...]
#
# Every host of the slice runs this with the same arguments and decides its own
# role; nothing outside the pods orchestrates them.
#
# Environment, from manifests/workloads/ray-multihost-slice.yaml:
#   HEAD_HOST      the DNS name of index 0, which every host addresses.
#   NUM_HOSTS      hosts in the slice; the head waits for this many Ray nodes.
#   JOB_COMPLETION_INDEX  set by Kubernetes. 0 is the head.
set -uo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

: "${HEAD_HOST:?HEAD_HOST must name index 0 of the slice}"
: "${NUM_HOSTS:?NUM_HOSTS must be set}"
INDEX="${JOB_COMPLETION_INDEX:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.buildkite/kubernetes/multihost_lib.sh
# shellcheck disable=SC1091
source "${HERE}/multihost_lib.sh"
pod_setup

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export TPU_MULTIHOST_BACKEND=ray
export JAX_PLATFORMS=""

RAY_PORT=6379

# How many Ray nodes the head can see, or 0 if it could not ask.
#
# The count comes back through a file: Ray's C++ logging floods stdout the
# moment a driver attaches, and the digit is lost in it.
#
# A node counts only once it advertises chips. A raylet registers as alive
# before libtpu has handed it its chips, so counting registrations lets the head
# start vLLM against a cluster with 0.0/16.0 TPU - which fails an hour later in
# a placement timeout that looks nothing like its cause.
alive_nodes() {
  local report="${TMPDIR:-/tmp}/ray_alive.$$"
  rm -f "$report"
  python3 -c "
import ray
ray.init(address='auto', logging_level='error')
n = sum(1 for n in ray.nodes()
        if n['Alive'] and n.get('Resources', {}).get('TPU', 0) > 0)
with open('$report', 'w') as f:
    f.write(str(n))
" >/dev/null 2>"$report.err"
  local alive
  alive=$(cat "$report" 2>/dev/null)
  rm -f "$report"
  echo "${alive:-0}"
}

# Why alive_nodes could not answer. The head is always its own TPU-bearing node,
# so a zero means the query failed, not that nobody has joined.
report_last_error() {
  local err="${TMPDIR:-/tmp}/ray_alive.$$.err"
  if [ -s "$err" ]; then
    echo "last ray query error:"
    tail -20 "$err"
  fi
  rm -f "$err"
}

run_worker() {
  # `ray start --address` fails outright against a head that has not come up.
  # A port probe rather than `ray status`, which cannot answer before this
  # worker has joined anything, nor after the head has gone.
  local deadline=$((SECONDS + 1800))
  until head_listening "$HEAD_HOST" "$RAY_PORT"; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "ERROR: no Ray head at ${HEAD_HOST}:${RAY_PORT} after 30 minutes"
      return 1
    fi
    sleep 10
  done

  # Not --block: nothing kills these pods, so a blocking worker would hold its
  # chips until the JobSet deadline and leave the step waiting on a JobSet that
  # can never complete.
  ray start --address="${HEAD_HOST}:${RAY_PORT}" || return 1
  echo "worker ${INDEX} joined ${HEAD_HOST}; waiting for the head to finish"
  wait_for_head_to_go "$HEAD_HOST" "$RAY_PORT"
  echo "worker ${INDEX}: head is gone; exiting"
  ray stop --force >/dev/null 2>&1 || true
}

run_head() {
  ray start --head --port="${RAY_PORT}" || return 1

  echo "--- Waiting for ${NUM_HOSTS} Ray nodes to be alive"
  local deadline=$((SECONDS + 1800)) alive reported=-1
  while :; do
    alive=$(alive_nodes)
    if [ "$alive" -ge "$NUM_HOSTS" ]; then
      echo "Ray cluster complete: ${alive} nodes."
      break
    fi
    if [ "$alive" != "$reported" ]; then
      echo "  ${alive}/${NUM_HOSTS} nodes alive"
      reported="$alive"
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "ERROR: Ray cluster incomplete after 30 minutes (alive=${alive}/${NUM_HOSTS})"
      report_last_error
      ray status || true
      return 1
    fi
    sleep 15
  done
  ray status || true

  echo "--- Running on the head: $*"
  "$@"
  local rc=$?

  return "$rc"
}

if [ "$INDEX" = "0" ]; then
  run_head "$@"
  rc=$?
  # A failed upload decides the step only when the work itself passed; a run
  # that already failed keeps its own exit code.
  if ! publish_results && [ "$rc" -eq 0 ]; then
    rc=1
  fi
else
  run_worker
  rc=$?
fi
exit "$rc"
