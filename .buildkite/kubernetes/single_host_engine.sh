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

# Runs an engine that owns one host, on a fleet that assumes it might not.
#
#   single_host_engine.sh <command> [args...]
#
# GKE gives every TPU pod the environment of a slice member - a coordinator
# address, a worker hostname list, the megascale ports - because a pod is
# normally one host of a slice that has several. An engine that reads those
# tries to rendezvous with peers, and for a workload that is the whole slice
# there are none: the ranks reach the distributed barrier and stop there, with
# the model loaded and the process idle, which reads as a slow boot rather than
# a deadlock.
#
# A test written for a VM never has to think about this, because a VM has none
# of these set. So the pod unsets them on its behalf, and states the shape it
# actually has. disagg_engine.sh does the same thing for the 1P1D benchmark;
# this is that list, in a form the manifests can share.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

# Emptied rather than unset: the runtime treats an empty coordinator as "no
# coordinator", where an unset one can fall back to a default it infers.
export MEGASCALE_COORDINATOR_ADDRESS=''
export MEGASCALE_NUM_SLICES=''
export MEGASCALE_PORT=''
export MEGASCALE_SLICE_ID=''
export TPU_PROCESS_ADDRESSES=''
export TPU_PROCESS_PORT=''
export TPU_WORKER_HOSTNAMES=''

# One host, four chips. TPU_WORKER_ID is already 0 and stays that way.
export TPU_HOST_BOUNDS=1,1,1
export TPU_TOPOLOGY=2x2x1

# The engine this pod starts writes to a file rather than to stdout, because the
# test backgrounds it and reads the file back when it has to report a failure.
# That is fine on a VM, where the file outlives the run. Here the pod is deleted
# with it, so STREAM_LOG names the one this role writes and it is copied to the
# step's output as it is written.
#
# Only for the case the test cannot report itself: a server that dies gets its
# last hundred lines dumped by the test, but one that never finishes starting
# gets nothing, and that is the failure this fleet produced twice.
#
# -F, not -f: the file does not exist yet, and the engine creates it a minute or
# so from now.
if [[ -n "${STREAM_LOG:-}" ]]; then
  mkdir -p "$(dirname "${STREAM_LOG}")"
  tail -F "${STREAM_LOG}" 2>/dev/null &
fi

exec "$@"
