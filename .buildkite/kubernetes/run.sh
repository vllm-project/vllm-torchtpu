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

# Runs one command in a single pod: on one TPU host, or with SHAPE=cpu on the
# fleet's cpu queue, for work that needs the test environment and no chip.
#
#   SHAPE=<machine-type>/<topology> .buildkite/kubernetes/run.sh <command> [args...]
#   SHAPE=cpu                       .buildkite/kubernetes/run.sh <command> [args...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: SHAPE=<machine-type>/<topology>|cpu $0 <command> [args...]" >&2
  exit 2
fi

# The launcher's own machine-type and topology names, kept as one token so a
# step cannot change half of the pair.
# From the step's environment, not an argument: every step already sets SHAPE
# from one of the shape anchors, and passed the same value straight back in.
shape="${SHAPE:-}"
if [[ "$shape" == "cpu" ]]; then
  hardware=(--cpu)
else
  machine_type="${shape%%/*}"
  topology="${shape#*/}"
  if [[ -z "$shape" || "$machine_type" == "$shape" || -z "$topology" ]]; then
    echo "$0: SHAPE must be <machine-type>/<topology> or cpu, got '${shape}'" >&2
    exit 2
  fi
  hardware=(--machine-type "$machine_type" --topology "$topology")
fi

# Exports WORKLOAD_IMAGE and fills `env_args`.
# shellcheck source=.buildkite/kubernetes/common.sh
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# A step that wants artifacts out of the pod names a directory in ARTIFACTS_DIR
# and writes into it; everything under it is uploaded, and the relative path is
# the artifact name. The pod is deleted when the step ends, so it uploads its
# own. Unset means the step produces nothing to keep - most do not.
#
# Single-quoted on purpose: this is a program for the pod shell, not for this
# one, so $@ and $? have to arrive unexpanded.
# shellcheck disable=SC2016
IN_POD='
  set -o pipefail
  [ -n "${ARTIFACTS_DIR:-}" ] && mkdir -p "$ARTIFACTS_DIR"
  "$@"
  rc=$?
  if [ -n "${ARTIFACTS_DIR:-}" ] && [ -n "$(ls -A "$ARTIFACTS_DIR" 2>/dev/null)" ]; then
    # A rig that points a `latest_*` symlink at a run directory leaves it
    # dangling once that directory is gone, and the upload globs it, opens it
    # and fails the step over a file the test never meant to keep. The three
    # bare-metal runners each drop these the same way before uploading.
    find "$ARTIFACTS_DIR" -type l ! -exec test -e {} \; -delete 2>/dev/null || true
    # A lost upload is indistinguishable from a step that produced nothing, so
    # it fails the step - but never masks a failure from the command itself.
    if ! buildkite-agent artifact upload "$ARTIFACTS_DIR/**/*"; then
      echo "ERROR: artifacts were produced but could not be uploaded" >&2
      [ "$rc" -eq 0 ] && rc=1
    fi
  fi
  exit $rc
'

# shellcheck disable=SC2154  # env_args comes from common.sh
exec /opt/launcher/launch \
  "${hardware[@]}" \
  "${env_args[@]}" \
  -- bash -c "$IN_POD" -- "$@"
