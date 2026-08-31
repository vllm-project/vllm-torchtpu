#!/bin/bash
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

# Reset a persistent results directory to an empty, world-writable state.
#
# Usage: reset_results_dir.sh <dir> [docker-image]
#        ssh host "bash -s" -- <dir> < reset_results_dir.sh
#
# These directories are bind-mounted into test containers that run as root. A
# container that forgets `umask 000` leaves behind 0755 root-owned directories,
# and the buildkite-agent user then cannot unlink anything inside them. A plain
# `rm -rf` fails with "Permission denied", and under `set -e` that kills the job
# before a single test runs. Worse, nothing ever removes the leftovers, so every
# later job on that agent dies the same way -- the host stays wedged until
# someone cleans it by hand. That is exactly how one unmerged multi-host disagg
# branch took out both v7x-16 agents for most of a day.
#
# So escalate instead of giving up: agent rm, sudo rm, root container rm, and
# finally rename the directory out of the way (a rename only needs write access
# on the *parent*, which the agent always has). Only a failed rename is fatal.

set -uo pipefail

TARGET="${1:?usage: reset_results_dir.sh <dir> [docker-image]}"
IMAGE="${2:-${RESET_RESULTS_DIR_IMAGE:-}}"

is_clear() { [ ! -e "${TARGET}" ]; }

echo "--- Resetting results directory: ${TARGET}"

rm -rf "${TARGET}" 2>/dev/null

if ! is_clear; then
  echo "plain rm left contents behind (root-owned leftovers?); retrying with sudo"
  sudo -n rm -rf "${TARGET}" 2>/dev/null
fi

if ! is_clear && [ -n "${IMAGE}" ]; then
  echo "sudo rm did not clear it; retrying inside a root container"
  docker run --rm -v "$(dirname "${TARGET}"):/reset_parent" "${IMAGE}" \
    rm -rf "/reset_parent/$(basename "${TARGET}")" >/dev/null 2>&1
fi

if ! is_clear; then
  # Nothing could unlink the contents. Rename the whole tree aside so this job
  # still gets a clean directory, then make one more (best-effort) attempt to
  # delete the renamed copy -- these agents only have a 100G boot disk.
  STALE="${TARGET}.stale.${BUILDKITE_JOB_ID:-$$}"
  if ! mv "${TARGET}" "${STALE}" 2>/dev/null; then
    echo "ERROR: cannot clear or rename ${TARGET}; this agent needs manual cleanup"
    ls -la "${TARGET}" || true
    exit 1
  fi
  echo "WARNING: could not delete ${TARGET}; moved it to ${STALE}"
  echo "WARNING: a container wrote root-owned files here without 'umask 000'"
  sudo -n rm -rf "${STALE}" 2>/dev/null || true
  if [ -n "${IMAGE}" ] && [ -e "${STALE}" ]; then
    docker run --rm -v "$(dirname "${STALE}"):/reset_parent" "${IMAGE}" \
      rm -rf "/reset_parent/$(basename "${STALE}")" >/dev/null 2>&1 || true
  fi
  [ -e "${STALE}" ] && echo "WARNING: ${STALE} is still on disk and is leaking space"
fi

mkdir -p "${TARGET}"
chmod 777 "${TARGET}" 2>/dev/null || true
