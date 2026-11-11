#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
#   dsv41_publish_check.sh <benchmark-dir> <command> [args...]
#
# Runs a DSV4.1 throughput benchmark, then publishes its result to the step's
# GitHub check with dsv41_check_summary.py, whatever the benchmark's outcome:
# a failed run publishes "no valid throughput" rather than leaving the
# placeholder. Runs inside the pod, where BUILDKITE_* and buildkite-agent are
# present; the VM lane did this from a post-command hook on the agent.
#
# Exits with the benchmark's status when it failed, else the publisher's, so a
# lost check update fails the step without masking a benchmark failure.
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <benchmark-dir> <command> [args...]" >&2
  exit 2
fi
root="$1"
shift

status=0
"$@" || status=$?

summary_status=0
python3 "$(dirname "${BASH_SOURCE[0]}")/dsv41_check_summary.py" \
  "$root" "$status" "${BUILDKITE_COMMIT:-unknown}" \
  "${BUILDKITE_BUILD_URL:-}#${BUILDKITE_JOB_ID:-}" || summary_status=$?
if [[ "$status" != 0 ]]; then
  exit "$status"
fi
exit "$summary_status"
