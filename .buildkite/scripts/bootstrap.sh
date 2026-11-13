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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/configs/pipeline_config.sh"

JOB_PRIORITY="${PRIORITY_PRE_MERGE:-6}"
export JOB_PRIORITY
if command -v buildkite-agent >/dev/null 2>&1; then
  buildkite-agent meta-data set "JOB_PRIORITY" "$JOB_PRIORITY" || true
fi

echo "--- Buildkite Bootstrap Finished (qwen3p5_397b_submission_1001 branch)"
exit 0
