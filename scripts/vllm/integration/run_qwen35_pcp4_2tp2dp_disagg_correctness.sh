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

# Single-host Qwen3.5 PD disagg correctness on the production settings:
# PCP4 -> TP2xDP2 with the shared production env. Runs the same test as
# run_qwen35_p4d2_disagg_correctness.sh, which stays on the defaults.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${script_dir}/qwen35_pd_disagg_common_env.sh"

export DECODE_TP="${DECODE_TP:-2}"
export DECODE_DP="${DECODE_DP:-2}"
export TPU_RAIDEN_DST_SHARDS="${TPU_RAIDEN_DST_SHARDS:-${DECODE_TP}}"
export RUN_ROOT="${RUN_ROOT:-/tmp/qwen35_pcp4_2tp2dp_disagg_ci}"

exec bash "${script_dir}/run_qwen35_p4d2_disagg_correctness.sh" "$@"
