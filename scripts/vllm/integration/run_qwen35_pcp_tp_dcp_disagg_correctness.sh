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

# Disaggregated serving correctness test for Qwen3.5 with combined
# PCP+TP prefill (producer) and TP+DCP decode (consumer) hybrid state handoff.
# Scaled functional geometry: PCP4TP1 -> TP4DCP2 (8 devices on a single host).

set -euo pipefail
umask 000

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROLE="${ROLE:-all}"
export PREFILL_PCP="${PREFILL_PCP:-4}"
export PREFILL_TP="${PREFILL_TP:-1}"
export DECODE_TP="${DECODE_TP:-4}"
export DECODE_DP="${DECODE_DP:-1}"
export DECODE_DCP="${DECODE_DCP:-2}"
export PREFILL_BLOCK_SIZE="${PREFILL_BLOCK_SIZE:-256}"
export DECODE_BLOCK_SIZE="${DECODE_BLOCK_SIZE:-256}"
export PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE="${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE:-128}"
export TPU_RAIDEN_DST_SHARDS="${TPU_RAIDEN_DST_SHARDS:-${DECODE_TP}}"
export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-HND}"
export TPU_RAIDEN_TRANSFER_PARALLELISM="${TPU_RAIDEN_TRANSFER_PARALLELISM:-$((PREFILL_PCP * PREFILL_TP))}"
export RUN_ROOT="${RUN_ROOT:-/tmp/qwen35_pcp_tp_dcp_disagg_ci}"

exec bash "${script_dir}/run_qwen35_p8d8_disagg_correctness.sh" "$@"
