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

# shellcheck disable=SC2034  # Read by qwen35_pd.sh and run_benchmarks.sh
# The C3 recipe (qwen3.5-397b-fp8-pd.sh) at the shape of its correctness gate:
# one prefill and one decode engine, DP4 x TP2 decode, real MTP rejection
# sampling, trade-offs on (validation entry gate-dp4tp2-1p1d, arm
# tradeoffs_on). 8 chips. Its baselines are this file's name.
PD_PREFILL_REPLICAS=1
PD_DECODE_REPLICAS=1
PD_DECODE_LAYOUT=dp4tp2
PD_TRADEOFFS=on
PD_MTP_SYNTHETIC_AL=""
# shellcheck source=scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-pd.sh
source "$(dirname "${BASH_SOURCE[0]}")/qwen3.5-397b-fp8-pd.sh"
