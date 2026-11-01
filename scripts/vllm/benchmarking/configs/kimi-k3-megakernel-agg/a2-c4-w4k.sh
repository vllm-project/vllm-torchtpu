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

# a2-c4 with prefill windows capped at 4096 tokens (EP8 fused MoE is within 8% of
# EP16 up to 4096 and 22% slower at 8192).
# shellcheck disable=SC2034  # read by agg-base.sh
AGG_PREFILL_WINDOW=4096
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel-agg/a2-c4.sh
source "$(dirname "${BASH_SOURCE[0]}")/a2-c4.sh"
