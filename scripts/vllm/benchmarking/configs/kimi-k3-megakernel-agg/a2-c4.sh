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

# Kimi-K3 megakernel aggregate point: concurrency 4 on one 2x2x4 slice,
# --max-num-seqs 2, DSpark with 7 drafts at synthetic acceptance 3.78, as the
# 1016 set's point 2 decode. See agg-base.sh.

# shellcheck disable=SC2034  # read by agg-base.sh
AGG_CONCURRENCY=4
AGG_MAX_NUM_SEQS=2
AGG_DRAFTS=7
AGG_ACCEPTANCE=3.78
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel-agg/agg-base.sh
source "$(dirname "${BASH_SOURCE[0]}")/agg-base.sh"
