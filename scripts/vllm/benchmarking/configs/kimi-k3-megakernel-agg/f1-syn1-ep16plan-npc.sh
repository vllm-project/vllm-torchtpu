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
# f1-syn1-ep16plan without prefix caching (as the GB300 baseline runs; the
# trace has no prefix hits), c=1..64 with 64 seats. Engine needs #2615.

export K3_AGG_EP16=1
export K3_DECODE_EXPERT_PLAN=plan_formosa_ep16.npz
# shellcheck disable=SC2034  # read by agg-base.sh and the client
AGG_CONCURRENCY="1 2 4 8 16 32 64"
AGG_MAX_NUM_SEQS=64
AGG_DRAFTS=0
AGG_ACCEPTANCE=0
AGG_PREFIX_CACHING=0
FORMOSA_BENCH=syn1
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel-agg/agg-base.sh
source "$(dirname "${BASH_SOURCE[0]}")/agg-base.sh"
