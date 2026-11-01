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
# Formosa synthetic_v1 (timed trace a_right_long, N=256, c=1..32) on the
# aggregate engine, without speculative decoding, as the GB300 baseline runs.

# shellcheck disable=SC2034  # read by agg-base.sh and the client
AGG_CONCURRENCY="1 2 4 8 16 32"
AGG_MAX_NUM_SEQS=32
AGG_DRAFTS=0
AGG_ACCEPTANCE=0
FORMOSA_BENCH=syn1
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel-agg/agg-base.sh
source "$(dirname "${BASH_SOURCE[0]}")/agg-base.sh"
