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

# Kimi-K3 megakernel recipe point 4 of the Oct 16 set: concurrency 16; decode
# admits 8 sequences, DSpark (MLA draft) with 7 drafts at batch size 1 and 3 at
# 2-8, DCP32; decode-host store servers. See ../kimi-k3-megakernel/recipe-base.sh.

# shellcheck disable=SC2034  # read by recipe-base.sh
RECIPE_SET=oct16
RECIPE_CONCURRENCY=16
RECIPE_DECODE=dspark_mixed
RECIPE_DECODE_MAX_NUM_SEQS=8
RECIPE_DRAFTS_PER_BATCH_SIZE='[[1,1,7],[2,8,3]]'
RECIPE_STORE_SERVERS=1
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel/recipe-base.sh
source "$(dirname "${BASH_SOURCE[0]}")/../kimi-k3-megakernel/recipe-base.sh"
