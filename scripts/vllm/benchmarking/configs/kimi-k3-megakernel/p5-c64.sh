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

# Recipe point 5: concurrency 64, decode admits 96 sequences, no speculation,
# two attention groups of DCP 16 inside the TP32 rank group, and a larger KV
# reservation on both roles (22 GiB prefill, 28 GiB decode per rank).
# See recipe-base.sh.

# shellcheck disable=SC2034  # read by recipe-base.sh
RECIPE_CONCURRENCY=64
RECIPE_DECODE_MAX_NUM_SEQS=96
RECIPE_ATTENTION_DP=2
RECIPE_DECODE_DCP=16
RECIPE_PREFILL_KV_BYTES=23622320128
RECIPE_DECODE_KV_BYTES=30064771072
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/recipe-base.sh"
