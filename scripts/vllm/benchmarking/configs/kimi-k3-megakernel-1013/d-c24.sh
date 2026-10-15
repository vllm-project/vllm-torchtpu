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

# Kimi-K3 megakernel recipe pruned point d of the Oct 13 set: concurrency 24;
# decode admits 16 sequences, DSpark with 3 drafts at synthetic acceptance 2.87,
# DCP32; decode-host store servers. See ../kimi-k3-megakernel/recipe-base.sh.

# shellcheck disable=SC2034  # read by recipe-base.sh
RECIPE_SET=oct13
RECIPE_CONCURRENCY=24
RECIPE_DECODE=dspark
RECIPE_DECODE_MAX_NUM_SEQS=16
RECIPE_DRAFTS=3
RECIPE_ACCEPTANCE=2.87
RECIPE_STORE_SERVERS=1
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel/recipe-base.sh
source "$(dirname "${BASH_SOURCE[0]}")/../kimi-k3-megakernel/recipe-base.sh"
