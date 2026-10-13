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

# Recipe point 4: concurrency 32, decode admits 16 sequences (the rest queue),
# DSpark 3 drafts (synthetic acceptance 2.87), DCP 32.
# See recipe-base.sh.

# shellcheck disable=SC2034  # read by recipe-base.sh
RECIPE_CONCURRENCY=32
RECIPE_DECODE_MAX_NUM_SEQS=16
RECIPE_DECODE_DRAFTS=3
RECIPE_DECODE_ACCEPTANCE=2.87
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/recipe-base.sh"
