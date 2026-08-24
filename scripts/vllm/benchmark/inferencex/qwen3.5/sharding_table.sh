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

# Share one point-to-sharding mapping across benchmark entry points.
# Unknown points let callers choose their own fallback.

declare -A SHARDING_TABLE=(
  [8192,1024,4]=TP8_EP
  [8192,1024,8]=TP8_EP
  [8192,1024,16]=TP8_EP
  [8192,1024,32]=TP8_EP
  [8192,1024,64]=TP8_EP
  [8192,1024,128]=DP8_EP
  [8192,1024,256]=DP8_EP
)

sharding_for() {
  local key="$1,$2,$3"
  local found="${SHARDING_TABLE[$key]:-}"
  [ -n "$found" ] || return 1
  printf '%s' "$found"
}
