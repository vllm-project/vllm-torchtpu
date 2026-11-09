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
set -Eeuo pipefail
results="${1:?results directory is required}"
mkdir -p "$results"
# Keep these local TPU checks on the original 8-device host. Each process
# releases the TPU before the next check starts.
python -m pytest -q tests/kernels/fused_moe/test_fused_ep_moe_v2.py \
  -k row_padded_flat_scales --junitxml="$results/kernel-scales.xml"
