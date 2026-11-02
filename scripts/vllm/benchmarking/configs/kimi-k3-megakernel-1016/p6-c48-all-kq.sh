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
# p6-c48-all with the prefill MoE prologue options: three weight-panel DMA banks
# and the one-shot MoE reduce-scatter up to 1024-token windows (engine
# dev/k3-prefill-kernel 2fbc18cf5 or later).
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel-1016/p6-c48-all.sh
source "$(dirname "${BASH_SOURCE[0]}")/p6-c48-all.sh"
export K3_PREFILL_MOE_DMA_BANKS=3
export K3_PREFILL_MOE_ONESHOT_MAX=1024
