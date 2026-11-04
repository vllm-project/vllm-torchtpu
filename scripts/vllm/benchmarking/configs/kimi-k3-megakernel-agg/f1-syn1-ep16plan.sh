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
# f1-syn1 on an engine with EP16 aggregate experts and a static expert placement
# plan for the uniform-vocab timed-trace prompts (dev/k3-agg-formosa-place-ep16).

export K3_AGG_EP16=1
export K3_DECODE_EXPERT_PLAN=plan_formosa_ep16.npz
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel-agg/f1-syn1.sh
source "$(dirname "${BASH_SOURCE[0]}")/f1-syn1.sh"
