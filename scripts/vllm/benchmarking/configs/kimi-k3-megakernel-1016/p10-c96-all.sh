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
# p10-c96 with the p8-c72-all flags: hot-expert MXFP4 replicas (p8-c72 plan), fp8 wires, fp8 prefix attention, prefill expert placement (p6-c48 plan), SJF prefill queue.
# Needs an engine with the flags (dev/k3-kernel-numerics d5a48b7b3 or later).
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-megakernel-1016/p10-c96.sh
source "$(dirname "${BASH_SOURCE[0]}")/p10-c96.sh"
export K3_REPLICA_PLAN=replica_plan_p8c72_plain4.npz
export K3_FP8_WIRES=t0,t10
export K3_PREFILL_PREFIX_FP8=1
export K3_PREFILL_EXPERT_PLAN=plan_prefill_serve.npz
export K3_PREFILL_SJF=1
