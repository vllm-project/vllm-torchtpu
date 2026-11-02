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

# Production engine and Raiden settings shared by the Qwen3.5 PD disagg CI
# suites that run them; qwen35-pcp4-dp4 stays on the defaults.

export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-HND}"
export TPU_PREMAPPED_BUFFER_SIZE="${TPU_PREMAPPED_BUFFER_SIZE:-4294967296}"
export TPU_PARALLEL_PRECOMPILE="${TPU_PARALLEL_PRECOMPILE:-1}"
export TPU_MOE_COLLECTION_CHUNK_SIZE="${TPU_MOE_COLLECTION_CHUNK_SIZE:-65536}"
export RAGGED_GATHER_REDUCE_VERSION="${RAGGED_GATHER_REDUCE_VERSION:-v3}"
export TPU_MOE_OWNER_OUTPUT_MODE="${TPU_MOE_OWNER_OUTPUT_MODE:-on}"
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"
export PREFILL_MOE_FUSED_EP_KERNEL_IMPL="${PREFILL_MOE_FUSED_EP_KERNEL_IMPL-adaptive}"
export TPU_RAIDEN_POOL_STAGING_LEASES="${TPU_RAIDEN_POOL_STAGING_LEASES:-1}"
export TPU_RAIDEN_PREFIX_AWARE_LOAD="${TPU_RAIDEN_PREFIX_AWARE_LOAD:-1}"
export TPU_RAIDEN_TELEMETRY_BACKENDS="${TPU_RAIDEN_TELEMETRY_BACKENDS:-buffered}"
