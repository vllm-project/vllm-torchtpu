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
# Provenance: model-owned copy of
# src/vllm_torchtpu/kernels/experimental/batched_rpa/__init__.py
#   (reference at 33ee547ba947, Qwen3.8 changes through
#   2957c85f0de3f974347e4111cb4487955d362658).
#   Owned by models/qwen38_flash from this copy on.

# NOTE: all of the code in this directory is experimental and not fully tested!
# To enable usage of this kernel in a full run, pass `--attention-backend CUSTOM`
# to vLLM. `TpuPlatform.pre_register_and_update` registers
# `PallasBatchedRPAAttentionBackend` under `AttentionBackendEnum.CUSTOM`.
