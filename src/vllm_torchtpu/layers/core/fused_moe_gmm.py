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
"""Fused MoE GMM operator and unpermutation logic.

Re-exports core primitives from `llm_tpu_core.layers.fused_moe_gmm`.
"""

from vllm_torchtpu.llm_tpu_core.layers.fused_moe_gmm import (
    _counting_sort_positions,
    _select_ragged_gather_reduce,
    fused_moe_func,
    gmm_wrapper,
    moe_gmm,
    prepare_routed_gmm_inputs,
    quantize_to_native_fp4_kmajor,
    ragged_gather_reduce,
    requant_unpack_kmajor,
    resolve_onehot_permute_threshold,
    unpack_fp4_to_e2m1,
)

__all__ = [
    "_counting_sort_positions",
    "_select_ragged_gather_reduce",
    "fused_moe_func",
    "gmm_wrapper",
    "moe_gmm",
    "prepare_routed_gmm_inputs",
    "quantize_to_native_fp4_kmajor",
    "ragged_gather_reduce",
    "requant_unpack_kmajor",
    "resolve_onehot_permute_threshold",
    "unpack_fp4_to_e2m1",
]
