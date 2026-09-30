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

import sys
from typing import Any

from vllm_torchtpu.llm_tpu_core.layers import fused_moe_gmm as _core_fused_moe_gmm
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

_this_module = sys.modules[__name__]

for _k in dir(_core_fused_moe_gmm):
    if not (_k.startswith("__") and _k.endswith("__")):
        setattr(_this_module, _k, getattr(_core_fused_moe_gmm, _k))


class _FusedMoeGmmModuleWrapper:
    def __init__(self, wrapped_module):
        self.__dict__["_wrapped_module"] = wrapped_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value
        setattr(self._wrapped_module, name, value)
        if hasattr(_core_fused_moe_gmm, name):
            setattr(_core_fused_moe_gmm, name, value)


sys.modules[__name__] = _FusedMoeGmmModuleWrapper(_this_module)
