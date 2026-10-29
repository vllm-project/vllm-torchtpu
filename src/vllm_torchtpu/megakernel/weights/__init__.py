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
"""Generic declarative weight loaders (`megakernel.weights`)."""

from __future__ import annotations

from .eplb import default_eplb_cores, encode_eplb_ecopy, resolve_mapping_ecopy
from .loader import MappingWeightSource, load_weights
from .logical import (
    load_hf,
    load_logical_weights,
    pack_logical_weights,
    pack_spec_logical_weights,
    resolve_spec_and_norms,
    switch_branch_for_layer,
)
from .ops import (
    ArrayOps,
    JaxArrayOps,
    WeightSource,
    apply_decl_pad,
    slice_alias_targets,
    target_axis_widths,
)
from .synthetic import (
    pack_inputs,
    pack_spec_inputs,
    random_hf_weights,
    random_inputs,
    random_logical_inputs,
    random_spec_inputs,
    random_spec_logical_weights,
)
from .vllm_synth import SyntheticVLLMWeightSource, normalize_hf_for_vllm

__all__ = [
    "ArrayOps",
    "JaxArrayOps",
    "MappingWeightSource",
    "SyntheticVLLMWeightSource",
    "WeightSource",
    "apply_decl_pad",
    "default_eplb_cores",
    "encode_eplb_ecopy",
    "load_hf",
    "load_logical_weights",
    "load_weights",
    "normalize_hf_for_vllm",
    "pack_inputs",
    "pack_logical_weights",
    "pack_spec_inputs",
    "pack_spec_logical_weights",
    "random_hf_weights",
    "random_inputs",
    "random_logical_inputs",
    "random_spec_inputs",
    "random_spec_logical_weights",
    "resolve_mapping_ecopy",
    "resolve_spec_and_norms",
    "slice_alias_targets",
    "switch_branch_for_layer",
    "target_axis_widths",
]
