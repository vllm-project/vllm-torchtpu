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
"""Fused expert-parallel MoE kernel v3: the fused_ep_moe entry point, its
Config, the weight formats, the VMEM accounting and the layout constants a
caller has to match.
"""

from vllm_torchtpu.kernels.fused_moe.v3 import device, ffn, layer, layout, vmem
from vllm_torchtpu.kernels.fused_moe.v3.config import Config
from vllm_torchtpu.kernels.fused_moe.v3.formats import (
    PACKED_ROW_TILE,
    WEIGHT_BLOCK_DEFAULT,
    WEIGHT_FORMAT_NAMES,
    WEIGHT_FORMS,
    WeightFormat,
    weight_form,
    weight_format_of_dtype,
)
from vllm_torchtpu.kernels.fused_moe.v3.layer import fused_ep_moe
from vllm_torchtpu.kernels.fused_moe.v3.layout import LANES, MESH_AXIS, ROW_BLOCK
from vllm_torchtpu.kernels.fused_moe.v3.routing_tables import (
    ALIGNMENT_SLOT_FIELD,
    routing_block,
)

SUPPORTED_GENERATIONS = device.SUPPORTED_GENERATIONS
chip_generation = device.chip_generation
supported_on = device.supported_on

__all__ = [
    "device",
    "ffn",
    "layer",
    "layout",
    "vmem",
    "ALIGNMENT_SLOT_FIELD",
    "Config",
    "LANES",
    "MESH_AXIS",
    "PACKED_ROW_TILE",
    "ROW_BLOCK",
    "SUPPORTED_GENERATIONS",
    "WEIGHT_BLOCK_DEFAULT",
    "WEIGHT_FORMS",
    "WEIGHT_FORMAT_NAMES",
    "WeightFormat",
    "chip_generation",
    "fused_ep_moe",
    "routing_block",
    "supported_on",
    "weight_form",
    "weight_format_of_dtype",
]
