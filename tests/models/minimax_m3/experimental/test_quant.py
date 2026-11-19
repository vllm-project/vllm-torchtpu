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
"""MiniMax M3 checkpoint quant config on CPU."""

import pytest
import torch

from vllm_torchtpu.models.minimax_m3.experimental.quant import (
    MiniMaxM3QuantConfig,
    dequantize_mxfp8,
)

_LM = "language_model.model"


def _quant_dict() -> dict:
    return {
        "quant_method": "mxfp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [1, 32],
        "group_size": 16,
        "moe_quant_algo": "NVFP4",
        "moe_quant_format": "nvfp4-pack-quantized",
        "ignored_layers": [
            f"{_LM}.layers.1.block_sparse_moe.gate",
            "language_model.lm_head",
            f"{_LM}.embed_tokens",
            "vision_tower",
        ],
    }


def test_dequantize_mxfp8_applies_power_of_two_block_scales() -> None:
    weight = torch.tensor([[1.0, -2.0] * 32]).to(torch.float8_e4m3fn)
    scale = torch.tensor([[127 + 3, 127 - 1]], dtype=torch.uint8)
    out = dequantize_mxfp8(weight, scale)
    assert out.dtype == torch.bfloat16
    expected = torch.tensor([[8.0, -16.0] * 16 + [0.5, -1.0] * 16])
    torch.testing.assert_close(out.float(), expected)


def test_quant_config_rejects_other_schemes() -> None:
    config = _quant_dict() | {"moe_quant_algo": "MXFP4"}
    with pytest.raises(NotImplementedError):
        MiniMaxM3QuantConfig.from_config(config)
