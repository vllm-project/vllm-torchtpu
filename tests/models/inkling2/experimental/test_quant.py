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
"""CPU unit tests for Inkling 2 checkpoint quantization metadata and codecs."""

from __future__ import annotations

import pytest
import torch

from vllm_torchtpu.models.inkling2.experimental.quant import (
    Inkling2QuantMap,
    dequantize_mxfp8,
    dequantize_nvfp4,
    unpack_uint8_to_fp4,
)


def _sample_hf_quant_config() -> dict:
    return {
        "model": {
            "parameter_groups": [
                {
                    "compute_dtype": "float32",
                    "tensors": ["model.llm.attn_res.output_query"],
                },
                {
                    "compute_dtype": "bfloat16",
                    "tensors": ["model.llm.embed.weight"],
                },
                {
                    "compute_dtype": "nvfp4",
                    "tensors": [
                        f"model.llm.layers.{layer}.mlp.experts.w13_weight"
                        for layer in range(20)
                    ],
                },
                {
                    "compute_dtype": "mxfp8",
                    "tensors": [
                        f"model.llm.layers.{layer}.mlp.experts.w13_weight"
                        for layer in range(20, 24)
                    ],
                },
            ]
        }
    }


def test_quant_map_routes_mixed_nvfp4_and_mxfp8_layers() -> None:
    qmap = Inkling2QuantMap.from_hf_quant_config(_sample_hf_quant_config())
    assert qmap.expert_format(0) == "nvfp4"
    assert qmap.expert_format(19) == "nvfp4"
    assert not qmap.is_mxfp8_layer(19)
    assert qmap.expert_format(20) == "mxfp8"
    assert qmap.expert_format(23) == "mxfp8"
    assert qmap.is_mxfp8_layer(20)
    with pytest.raises(KeyError):
        qmap.expert_format(24)


def test_quant_map_rejects_unknown_compute_dtype() -> None:
    bad = {"model": {"parameter_groups": [{"compute_dtype": "int2", "tensors": []}]}}
    with pytest.raises(ValueError, match="Unsupported Inkling2 compute_dtype"):
        Inkling2QuantMap.from_hf_quant_config(bad)


def test_dequantize_mxfp8_exact_power_of_two_batched() -> None:
    row = torch.tensor([1.0, -2.0] * 32).to(torch.float8_e4m3fn)
    weight = torch.stack((row, row)).unsqueeze(1)
    scales = torch.tensor([[[127 + 3, 127 - 1]], [[127, 127 + 2]]], dtype=torch.uint8)
    out = dequantize_mxfp8(weight, scales)
    assert out.dtype == torch.bfloat16
    assert out.shape == (2, 1, 64)
    expected_e0 = torch.tensor([8.0, -16.0] * 16 + [0.5, -1.0] * 16)
    expected_e1 = torch.tensor([1.0, -2.0] * 16 + [4.0, -8.0] * 16)
    torch.testing.assert_close(out[0, 0].float(), expected_e0)
    torch.testing.assert_close(out[1, 0].float(), expected_e1)


def test_dequantize_mxfp8_rejects_bad_dtype_or_shape() -> None:
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        dequantize_mxfp8(
            torch.zeros(1, 32, dtype=torch.bfloat16),
            torch.full((1, 1), 127, dtype=torch.uint8),
        )
    with pytest.raises(ValueError, match="positive multiple"):
        dequantize_mxfp8(
            torch.zeros(1, 30, dtype=torch.float8_e4m3fn),
            torch.full((1, 4), 127, dtype=torch.uint8),
        )


def test_dequantize_nvfp4_block16_and_global_scale() -> None:
    # Low nibble = 0x02 (+1.0), high nibble = 0x0C (-2.0) -> byte 0xC2
    packed = torch.full((2, 1, 16), 0xC2, dtype=torch.uint8)
    unpacked = unpack_uint8_to_fp4(packed)
    assert unpacked.shape == (2, 1, 32)
    assert unpacked[0, 0, 0].item() == 1.0
    assert unpacked[0, 0, 1].item() == -2.0

    block_scales = torch.tensor([[[2.0, 0.5]], [[4.0, 1.0]]], dtype=torch.float8_e4m3fn)
    amax = torch.tensor([10.0, 20.0], dtype=torch.float32)
    denom = torch.tensor([20.0], dtype=torch.float32)
    out = dequantize_nvfp4(packed, block_scales, amax, denom, out_dtype=torch.float32)
    expected_e0 = torch.tensor([1.0, -2.0] * 8 + [0.25, -0.5] * 8)
    expected_e1 = torch.tensor([4.0, -8.0] * 8 + [1.0, -2.0] * 8)
    torch.testing.assert_close(out[0, 0], expected_e0)
    torch.testing.assert_close(out[1, 0], expected_e1)


def test_unpack_uint8_to_fp4_exhaustive_256_bytes() -> None:
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
    )
    all_bytes = torch.arange(256, dtype=torch.uint8)
    unpacked = unpack_uint8_to_fp4(all_bytes)
    expected = torch.stack(
        (lut[(all_bytes & 0x0F).long()], lut[((all_bytes >> 4) & 0x0F).long()]),
        dim=-1,
    ).reshape(512)
    assert unpacked.shape == (512,)
    assert torch.equal(unpacked.view(torch.int32), expected.view(torch.int32))
