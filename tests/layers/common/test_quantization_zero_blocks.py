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
"""Regression tests for zero-valued quantization blocks."""

import torch

from vllm_torchtpu.layers.common.quantization import (quantize_tensor,
                                                      quantize_tensor_to_fp4)


def test_quantize_tensor_keeps_zero_rows_finite(device):
    tensor = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, -2.0, 3.0, -4.0]],
        device=device,
        dtype=torch.float32,
    )

    quantized, scale = quantize_tensor(tensor,
                                       quant_dtype=torch.float8_e4m3fn,
                                       axis=-1,
                                       block_size=None)

    assert not torch.isnan(quantized.float()).any()
    assert not torch.isinf(quantized.float()).any()
    assert torch.equal(quantized[0].float(),
                       torch.zeros_like(quantized[0].float()))
    assert scale[0].item() == 0.0


def test_quantize_tensor_uses_reciprocal_multiply_scale_rounding():
    tensor = torch.tensor([[0.0010000200709328055]],
                          device="cpu",
                          dtype=torch.float32)
    dtype_max = float(torch.finfo(torch.float8_e4m3fn).max)
    expected_scale = tensor.abs().amax(dim=-1, keepdim=True) * torch.tensor(
        1.0 / dtype_max, dtype=torch.float32)

    _, scale = quantize_tensor(tensor,
                               quant_dtype=torch.float8_e4m3fn,
                               axis=-1,
                               block_size=None)

    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)


def test_quantize_tensor_to_fp4_keeps_zero_blocks_finite(device):
    tensor = torch.tensor(
        [[0.0, 0.0, 1.0, -1.0], [2.0, -2.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )

    indices, scale = quantize_tensor_to_fp4(tensor, axis=-1, block_size=2)
    dequantized = indices.float()

    assert not torch.isnan(dequantized).any()
    assert not torch.isinf(dequantized).any()
    assert torch.equal(indices[0, :2], torch.zeros_like(indices[0, :2]))
    assert torch.equal(indices[1, 2:], torch.zeros_like(indices[1, 2:]))
    assert torch.allclose(
        scale,
        torch.tensor([[0.0, 1.0 / 6.0], [2.0 / 6.0, 0.0]],
                     device=device,
                     dtype=torch.float32),
    )
