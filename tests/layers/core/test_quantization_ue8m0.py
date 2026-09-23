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
"""Tests for the UE8M0 (power-of-two) block scales in `quantize_tensor`."""

import torch

from vllm_torchtpu.layers.core.quantization import e8m0_to_fp32, quantize_tensor

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


def _quantize_ue8m0(tensor: torch.Tensor, block_size: int | None = None):
    return quantize_tensor(
        tensor, FP8_DTYPE, axis=-1, block_size=block_size, use_ue8m0=True
    )


def test_scale_is_a_ue8m0_byte(device):
    tensor = torch.randn(4, 128, device=device, dtype=torch.bfloat16)

    _, scale = _quantize_ue8m0(tensor, block_size=128)

    assert scale.dtype == torch.uint8
    assert scale.shape == (4, 1)
    # Decoded scales are exact powers of two, and the byte round-trips through
    # torch's own e8m0 encoder.
    decoded = e8m0_to_fp32(scale)
    assert torch.equal(torch.exp2(torch.log2(decoded).round()), decoded)
    assert torch.equal(decoded.to(torch.float8_e8m0fnu).view(torch.uint8), scale)


def test_dequantized_values_are_close(device):
    torch.manual_seed(0)
    tensor = torch.randn(16, 512, device=device, dtype=torch.bfloat16) * 3.0

    quantized, scale = _quantize_ue8m0(tensor, block_size=128)
    dequantized = quantized.float().reshape(16, 4, 128) * e8m0_to_fp32(scale).unsqueeze(
        -1
    )
    dequantized = dequantized.reshape(16, 512)

    # A ue8m0 scale wastes up to one octave of fp8 range, so the relative
    # error budget is ~2x the 4-bit-mantissa fp8 step.
    reference = tensor.float()
    error = (dequantized - reference).abs()
    tolerance = reference.abs().amax() * 2 / 2**4
    assert (error <= tolerance).all(), error.max()
