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
"""Tests for the per-row fp8 activation quantizer used by the fp8 all-gather."""

import pytest
import torch

from vllm_torchtpu.layers.core.quantization import dynamic_row_quantize_fp8

FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _dequantize(quantized: torch.Tensor,
                scale: torch.Tensor) -> torch.Tensor:
    return quantized.to(torch.float32) * scale


@pytest.mark.parametrize("dtype",
                         [torch.bfloat16, torch.float16, torch.float32])
def test_row_quantize_round_trip(dtype):
    torch.manual_seed(0)
    tensor = torch.randn(16, 256, dtype=dtype)

    quantized, scale = dynamic_row_quantize_fp8(tensor)

    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert scale.shape == (16, 1)
    # e4m3 keeps four mantissa bits, so a value can land half a step -- 1/32 of
    # its own magnitude -- off. The row's amax sets the step, hence the atol.
    torch.testing.assert_close(_dequantize(quantized, scale),
                               tensor.to(torch.float32),
                               rtol=1.0 / 32,
                               atol=float(tensor.abs().max()) / 32)


def test_row_quantize_scales_each_row_independently():
    """Every row should reach the top of the fp8 range, whatever its size."""
    tensor = torch.tensor([[1e-3, -5e-4], [1e4, 2e3]], dtype=torch.float32)

    quantized, scale = dynamic_row_quantize_fp8(tensor)

    torch.testing.assert_close(
        quantized.to(torch.float32).abs().amax(dim=-1),
        torch.full((2, ), FP8_MAX))
    torch.testing.assert_close(scale.flatten(),
                               torch.tensor([1e-3, 1e4]) / FP8_MAX)


@pytest.mark.parametrize(
    "row",
    [
        pytest.param([0.0, 0.0, 0.0], id="all_zero"),
        pytest.param([float("inf"), 1.0, 2.0], id="infinite"),
        pytest.param([float("nan"), 1.0, 2.0], id="nan"),
        # Small enough that 1/scale is not an f32 number. The reciprocal-times
        # -row form would write NaN across the whole row here.
        pytest.param([1e-38, 0.0, 0.0], id="unscalably_small"),
    ],
)
def test_row_quantize_zeroes_rows_it_cannot_scale(row):
    tensor = torch.tensor([row, [1.0, -2.0, 4.0]], dtype=torch.float32)

    quantized, scale = dynamic_row_quantize_fp8(tensor)

    assert torch.isfinite(quantized.to(torch.float32)).all()
    assert torch.equal(quantized[0].to(torch.float32), torch.zeros(3))
    assert scale[0].item() == 0.0
    # The neighbouring row is quantized as if the bad one were not there.
    assert scale[1].item() == pytest.approx(4.0 / FP8_MAX)


def test_row_quantize_keeps_small_float16_rows_finite():
    """A row far below 1.0 needs a multiplier f16 cannot hold.

    ``1e-3`` asks for a scale reciprocal of ~4.5e5, well past f16's 65504, so
    rescaling in the row's own dtype would send the whole row to infinity and
    then to fp8's saturation value.
    """
    tensor = torch.tensor([[1e-3, -5e-4, 2.5e-4]], dtype=torch.float16)

    quantized, scale = dynamic_row_quantize_fp8(tensor)

    torch.testing.assert_close(_dequantize(quantized, scale),
                               tensor.to(torch.float32),
                               rtol=1.0 / 32,
                               atol=1e-3 / 32)


def test_row_quantize_survives_the_int8_bitcast_the_gather_uses():
    """The collective moves the rows as int8; the bits have to come back."""
    torch.manual_seed(0)
    quantized, scale = dynamic_row_quantize_fp8(
        torch.randn(8, 64, dtype=torch.bfloat16))

    round_tripped = quantized.view(torch.int8).view(torch.float8_e4m3fn)

    assert torch.equal(round_tripped.to(torch.float32),
                       quantized.to(torch.float32))
    assert scale.shape == (8, 1)
