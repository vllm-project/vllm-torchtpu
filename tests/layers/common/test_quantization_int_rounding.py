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
"""`quantize_tensor` rounds to nearest on integer targets, never truncates.

A float->int cast truncates toward zero, which is a systematic half-LSB bias on
every element rather than a rounding wash: the error becomes uniform over one
LSB (rms `LSB/sqrt(3)`) instead of over half an LSB (rms `LSB/sqrt(12)`), i.e.
exactly 2x. Float targets round to nearest inside the cast itself, so they must
stay untouched.
"""

import pytest
import torch

from vllm_torchtpu.layers.common.quantization import (dequantize_tensor,
                                                      quantize_tensor)


def _truncating_quantize(tensor: torch.Tensor, quant_dtype: torch.dtype,
                         block_size: int | None):
    """`quantize_tensor`'s integer path as it behaved before the round.

    Reuses the shipped scale so the two paths differ only in the cast.
    """
    _, scale = quantize_tensor(tensor,
                               quant_dtype=quant_dtype,
                               axis=-1,
                               block_size=block_size)
    info = torch.iinfo(quant_dtype)
    blocks = tensor.reshape(*tensor.shape[:-1], scale.shape[-1], -1)
    # Reciprocal multiply, not division -- `quantize_tensor` does this
    # deliberately and it moves tie points by an ULP.
    q = torch.clamp(blocks * (1.0 / scale).unsqueeze(-1),
                    min=float(info.min),
                    max=float(info.max)).to(quant_dtype)
    return q.reshape_as(tensor), scale


def _rel_l2(ref: torch.Tensor, got: torch.Tensor) -> float:
    return (torch.linalg.vector_norm(got - ref) /
            torch.linalg.vector_norm(ref)).item()


def test_half_lsb_ties_land_on_the_nearest_grid_point_half_to_even():
    """Pins the tie-breaking convention: round-half-to-even, as `torch.round`.

    Half-to-even is what this repo's JAX quantizer already does (`jnp.round` in
    `kernels/quantized_matmul/util.quantize_block`) and what IEEE 754 defaults
    to; half-away-from-zero would swap the sign of the residual bias instead of
    removing it.
    """
    # abs_max == 127 == iinfo(int8).max makes the scale exactly 1.0, so every
    # value below is an exact .5 tie on the int8 grid with no fp32 slack.
    tensor = torch.tensor([[127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 63.5]])

    q, scale = quantize_tensor(tensor, quant_dtype=torch.int8, axis=-1)

    assert scale.item() == 1.0
    torch.testing.assert_close(
        q,
        torch.tensor([[127, 0, 2, 2, 0, -2, -2, 64]], dtype=torch.int8),
    )
    # What the truncating cast produced, for the record.
    torch.testing.assert_close(
        _truncating_quantize(tensor, torch.int8, None)[0],
        torch.tensor([[127, 0, 1, 2, 0, -1, -2, 63]], dtype=torch.int8),
    )


@pytest.mark.parametrize("block_size", [None, 512])
def test_rounding_is_strictly_closer_than_truncation(block_size):
    torch.manual_seed(0)
    tensor = torch.randn(8, 4096)

    q, scale = quantize_tensor(tensor,
                               quant_dtype=torch.int8,
                               axis=-1,
                               block_size=block_size)
    q_trunc, scale_trunc = _truncating_quantize(tensor, torch.int8, block_size)

    assert torch.equal(scale, scale_trunc), "only the cast may differ"
    rounded = _rel_l2(tensor, dequantize_tensor(q, scale, axis=(-1, )))
    truncated = _rel_l2(tensor, dequantize_tensor(q_trunc, scale, axis=(-1, )))
    assert rounded < truncated
    assert truncated / rounded == pytest.approx(2.0, abs=0.05)


def test_integer_error_matches_the_uniform_half_lsb_bound():
    """The analytic claim, as an assertion: no bias, rms `LSB/sqrt(12)`."""
    torch.manual_seed(0)
    tensor = torch.randn(4, 8192)

    q, scale = quantize_tensor(tensor, quant_dtype=torch.int8, axis=-1)
    err_lsb = (tensor - dequantize_tensor(q, scale, axis=(-1, ))) / scale

    assert err_lsb.abs().max().item() <= 0.5 + 1e-3
    assert err_lsb.mean().abs().item() < 0.01  # truncation gives ~0.5
    assert err_lsb.pow(2).mean().sqrt().item() == pytest.approx(12**-0.5,
                                                                rel=0.05)


def test_float_targets_keep_the_sub_unit_grid():
    """fp8 rounds inside the cast already; the integer branch must not fire."""
    # abs_max == 448 == finfo(float8_e4m3fn).max, so the scale is again 1.0 and
    # the fractional values are exactly representable in fp8.
    tensor = torch.tensor([[448.0, 0.5, 1.5, 2.5]])

    q, scale = quantize_tensor(tensor,
                               quant_dtype=torch.float8_e4m3fn,
                               axis=-1)

    assert scale.item() == 1.0
    torch.testing.assert_close(q.float(), tensor)
