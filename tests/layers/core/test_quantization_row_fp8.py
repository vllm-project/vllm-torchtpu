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
"""Tests for the fp8 activation quantizer used by the fp8 all-gather."""

import pytest
import torch

from vllm_torchtpu.layers.core.quantization import (FP8_ACT_BLOCK_SIZE,
                                                    dequantize_block_fp8,
                                                    dynamic_block_quantize_fp8,
                                                    dynamic_row_quantize_fp8)

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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16,
                                   torch.float32])
@pytest.mark.parametrize(("size", "padded", "blocks"), [
    pytest.param(3 * FP8_ACT_BLOCK_SIZE, 3 * FP8_ACT_BLOCK_SIZE, 3,
                 id="exact_blocks"),
    pytest.param(FP8_ACT_BLOCK_SIZE + 1, 2 * FP8_ACT_BLOCK_SIZE, 2,
                 id="one_over"),
    pytest.param(7, FP8_ACT_BLOCK_SIZE, 1, id="under_one_block"),
])
def test_block_quantize_round_trip(dtype, size, padded, blocks):
    torch.manual_seed(0)
    tensor = torch.randn(16, size, dtype=dtype)

    quantized, scale = dynamic_block_quantize_fp8(
        tensor, block_size=FP8_ACT_BLOCK_SIZE)

    assert quantized.shape == (16, padded)
    assert scale.shape == (16, blocks)
    restored = dequantize_block_fp8(quantized, scale, dtype, size)
    assert restored.shape == tensor.shape
    torch.testing.assert_close(restored.to(torch.float32),
                               tensor.to(torch.float32),
                               rtol=1.0 / 32,
                               atol=float(tensor.abs().max()) / 32)


def test_block_quantize_zero_pads_the_tail_and_scales_it_with_its_block():
    """The pad is zeros, and it does not drag its block's scale down.

    The tail block holds one real column and 511 zeros; its amax has to come
    from that column alone, or the padding would set the scale for the block.
    """
    tensor = torch.zeros(1, FP8_ACT_BLOCK_SIZE + 1)
    tensor[0, :FP8_ACT_BLOCK_SIZE] = 1e3
    tensor[0, FP8_ACT_BLOCK_SIZE] = 2.0

    quantized, scale = dynamic_block_quantize_fp8(
        tensor, block_size=FP8_ACT_BLOCK_SIZE)

    torch.testing.assert_close(scale.flatten(),
                               torch.tensor([1e3, 2.0]) / FP8_MAX)
    # The pad quantizes to exact zeros, so dropping it changes nothing.
    padded = quantized.to(torch.float32)
    assert torch.equal(padded[0, FP8_ACT_BLOCK_SIZE + 1:],
                       torch.zeros(FP8_ACT_BLOCK_SIZE - 1))


def test_block_quantize_is_idempotent_at_the_kernel_s_own_block_width():
    """This is the whole reason the gather quantizes per block, not per row.

    The gmm re-quantizes its LHS in `FP8_ACT_BLOCK_SIZE`-wide blocks along K.
    A block's amax survives its own quantization -- the element that set it
    maps to e4m3's maximum exactly, so the second pass recovers the same scale
    and re-encodes the same codes. Rounding on the same grid at the wire
    therefore costs the kernel nothing: it ends up multiplying bit-identical
    values. Per-row scaling moves them off that grid, and the second rounding
    becomes real.

    Note what this test does *not* claim. Per-block scaling is not measurably
    more faithful to the original activations -- e4m3 carries enough dynamic
    range that a row scale only starts to lose blocks once one block outweighs
    another by ~1e5, which activations do not do. The win is the free second
    rounding, not the first one.
    """
    torch.manual_seed(0)
    hidden = 4 * FP8_ACT_BLOCK_SIZE
    tensor = torch.randn(16, hidden, dtype=torch.bfloat16)

    def kernel_lhs(activations: torch.Tensor) -> torch.Tensor:
        """Stand-in for the gmm's own LHS quantization."""
        q, s = dynamic_block_quantize_fp8(activations.to(torch.bfloat16),
                                          block_size=FP8_ACT_BLOCK_SIZE)
        return dequantize_block_fp8(q, s, torch.float32, hidden)

    # What the kernel would multiply if the gather had not been quantized.
    reference = kernel_lhs(tensor)

    blocked_q, blocked_s = dynamic_block_quantize_fp8(
        tensor, block_size=FP8_ACT_BLOCK_SIZE)
    through_blocks = kernel_lhs(
        dequantize_block_fp8(blocked_q, blocked_s, torch.bfloat16, hidden))
    assert torch.equal(through_blocks, reference)

    row_q, row_s = dynamic_row_quantize_fp8(tensor)
    through_row = kernel_lhs(_dequantize(row_q, row_s).to(torch.bfloat16))
    assert not torch.equal(through_row, reference)


def test_block_quantize_zeroes_only_the_block_it_cannot_scale():
    """A bad block must not take the rest of the row down with it."""
    tensor = torch.ones(1, 2 * FP8_ACT_BLOCK_SIZE)
    tensor[0, 3] = float("inf")

    quantized, scale = dynamic_block_quantize_fp8(
        tensor, block_size=FP8_ACT_BLOCK_SIZE)
    restored = dequantize_block_fp8(quantized, scale, torch.float32)

    assert torch.isfinite(restored).all()
    assert scale[0, 0].item() == 0.0
    assert torch.equal(restored[0, :FP8_ACT_BLOCK_SIZE],
                       torch.zeros(FP8_ACT_BLOCK_SIZE))
    torch.testing.assert_close(restored[0, FP8_ACT_BLOCK_SIZE:],
                               torch.ones(FP8_ACT_BLOCK_SIZE))


def test_row_quantize_is_the_single_block_case():
    """`dynamic_row_quantize_fp8` is a thin alias, not a second implementation."""
    torch.manual_seed(0)
    tensor = torch.randn(8, 256, dtype=torch.bfloat16)

    row_q, row_s = dynamic_row_quantize_fp8(tensor)
    blk_q, blk_s = dynamic_block_quantize_fp8(tensor, block_size=256)

    assert torch.equal(row_q.to(torch.float32), blk_q.to(torch.float32))
    assert torch.equal(row_s, blk_s)


@pytest.mark.parametrize("block_size", [0, -1])
def test_block_quantize_rejects_a_non_positive_block(block_size):
    with pytest.raises(ValueError, match="block_size must be positive"):
        dynamic_block_quantize_fp8(torch.zeros(2, 8), block_size=block_size)
