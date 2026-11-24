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
"""Tests for the power-of-two (UE8M0) MoE FP4 block scale.

The scale is what `MOE_REQUANTIZE_SCALE_MODE=pow2` selects instead of the
default `amax / 6`: the smallest power of two at or above it. It keeps the
kernel contract (FP32 scale of the same shape), so the only thing under test
here is that the rounding is exact and minimal, and that requantizing with it
leaves the scale layout untouched.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.core.fused_moe_gmm import (
    ceil_pow2_scale,
    quantize_to_native_fp4_kmajor,
)

# Exponent range of normal FP32; the bit trick rounds subnormals up to the
# smallest normal power of two, which is documented and never hit by weights.
_MIN_EXP = -126
_MAX_EXP = 127


def _sample_scales() -> np.ndarray:
    """Normal-range scales, biased to the exact powers and their neighbours."""
    values = []
    for exponent in range(_MIN_EXP, _MAX_EXP, 5):
        power = np.float32(2.0) ** np.float32(exponent)
        for mantissa in (
            np.float32(1.0),
            np.nextafter(np.float32(1.0), np.float32(0)),
            np.nextafter(np.float32(1.0), np.float32(2)),
            np.float32(1.5),
            np.float32(1.9999999),
        ):
            value = np.float32(power * mantissa)
            if np.isfinite(value):
                values.append(value)
    return np.asarray(values, dtype=np.float32)


def test_ceil_pow2_scale_is_the_smallest_power_of_two_at_or_above():
    scales = _sample_scales()
    got = np.asarray(ceil_pow2_scale(jnp.asarray(scales)))
    bits = got.view(np.uint32)

    # A power of two has only an exponent field.
    assert np.all(got > 0)
    assert np.all((bits & np.uint32(0x7FFFFF)) == 0)
    # Minimal: scale <= result < 2 * scale, and never below the input.
    assert np.all(got >= scales)
    assert np.all(got < 2 * scales)
    # Matches the float64 oracle where the oracle is itself exact.
    expected = np.exp2(np.ceil(np.log2(scales.astype(np.float64))))
    assert np.all(got.astype(np.float64) == expected)


def test_ceil_pow2_scale_keeps_an_all_zero_block_at_zero():
    got = np.asarray(ceil_pow2_scale(jnp.zeros((4,), jnp.float32)))
    assert np.all(got == 0)


def _skip_without_float4():
    try:
        jnp.asarray([0.0, 1.0], jnp.float32).astype(jnp.float4_e2m1fn)
    except Exception as exc:  # pragma: no cover - device dependent
        pytest.skip(f"jax float4_e2m1fn unavailable: {exc}")


def test_pow2_requant_keeps_the_scale_layout_and_stays_a_power_of_two():
    _skip_without_float4()
    rng = np.random.default_rng(0)
    # 2560 = 2304 + 256, the padded V4.1 w2 contraction at block 512.
    weight = rng.standard_normal((2560, 4)).astype(np.float32)

    _, scale_fp32 = quantize_to_native_fp4_kmajor(weight, block=512)
    _, scale_pow2 = quantize_to_native_fp4_kmajor(weight, block=512, scale_mode="pow2")

    assert scale_pow2.shape == scale_fp32.shape == (5, 1, 4)
    assert scale_pow2.dtype == scale_fp32.dtype == jnp.float32

    got = np.asarray(scale_pow2)
    bits = got.view(np.uint32)
    assert np.all((bits & np.uint32(0x7FFFFF)) == 0)
    assert np.all(got >= np.asarray(scale_fp32))
    assert np.all(got < 2 * np.asarray(scale_fp32))


def test_pow2_requant_is_a_no_op_when_the_block_scale_is_already_a_power_of_two():
    _skip_without_float4()
    # Every block is grid * 2^-3 with amax 6 * 2^-3, so amax / 6 == 2^-3 and
    # both modes must produce the same scale and the same codes.
    grid = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -6.0, -3.0])
    block = np.resize(grid, 512) * np.float32(2.0**-3)
    weight = np.stack([block, block[::-1].copy(), -block, block * 0.5], axis=1)

    codewords_fp32, scale_fp32 = quantize_to_native_fp4_kmajor(weight, block=512)
    codewords_pow2, scale_pow2 = quantize_to_native_fp4_kmajor(
        weight, block=512, scale_mode="pow2"
    )

    assert np.array_equal(np.asarray(scale_fp32), np.asarray(scale_pow2))
    assert np.array_equal(np.asarray(codewords_fp32), np.asarray(codewords_pow2))
    # Every column's amax / 6 is already a power of two, so the pow2 mode must
    # not move it: 2^-3 for the three columns with amax 6 * 2^-3, 2^-4 for the
    # halved one.
    expected_scales = np.float32([[0.125, 0.125, 0.125, 0.0625]])
    assert np.all(np.asarray(scale_pow2)[..., 0, :] == expected_scales)


@pytest.mark.parametrize("pack", [False, True])
def test_four_over_six_selects_per_block_and_column(pack):
    _skip_without_float4()
    rows = np.tile([[4.0, 6.0, 0.0], [1.5, 1.0, 0.0]], (256, 1))
    w = np.concatenate([rows, rows[:, [1, 0, 2]]]).astype(np.float32)
    codes, scales = quantize_to_native_fp4_kmajor(
        w, block=512, pack=pack, scale_mode="four_over_six"
    )
    np.testing.assert_array_equal(scales, [[[1.0, 1.0, 0.0]], [[1.0, 1.0, 0.0]]])
    expected = jnp.asarray(w).astype(jnp.float4_e2m1fn)
    if pack:
        from jax import lax

        pairs = expected.reshape(512, 2, 3).swapaxes(-1, -2)
        expected = lax.bitcast_convert_type(pairs, jnp.uint8)
    np.testing.assert_array_equal(codes, expected)
