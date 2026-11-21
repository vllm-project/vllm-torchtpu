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
"""Fused EP MoE v3's per-row fp8 quantization (``rowquant``), run on CPU
against a numpy reference written out here.

The token exchange sends each token's row as fp8 with one f32 scale per row
(or per block of the row), and the expert FFN requantizes its intermediate
the same way: the scale is the row's absolute maximum over 448, rounded to
the scale dtype first, and a row whose maximum is zero or not finite is sent
as zeros with scale zero rather than with an infinite scale.
"""

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v3 import rowquant

pytestmark = pytest.mark.cpu_test

FP8 = ml_dtypes.float8_e4m3fn


def _rows(n=16, width=256, seed=0):
    rng = np.random.default_rng(seed)
    rows = (rng.standard_normal((n, width)) * rng.uniform(0.01, 30, (n, 1))).astype(
        np.float32
    )
    rows[3] = 0.0  # no usable scale: sent as zeros
    rows[5, 17] = np.inf  # not finite: sent as zeros
    rows[7, 200] = np.nan  # not finite: sent as zeros
    return np.asarray(jnp.asarray(rows, jnp.bfloat16).astype(jnp.float32))


def _reference(rows, scale_dtype=np.float32):
    """(the scale each row should get, its maximum over 448 rounded to
    `scale_dtype`, as f32 [n, 1]; zero where that is not finite and
    positive)."""
    amax = np.abs(rows).max(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", over="ignore"):
        scale = (amax / np.float32(448.0)).astype(scale_dtype).astype(np.float32)
    scalable = np.isfinite(scale) & (scale > 0)
    return np.where(scalable, scale, np.float32(0.0)).astype(np.float32)


def _quantized_by(rows, scale):
    """The fp8 rows (as f32) a scale gives: each row times the scale's f32
    inverse, cast to fp8; a row without a positive scale is zero."""
    scalable = scale > 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inverse = np.where(scalable, np.float32(1.0) / scale, np.float32(0.0))
        product = (rows * inverse).astype(np.float32)
    return np.where(scalable, product, np.float32(0.0)).astype(FP8).astype(np.float32)


def _check(got_q, got_scale, rows, scale_dtype=np.float32):
    """The scale is the row's maximum over 448 (the f32 division may round
    differently in the compiled program, so to two f32 ulps, or one step of
    a bf16 scale), and the fp8 values are exactly the ones that scale gives."""
    want = _reference(rows, scale_dtype)
    rtol = 2.0**-7 if scale_dtype != np.float32 else 2.0**-22
    np.testing.assert_allclose(got_scale, want, rtol=rtol, atol=0)
    np.testing.assert_array_equal(got_scale == 0, want == 0)
    np.testing.assert_array_equal(got_q, _quantized_by(rows, got_scale))


def _quantize(rows, **kwargs):
    q, scales = rowquant.quantize_rows(jnp.asarray(rows, jnp.bfloat16), **kwargs)
    assert q.dtype == jnp.float8_e4m3fn and scales.dtype == jnp.float32
    return np.asarray(q.astype(jnp.float32)), np.asarray(scales)


@pytest.mark.parametrize(
    "scale_dtype", [jnp.float32, jnp.bfloat16], ids=["f32_scale", "bf16_scale"]
)
def test_one_scale_per_row_matches_the_reference(scale_dtype):
    rows = _rows()
    got_q, got_scale = _quantize(rows, scale_dtype=scale_dtype)
    ref_dtype = ml_dtypes.bfloat16 if scale_dtype == jnp.bfloat16 else np.float32
    assert got_scale.shape == (rows.shape[0], 1)
    _check(got_q, got_scale, rows, ref_dtype)
    if scale_dtype == jnp.bfloat16:  # the scale used is the rounded one
        rounded = got_scale.astype(ml_dtypes.bfloat16).astype(np.float32)
        np.testing.assert_array_equal(got_scale, rounded)


def test_rows_without_a_usable_scale_go_out_as_zeros():
    """A zero row and rows holding inf or NaN: scale zero and every value
    zero, never an infinite scale or a NaN value on the wire."""
    got_q, got_scale = _quantize(_rows())
    for r in (3, 5, 7):
        assert got_scale[r, 0] == 0.0
        assert not got_q[r].any()
    assert np.isfinite(got_q).all() and np.isfinite(got_scale).all()
    # Every other row has a positive scale and reaches the fp8 maximum.
    others = [r for r in range(got_q.shape[0]) if r not in (3, 5, 7)]
    assert (got_scale[others] > 0).all()
    np.testing.assert_array_equal(np.abs(got_q[others]).max(axis=-1), 448.0)


def test_one_scale_per_block_quantizes_each_block_on_its_own():
    """With a block, each block of a row is quantized as a row of its own."""
    rows, block = _rows(), 128
    got_q, got_scale = _quantize(rows, block=block)
    assert got_scale.shape == (rows.shape[0], rows.shape[1] // block)
    for b in range(rows.shape[1] // block):
        cols = slice(b * block, (b + 1) * block)
        _check(got_q[:, cols], got_scale[:, b : b + 1], rows[:, cols])
    # Row 5's inf sits in block 0 and row 7's NaN in block 1: only that block
    # of the row goes out as zeros.
    assert got_scale[5, 0] == 0.0 and got_scale[5, 1] > 0
    assert got_scale[7, 1] == 0.0 and got_scale[7, 0] > 0


def test_a_block_that_does_not_divide_the_row_is_refused():
    with pytest.raises(ValueError, match="whole number of 96-value blocks"):
        rowquant.quantize_rows(jnp.zeros((4, 256), jnp.bfloat16), block=96)
