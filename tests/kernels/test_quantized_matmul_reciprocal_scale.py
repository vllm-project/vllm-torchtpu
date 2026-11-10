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
"""The quantized-matmul quantize helpers in both of their modes.

By default they keep main's arithmetic (divide by the scale); with
RECIPROCAL_SCALE they multiply by its reciprocal, and quantize_array also
guards an all-zero row's scale and clips FP8 before the cast. Both modes are
checked bit for bit against the formulas written out here, on rows that
include an all-zero row and values near the FP8 maximum.

XLA compiles a division by a per-row scale as a multiply by the scale's
reciprocal, so on CPU the two forms of the division give the same bits; only
quantize_array's zero guard and clip show in them. The traced program is what
a Pallas kernel lowers, so the tests also check that each mode traces its own
form there.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul import kernel, util

pytestmark = pytest.mark.cpu_test

FP8 = jnp.float8_e4m3fn


@pytest.fixture
def mode(monkeypatch):
    def set_mode(reciprocal):
        monkeypatch.setattr(util, "RECIPROCAL_SCALE", reciprocal)

    return set_mode


def _rows(seed=0):
    """Random rows, an all-zero row, values at and past the FP8 maximum, and
    rows of FP8 rounding midpoints times the row's scale, where an IEEE
    division by the scale and a multiply by its reciprocal round to different
    FP8 values (XLA compiles both as the multiply; see the module docstring)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((8, 256)).astype(np.float32) * 3.0
    x[1] = 0.0  # an all-zero row
    x[2, :4] = [447.9, -447.9, 1e-3, 3e3]  # near and far past the FP8 maximum
    codes = jnp.arange(1, 127, dtype=jnp.uint8)  # positive finite FP8 values
    values = np.asarray(codes.view(FP8).astype(jnp.float32))
    midpoints = (values[:-1] + values[1:]) / 2
    rows = []
    for r in range(56):
        scale = np.float32(1.0 + r / 97.0)
        row = np.zeros(256, np.float32)
        row[0] = np.float32(448.0) * scale  # the row's abs max: scale * 448
        row[1 : 1 + midpoints.size] = midpoints * scale
        rows.append(row)
    return jnp.asarray(np.concatenate([x, np.stack(rows)]))


def _bits(a):
    return np.asarray(a).view(np.uint8 if a.dtype == FP8 else np.uint32)


# ---- quantize_array (per row, inside the quantized-matmul kernel) ----


def _array_main(x, x_abs_max, dtype_max):
    scale = jnp.transpose(x_abs_max / dtype_max)
    return (x / scale).astype(FP8), scale.astype(jnp.float32)


def _array_reciprocal(x, x_abs_max, dtype_max):
    scale = jnp.transpose(x_abs_max * (1.0 / dtype_max))
    scale = jnp.where(scale == 0, 1.0, scale)
    x_s = jnp.clip(x * (jnp.float32(1.0) / scale), -dtype_max, dtype_max)
    return x_s.astype(FP8), scale.astype(jnp.float32)


@pytest.mark.parametrize("reciprocal", [False, True], ids=["main", "reciprocal"])
def test_quantize_array_matches_its_mode(mode, reciprocal):
    mode(reciprocal)
    x = _rows()
    x_abs_max = jnp.transpose(jnp.max(jnp.abs(x), axis=1, keepdims=True))
    dtype_max = float(jnp.finfo(FP8).max)
    want = (_array_reciprocal if reciprocal else _array_main)(x, x_abs_max, dtype_max)
    got = kernel.quantize_array(x, x_abs_max, FP8)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(_bits(g), _bits(w))


def test_the_array_modes_differ_on_these_rows():
    """Without rows where the formulas disagree, the bit test of each mode
    could not tell which branch ran (here: the all-zero row, which main's
    arithmetic turns into NaN)."""
    x = _rows()
    x_abs_max = jnp.transpose(jnp.max(jnp.abs(x), axis=1, keepdims=True))
    dtype_max = float(jnp.finfo(FP8).max)
    a = _array_main(x, x_abs_max, dtype_max)[0]
    b = _array_reciprocal(x, x_abs_max, dtype_max)[0]
    assert not np.array_equal(_bits(a), _bits(b))


def _full_shape_ops(fn, x):
    """The primitives of fn's traced program whose output has x's shape.

    A fresh function each time: JAX caches a trace per function, and the mode
    is read while tracing."""
    jaxpr = jax.make_jaxpr(lambda v: fn(v))(x).jaxpr
    return {
        eqn.primitive.name
        for eqn in jaxpr.eqns
        for out in eqn.outvars
        if out.aval.shape == x.shape
    }


def _quantize_array(x):
    x_abs_max = jnp.transpose(jnp.max(jnp.abs(x), axis=1, keepdims=True))
    return kernel.quantize_array(x, x_abs_max, FP8)


def _quantize_block(x):
    return util.quantize_block(x, axis=1, target_dtype=FP8)


@pytest.mark.parametrize(
    "fn", [_quantize_array, _quantize_block], ids=["array", "block"]
)
@pytest.mark.parametrize("reciprocal", [False, True], ids=["main", "reciprocal"])
def test_each_mode_traces_its_own_division(mode, fn, reciprocal):
    """Main's arithmetic divides the data by the scale; the reciprocal form
    divides only the per-row scale and multiplies the data by it."""
    mode(reciprocal)
    ops = _full_shape_ops(fn, _rows())
    if reciprocal:
        assert "mul" in ops and "div" not in ops, ops
    else:
        assert "div" in ops and "mul" not in ops, ops


def test_the_default_is_mains_arithmetic():
    assert util.RECIPROCAL_SCALE is False


# ---- quantize_block (per block, the activation and weight quantizer) ----


def _block_main(x, dtype_max, dtype_min):
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True) / dtype_max
    scale = jnp.where(scale == 0, 1.0, scale)
    return (x / scale).clip(dtype_min, dtype_max).astype(FP8), scale


def _block_reciprocal(x, dtype_max, dtype_min):
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True) / dtype_max
    scale = jnp.where(scale == 0, 1.0, scale)
    q = jnp.clip(x * (jnp.float32(1.0) / scale), dtype_min, dtype_max)
    return q.astype(FP8), scale


@pytest.mark.parametrize("reciprocal", [False, True], ids=["main", "reciprocal"])
def test_quantize_block_matches_its_mode(mode, reciprocal):
    mode(reciprocal)
    x = _rows(seed=1)
    dtype_max, dtype_min = util.get_max_min(FP8)
    want = (_block_reciprocal if reciprocal else _block_main)(x, dtype_max, dtype_min)
    got = util.quantize_block(x, axis=1, target_dtype=FP8)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(_bits(g), _bits(w))


def test_set_reciprocal_scale_switches_the_mode(monkeypatch):
    monkeypatch.setattr(util, "RECIPROCAL_SCALE", False)
    util.set_reciprocal_scale(True)
    assert util.RECIPROCAL_SCALE is True
    util.set_reciprocal_scale(False)
    assert util.RECIPROCAL_SCALE is False
