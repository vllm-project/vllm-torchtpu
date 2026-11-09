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
"""Tests for the opt-in optimization_barrier in quantize_block."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.quantized_matmul.util import (
    get_max_min,
    quantize_block,
    quantize_tensor,
)


def _ref_quantize_block_without_barrier(data, axis, target_dtype):
    abs_max = jnp.max(jnp.abs(data), axis=axis, keepdims=True)
    dtype_max, dtype_min = get_max_min(target_dtype)
    scale = abs_max / dtype_max
    scale = jnp.where(scale == 0, 1.0, scale)
    if jnp.issubdtype(target_dtype, jnp.floating):
        data_q = (data / scale).clip(dtype_min, dtype_max).astype(target_dtype)
    else:
        data_q = jnp.round(data / scale).astype(target_dtype)
    return data_q, scale


def _make_input():
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((16, 256), dtype=np.float32) * 12.0
    raw[0, :] = 0.0  # all-zero row exercises the `scale == 0` branch
    raw[1, 0] = 1000.0  # values beyond the FP8 range exercise clipping
    raw[1, 1] = -1000.0
    return jnp.asarray(raw, dtype=jnp.bfloat16)


def _assert_bits_equal(actual, expected):
    np.testing.assert_array_equal(
        np.asarray(actual.astype(jnp.float32)),
        np.asarray(expected.astype(jnp.float32)),
    )


@pytest.mark.parametrize("target_dtype", [jnp.float8_e4m3fn, jnp.int8])
def test_quantize_tensor_fences_reduction_and_preserves_bits(target_dtype):
    with jax.default_device(jax.devices("cpu")[0]):
        x = _make_input()

        jaxpr = jax.make_jaxpr(lambda inp: quantize_tensor(inp, target_dtype))(x)
        assert "optimization_barrier" in str(jaxpr)

        q_actual, s_actual = quantize_tensor(x, target_dtype)
        q_ref, s_ref = _ref_quantize_block_without_barrier(x, -1, target_dtype)
        _assert_bits_equal(q_actual, q_ref)
        np.testing.assert_array_equal(
            np.asarray(s_actual), np.asarray(s_ref.astype(jnp.float32))
        )


def test_block_quantize_tensor_fences_reduction_and_preserves_bits():
    block_size = 128
    with jax.default_device(jax.devices("cpu")[0]):
        x = _make_input()

        jaxpr = jax.make_jaxpr(
            lambda inp: quantize_tensor(inp, jnp.float8_e4m3fn, block_size=block_size)
        )(x)
        assert "optimization_barrier" in str(jaxpr)

        q_actual, s_actual = quantize_tensor(
            x, jnp.float8_e4m3fn, block_size=block_size
        )
        q_ref, s_ref = _ref_quantize_block_without_barrier(
            x.reshape(x.shape[0], -1, block_size), -1, jnp.float8_e4m3fn
        )
        _assert_bits_equal(q_actual, q_ref.reshape(x.shape))
        np.testing.assert_array_equal(
            np.asarray(s_actual),
            np.asarray(s_ref.transpose(1, 2, 0).astype(jnp.float32)),
        )


def test_quantize_block_has_no_barrier_by_default():
    with jax.default_device(jax.devices("cpu")[0]):
        x = _make_input()
        jaxpr = jax.make_jaxpr(lambda inp: quantize_block(inp, -1, jnp.float8_e4m3fn))(
            x
        )
        assert "optimization_barrier" not in str(jaxpr)

        q_actual, s_actual = quantize_block(x, -1, jnp.float8_e4m3fn)
        q_ref, s_ref = _ref_quantize_block_without_barrier(x, -1, jnp.float8_e4m3fn)
        _assert_bits_equal(q_actual, q_ref)
        np.testing.assert_array_equal(np.asarray(s_actual), np.asarray(s_ref))


def test_quantize_block_default_lowers_inside_pallas_tpu_kernel():
    """Pallas TPU (Mosaic) lowering has no rule for `optimization_barrier`.

    Kernels such as `blockwise_kernel.py` and `moe_tp_decode` call
    `quantize_block` inside a `pallas_call`, so its default must stay
    barrier-free. Lowering for the "tpu" platform runs on a CPU-only host.
    """

    def kernel(x_ref, q_ref, s_ref):
        q, s = quantize_block(x_ref[...].astype(jnp.float32), 1, jnp.float8_e4m3fn)
        q_ref[...] = q
        s_ref[...] = jnp.broadcast_to(s, s_ref.shape)

    def fn(x):
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct(x.shape, jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((x.shape[0], 128), jnp.float32),
            ),
        )(x)

    x = jax.ShapeDtypeStruct((32, 512), jnp.bfloat16)
    lowered = jax.jit(fn).trace(x).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()
