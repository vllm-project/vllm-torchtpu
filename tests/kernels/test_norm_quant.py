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
"""`add_rms_norm_quantize` against the unfused ops it replaces.

The values run in Pallas interpret mode on CPU; the lowering half checks that
the kernel compiles for an abstract TPU7x device.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.norm_quant import add_rms_norm_quantize
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

FP8 = jnp.float8_e4m3fn
EPS = 1e-6


def _rms_norm(x, w):
    """The TPU rms_norm provider: f32 math, rounded to the input dtype."""
    x32 = x.astype(jnp.float32)
    var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(var + EPS) * w.astype(jnp.float32)).astype(x.dtype)


def _reference(x, residual, w_post, scalar, w_next):
    h = (
        _rms_norm(x, w_post).astype(jnp.float32) + residual.astype(jnp.float32)
    ).astype(x.dtype)
    h = (h.astype(jnp.float32) * scalar.astype(jnp.float32)).astype(x.dtype)
    q, scale = quantize_tensor(_rms_norm(h, w_next), FP8)
    return h, q, scale


@pytest.mark.parametrize(
    "tokens,hidden,scalar", [(64, 512, 1.0), (128, 5376, 0.75), (48, 256, 2.0)]
)
def test_matches_the_unfused_ops(tokens, hidden, scalar):
    rng = np.random.default_rng(tokens)
    x = jnp.asarray(rng.standard_normal((tokens, hidden)) * 3, jnp.bfloat16)
    residual = jnp.asarray(rng.standard_normal((tokens, hidden)) * 10, jnp.bfloat16)
    w_post = jnp.asarray(rng.standard_normal(hidden), jnp.float32)
    w_next = jnp.asarray(rng.standard_normal(hidden), jnp.float32)
    s = jnp.asarray([scalar], jnp.bfloat16)
    got = add_rms_norm_quantize(
        x, residual, w_post, w_next, s, eps=EPS, quant_dtype=FP8, interpret=True
    )
    for a, b in zip(got, _reference(x, residual, w_post, s, w_next)):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(
            np.asarray(a.astype(jnp.float32)), np.asarray(b.astype(jnp.float32))
        )


def test_bytes_out_are_the_fp8_rows_bytes():
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.standard_normal((32, 256)) * 3, jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal(256), jnp.float32)
    s = jnp.ones(1, jnp.bfloat16)
    kwargs = dict(eps=EPS, quant_dtype=FP8, interpret=True)
    h, q, scale = add_rms_norm_quantize(x, x, w, w, s, **kwargs)
    h_b, q_b, scale_b = add_rms_norm_quantize(x, x, w, w, s, bytes_out=True, **kwargs)
    assert q_b.dtype == jnp.int8 and q_b.shape == q.shape
    np.testing.assert_array_equal(np.asarray(q_b), np.asarray(q.view(jnp.int8)))
    np.testing.assert_array_equal(
        np.asarray(h_b.view(jnp.uint16)), np.asarray(h.view(jnp.uint16))
    )
    np.testing.assert_array_equal(np.asarray(scale_b), np.asarray(scale))


def test_a_zero_row_quantizes_at_scale_one():
    x = jnp.zeros((16, 256), jnp.bfloat16)
    w = jnp.ones(256, jnp.float32)
    _, q, scale = add_rms_norm_quantize(
        x, x, w, w, jnp.ones(1, jnp.bfloat16), eps=EPS, quant_dtype=FP8, interpret=True
    )
    assert float(jnp.abs(q.astype(jnp.float32)).max()) == 0.0
    assert np.all(np.asarray(scale) == 1.0)


def test_lowers_for_tpu7x():
    device = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
    rows = jax.ShapeDtypeStruct((4096, 5376), jnp.bfloat16)
    weight = jax.ShapeDtypeStruct((5376,), jnp.float32)
    scalar = jax.ShapeDtypeStruct((1,), jnp.bfloat16)
    with jax.sharding.use_abstract_mesh(mesh):
        text = (
            jax.jit(lambda *a: add_rms_norm_quantize(*a, eps=EPS, quant_dtype=FP8))
            .trace(rows, rows, weight, weight, scalar)
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )
    assert "add_rms_norm_quantize" in text


def test_block_rows_need_a_power_of_two():
    x = jnp.zeros((64, 256), jnp.bfloat16)
    w = jnp.ones(256, jnp.float32)
    with pytest.raises(ValueError, match="power of two"):
        add_rms_norm_quantize(
            x, x, w, w, None, eps=EPS, quant_dtype=FP8, max_block_rows=96
        )
