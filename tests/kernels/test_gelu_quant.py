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
"""`gelu_tanh_and_mul_quantize` against GeluAndMul("tanh") then quantize_tensor.

The values run in Pallas interpret mode on CPU; the lowering half checks that
the kernel compiles for an abstract TPU7x device.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gelu_quant import block_rows, gelu_tanh_and_mul_quantize
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

FP8 = jnp.float8_e4m3fn


def _reference(x):
    """torch's tanh GELU decomposition in f32, rounded once, then quantized."""
    d = x.shape[1] // 2
    gate, up = x[:, :d].astype(jnp.float32), x[:, d:].astype(jnp.float32)
    inner = math.sqrt(2 / math.pi) * (gate + 0.044715 * gate * gate * gate)
    y = (0.5 * gate * (1 + jnp.tanh(inner)) * up).astype(x.dtype)
    return quantize_tensor(y, FP8)


@pytest.mark.parametrize("tokens,d", [(16, 256), (128, 1024), (192, 640)])
def test_matches_the_unfused_ops(tokens, d):
    rng = np.random.default_rng(tokens)
    x = jnp.asarray(rng.standard_normal((tokens, 2 * d)) * 3, jnp.bfloat16)
    x = x.at[1].set(0.0)  # an all-zero row quantizes at scale 1
    q, scale = gelu_tanh_and_mul_quantize(x, FP8, interpret=True)
    ref_q, ref_scale = _reference(x)
    assert q.dtype == FP8 and q.shape == (tokens, d)
    assert scale.dtype == jnp.float32 and scale.shape == (tokens, 1)
    np.testing.assert_array_equal(np.asarray(scale), np.asarray(ref_scale))
    assert float(scale[1, 0]) == 1.0
    got = np.asarray(q.astype(jnp.float32))
    want = np.asarray(ref_q.astype(jnp.float32))
    # Same arithmetic up to the tanh lowering: nearly every value identical,
    # the rest one fp8 step apart.
    assert (got == want).mean() > 0.999
    step = np.abs(want) * 2.0**-3 + 2.0**-9
    assert (np.abs(got - want) <= step).all()


@pytest.mark.parametrize("tokens,d", [(16, 256), (128, 1024)])
def test_bf16_math_stays_within_an_fp8_step(tokens, d):
    rng = np.random.default_rng(tokens + 1)
    x = jnp.asarray(rng.standard_normal((tokens, 2 * d)) * 3, jnp.bfloat16)
    x = x.at[1].set(0.0)
    q, scale = gelu_tanh_and_mul_quantize(x, FP8, bf16_math=True, interpret=True)
    ref_q, ref_scale = _reference(x)
    assert q.dtype == FP8 and scale.dtype == jnp.float32
    assert float(scale[1, 0]) == 1.0
    # The row maximum is the bf16 product's, a few bf16 steps from the f32 one.
    np.testing.assert_allclose(np.asarray(scale), np.asarray(ref_scale), rtol=2**-6)
    got = np.asarray(q.astype(jnp.float32) * scale)
    want = np.asarray(ref_q.astype(jnp.float32) * ref_scale)
    tol = np.abs(want) * 2.0**-3 + np.asarray(ref_scale) * 2.0**-6
    assert (np.abs(got - want) <= tol).all()
    assert (
        np.asarray(q.astype(jnp.float32)) == np.asarray(ref_q.astype(jnp.float32))
    ).mean() > 0.8


def test_bf16_math_refuses_f32_rows():
    with pytest.raises(ValueError, match="bf16"):
        gelu_tanh_and_mul_quantize(
            jnp.zeros((16, 512), jnp.float32), FP8, bf16_math=True, interpret=True
        )


def test_row_max_quantizes_to_the_dtype_max():
    x = jnp.zeros((16, 256), jnp.bfloat16).at[0, 0].set(6.0).at[0, 128].set(-6.0)
    q, scale = gelu_tanh_and_mul_quantize(x, FP8, interpret=True)
    assert float(jnp.abs(q.astype(jnp.float32)).max()) == 448.0


def test_block_rows_tile_every_bucket():
    # Gemma4-31B TP2's intermediate width per rank.
    assert [block_rows(n, 10752) for n in (16, 48, 96, 864, 16384)] == [
        16,
        16,
        32,
        32,
        256,
    ]


def test_block_rows_fit_the_vmem_limit_at_wider_ranks():
    # TP1's 21504 columns: 256-row blocks would take ~52 MiB of buffers.
    assert block_rows(16384, 21504) == 128


def test_refuses_a_partial_lane_half_width():
    with pytest.raises(ValueError, match="multiple of 128"):
        gelu_tanh_and_mul_quantize(
            jnp.zeros((16, 2 * 1056), jnp.bfloat16), FP8, interpret=True
        )


def test_lowers_for_tpu7x():
    device = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
    x = jax.ShapeDtypeStruct((1024, 2 * 10752), jnp.bfloat16)
    with jax.sharding.use_abstract_mesh(mesh):
        text = (
            jax.jit(lambda a: gelu_tanh_and_mul_quantize(a, FP8))
            .trace(x)
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )
    assert "gelu_tanh_and_mul_quantize" in text


def test_block_rows_need_a_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        block_rows(64, 10752, 96)
