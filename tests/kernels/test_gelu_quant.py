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
"""Tests for fused tanh GeGLU and per-token FP8 quantization."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gelu_quant import block_rows, gelu_and_mul_quantize
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

FP8 = jnp.float8_e4m3fn


def _reference(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    orig_prefix = x.shape[:-1]
    d = x.shape[-1] // 2
    x_2d = x.reshape(-1, x.shape[-1])
    gate = x_2d[:, :d].astype(jnp.float32)
    up = x_2d[:, d:].astype(jnp.float32)
    inner = math.sqrt(2.0 / math.pi) * (gate + 0.044715 * gate * gate * gate)
    y = (0.5 * gate * (1.0 + jnp.tanh(inner)) * up).astype(x.dtype)
    q_2d, s_2d = quantize_tensor(y, FP8)
    return q_2d.reshape(*orig_prefix, d), s_2d.reshape(*orig_prefix, 1)


@pytest.mark.parametrize(
    "shape",
    [(16, 512), (128, 2048), (192, 1280), (2, 8, 512)],
)
def test_gelu_and_mul_quantize_interpret_matches_reference(shape):
    rng = np.random.default_rng(sum(shape))
    with jax.default_device(jax.devices("cpu")[0]):
        x = jnp.asarray(rng.standard_normal(shape) * 3.0, jnp.bfloat16)
        x_flat = x.reshape(-1, shape[-1]).at[1].set(0.0)
        x = x_flat.reshape(shape)

        q, scale = gelu_and_mul_quantize(x, FP8, interpret=True)
        ref_q, ref_scale = _reference(x)

    d = shape[-1] // 2
    assert q.dtype == FP8 and q.shape == (*shape[:-1], d)
    assert scale.dtype == jnp.float32 and scale.shape == (*shape[:-1], 1)
    assert float(scale.reshape(-1, 1)[1, 0]) == 1.0
    np.testing.assert_allclose(np.asarray(scale), np.asarray(ref_scale), rtol=2**-6)
    got = np.asarray(q.astype(jnp.float32) * scale)
    want = np.asarray(ref_q.astype(jnp.float32) * ref_scale)
    tol = np.abs(want) * 2.0**-3 + np.asarray(ref_scale) * 2.0**-6
    assert (np.abs(got - want) <= tol).all()


def test_subnormal_row_with_zeros_avoids_nan():
    with jax.default_device(jax.devices("cpu")[0]):
        x = jnp.zeros((16, 256), jnp.bfloat16)
        x = x.at[0, 0].set(jnp.bfloat16(1.0)).at[0, 128].set(jnp.bfloat16(1e-37))
        q, scale = gelu_and_mul_quantize(x, FP8, interpret=True)
        assert not jnp.isnan(q.astype(jnp.float32)).any()
        assert not jnp.isnan(scale).any()


def test_row_max_saturates_fp8_max():
    with jax.default_device(jax.devices("cpu")[0]):
        x = jnp.zeros((16, 256), jnp.bfloat16).at[0, 0].set(6.0).at[0, 128].set(-6.0)
        q, _ = gelu_and_mul_quantize(x, FP8, interpret=True)
        assert float(jnp.abs(q.astype(jnp.float32)).max()) == 448.0


def test_block_rows_scales_with_vmem_budget():
    vmem_64m = 64 * 1024 * 1024
    vmem_32m = 32 * 1024 * 1024
    assert [
        block_rows(n, 10752, vmem_limit_bytes=vmem_64m)
        for n in (16, 48, 96, 864, 16384)
    ] == [16, 16, 32, 32, 256]
    assert block_rows(16384, 21504, vmem_limit_bytes=vmem_64m) == 128
    assert block_rows(16384, 10752, vmem_limit_bytes=vmem_32m) == 128


def test_validation_rejects_unsupported_inputs():
    with jax.default_device(jax.devices("cpu")[0]):
        with pytest.raises(ValueError, match="bfloat16"):
            gelu_and_mul_quantize(
                jnp.zeros((16, 256), jnp.float32), FP8, interpret=True
            )
        with pytest.raises(ValueError, match="multiple of 256"):
            gelu_and_mul_quantize(
                jnp.zeros((16, 2 * 1056), jnp.bfloat16), FP8, interpret=True
            )
        with pytest.raises(ValueError, match="multiple of 8"):
            gelu_and_mul_quantize(
                jnp.zeros((12, 256), jnp.bfloat16), FP8, interpret=True
            )
    with pytest.raises(ValueError, match="power of two"):
        block_rows(64, 1024, max_rows=96, vmem_limit_bytes=32 * 1024 * 1024)


def test_lowers_for_tpu7x():
    device = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
    x = jax.ShapeDtypeStruct((1024, 2 * 10752), jnp.bfloat16)
    with jax.sharding.use_abstract_mesh(mesh):
        text = (
            jax.jit(lambda a: gelu_and_mul_quantize(a, FP8))
            .trace(x)
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )
    assert "gelu_and_mul_quantize" in text


@pytest.mark.tpu_test
@pytest.mark.parametrize("shape", [(64, 1024), (4, 16, 512)])
def test_gelu_and_mul_quantize_on_tpu(shape):
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.standard_normal(shape) * 3.0, jnp.bfloat16)
    q, scale = gelu_and_mul_quantize(x, FP8, interpret=False)
    ref_q, ref_scale = _reference(x)

    assert q.dtype == FP8 and q.shape == (*shape[:-1], shape[-1] // 2)
    assert scale.dtype == jnp.float32 and scale.shape == (*shape[:-1], 1)
    np.testing.assert_allclose(np.asarray(scale), np.asarray(ref_scale), rtol=2**-6)
    got = np.asarray(q.astype(jnp.float32) * scale)
    want = np.asarray(ref_q.astype(jnp.float32) * ref_scale)
    tol = np.abs(want) * 2.0**-3 + np.asarray(ref_scale) * 2.0**-6
    assert (np.abs(got - want) <= tol).all()
