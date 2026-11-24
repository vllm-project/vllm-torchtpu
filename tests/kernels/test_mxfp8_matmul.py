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
"""mxfp8_matmul against an f32 matmul of the dequantized weight."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul.mxfp8_matmul import (
    MX_BLOCK,
    fits,
    mxfp8_matmul,
)

pytestmark = pytest.mark.tpu_test


def _operands(m: int, k: int, n: int):
    kw, ks, kx = jax.random.split(jax.random.PRNGKey(m * k + n), 3)
    w = (jax.random.normal(kw, (k, n)) * 2).astype(jnp.float8_e4m3fn)
    e = jax.random.randint(ks, (k // MX_BLOCK, n), -9, 3).astype(jnp.float32)
    scale = jnp.exp2(e).astype(jnp.bfloat16)
    x = jax.random.normal(kx, (m, k)).astype(jnp.bfloat16)
    dequant = w.astype(jnp.float32) * jnp.repeat(scale.astype(jnp.float32), 32, 0)
    # The dequantized weight is exact in bf16.
    np.testing.assert_array_equal(
        np.asarray(dequant.astype(jnp.bfloat16).astype(jnp.float32)),
        np.asarray(dequant),
    )
    return x, w, scale, dequant


def _check(out, x, dequant):
    ref = jnp.dot(x.astype(jnp.float32), dequant, precision="highest")
    # One bf16 rounding of an f32 sum: within half a bf16 ulp plus f32 noise.
    tol = jnp.abs(ref) * 2.0**-8 + 1e-3
    assert out.dtype == jnp.bfloat16
    assert bool(jnp.all(jnp.abs(out.astype(jnp.float32) - ref) <= tol))


@pytest.mark.parametrize(
    "m,k,n",
    [
        (8, 6144, 1536),  # MiniMax-M3 qkv at TP8, one decode request
        (24, 1024, 6144),  # o_proj, three requests
        (8, 6144, 3072),  # dense gate_up: 12 column chunks
        (8, 1536, 6144),  # dense down_proj
        (16, 384, 6144),  # shared-expert down_proj
        (256, 6144, 1536),  # qkv at the largest kernel bucket
        (256, 1024, 6144),  # o_proj at the largest kernel bucket
        (256, 6144, 3072),  # dense gate_up at the largest kernel bucket
    ],
)
def test_matches_dequantized_matmul(m, k, n):
    assert fits(m, k, n)
    x, w, scale, dequant = _operands(m, k, n)
    _check(mxfp8_matmul(x, w, scale), x, dequant)


def test_does_not_fit_a_weight_larger_than_vmem():
    # Dense gate_up at TP4: ran out of scoped VMEM when it was admitted.
    assert not fits(8, 6144, 6144)
    x, w, scale, _ = _operands(8, 6144, 6144)
    with pytest.raises(ValueError):
        mxfp8_matmul(x, w, scale)


def test_rejects_mismatched_scale():
    x, w, scale, _ = _operands(8, 512, 256)
    with pytest.raises(ValueError):
        mxfp8_matmul(x, w, scale[:, :128])
