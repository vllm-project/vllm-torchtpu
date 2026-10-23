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
"""Unit tests for `ops`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.megakernel.ops import (
    BF16,
    F32,
    bswitch,
    rms_zc,
    rmsnorm,
    rnd,
    rope_partial,
    rup,
    silu_mul,
)


def test_ops_primitives():
    """Verify `rup`, `rnd`, `rmsnorm`, `rope_partial`, `silu_mul`, and `bswitch`."""
    assert rup(1, 128) == 128
    assert rup(128, 128) == 128
    assert rup(129, 128) == 256

    x = jnp.ones((4, 128), dtype=F32)
    w = jnp.zeros((1, 128), dtype=F32)
    # Zero-centered RMSNorm: w=0 gives unit scale (1.0 + w = 1.0)
    y = rms_zc(x, w, 1e-6)
    assert jnp.allclose(y, jnp.ones_like(x), atol=1e-3)

    # Standard RMSNorm: w=0.5 gives 0.5 scale, while zero-centered gives 1.5
    w_half = jnp.full((1, 128), 0.5, dtype=F32)
    y_zc = rmsnorm(x, w_half, 1e-6, zero_centered=True)
    y_std = rmsnorm(x, w_half, 1e-6, zero_centered=False)
    assert jnp.allclose(y_zc, jnp.full_like(x, 1.5), atol=1e-2)
    assert jnp.allclose(y_std, jnp.full_like(x, 0.5), atol=1e-2)

    # Partial RoPE leaves trailing columns untouched
    cos = jnp.ones((1, 16), dtype=F32)
    sin = jnp.zeros((1, 16), dtype=F32)
    rx = rope_partial(x, cos, sin, 32)
    assert jnp.allclose(rx, rnd(x))

    # SwiGLU and bswitch
    sm = silu_mul(x, x)
    assert sm.dtype == BF16
    val = bswitch(jnp.int32(2), [lambda i=i: jnp.int32(i * 10) for i in range(4)])
    assert int(val) == 20


def test_fp32_intermediate_precision_rope_and_swiglu():
    """Verify `rope_partial` and `silu_mul` avoid multi-stage BF16 truncation drift."""
    rng = jax.random.PRNGKey(1234)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    x = jax.random.normal(k1, (16, 256), dtype=F32)
    cos = jax.random.normal(k2, (16, 32), dtype=F32)
    sin = jax.random.normal(k3, (16, 32), dtype=F32)
    up = jax.random.normal(k4, (16, 256), dtype=F32)

    out_rope = rope_partial(x, cos, sin, 64)
    assert out_rope.dtype == F32

    x1_64 = np.asarray(x[:, :32], dtype=np.float64)
    x2_64 = np.asarray(x[:, 32:64], dtype=np.float64)
    cos_64 = np.asarray(cos, dtype=np.float64)
    sin_64 = np.asarray(sin, dtype=np.float64)
    ref_rot = np.concatenate(
        [x1_64 * cos_64 - x2_64 * sin_64, x2_64 * cos_64 + x1_64 * sin_64],
        axis=-1,
    )
    err_fp32 = np.max(np.abs(np.asarray(out_rope[:, :64], dtype=np.float64) - ref_rot))
    x1, x2 = x[:, :32], x[:, 32:64]
    legacy_rot = jnp.concatenate(
        [rnd(rnd(x1 * cos) - rnd(x2 * sin)), rnd(rnd(x2 * cos) + rnd(x1 * sin))],
        axis=-1,
    )
    err_legacy = np.max(np.abs(np.asarray(legacy_rot, dtype=np.float64) - ref_rot))
    assert err_fp32 < 1e-6
    assert err_fp32 * 100.0 < err_legacy

    out_swiglu = silu_mul(x, up)
    assert out_swiglu.dtype == BF16
    expected_swiglu = (jax.nn.silu(x) * up).astype(BF16)
    np.testing.assert_array_equal(
        np.asarray(out_swiglu).view(np.uint16),
        np.asarray(expected_swiglu).view(np.uint16),
    )
