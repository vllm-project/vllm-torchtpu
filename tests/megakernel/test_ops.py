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

import jax.numpy as jnp

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
