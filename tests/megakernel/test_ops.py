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
import pytest

from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.ops import (
    BF16,
    F32,
    bswitch,
    col_chunk,
    gdn_gate_act,
    rms_zc,
    rmsnorm,
    rope_partial,
    round_bf16,
    round_up,
    silu_mul,
    validate_gdn_gate,
)


def test_ops_primitives():
    """Verify `round_up`, `round_bf16`, `rmsnorm`, `rope_partial`, and `silu_mul`."""
    assert round_up(1, 128) == 128
    assert round_up(128, 128) == 128
    assert round_up(129, 128) == 256

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
    assert jnp.allclose(rx, round_bf16(x))

    # SwiGLU and bswitch
    sm = silu_mul(x, x)
    assert sm.dtype == BF16
    val = bswitch(jnp.int32(2), [lambda i=i: jnp.int32(i * 10) for i in range(4)])
    assert int(val) == 20


def test_fp32_intermediate_precision_rope_and_swiglu():
    """Verify `round_intermediates=False` avoids multi-stage BF16 truncation drift."""
    rng = jax.random.PRNGKey(1234)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    x = jax.random.normal(k1, (16, 256), dtype=F32)
    cos = jax.random.normal(k2, (16, 32), dtype=F32)
    sin = jax.random.normal(k3, (16, 32), dtype=F32)
    up = jax.random.normal(k4, (16, 256), dtype=F32)

    out_rope = rope_partial(x, cos, sin, 64, round_intermediates=False)
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
        [
            round_bf16(round_bf16(x1 * cos) - round_bf16(x2 * sin)),
            round_bf16(round_bf16(x2 * cos) + round_bf16(x1 * sin)),
        ],
        axis=-1,
    )
    err_legacy = np.max(np.abs(np.asarray(legacy_rot, dtype=np.float64) - ref_rot))
    assert err_fp32 < 1e-6
    assert err_fp32 * 100.0 < err_legacy

    out_swiglu = silu_mul(x, up, round_intermediates=False)
    assert out_swiglu.dtype == BF16
    expected_swiglu = (jax.nn.silu(x) * up).astype(BF16)
    np.testing.assert_array_equal(
        np.asarray(out_swiglu).view(np.uint16),
        np.asarray(expected_swiglu).view(np.uint16),
    )


def test_default_bf16_rounded_intermediates_rope_and_swiglu():
    """Verify default `round_intermediates=True` matches BF16-rounded reference."""
    rng = jax.random.PRNGKey(1234)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    x = jax.random.normal(k1, (16, 256), dtype=F32)
    cos = jax.random.normal(k2, (16, 32), dtype=F32)
    sin = jax.random.normal(k3, (16, 32), dtype=F32)
    up = jax.random.normal(k4, (16, 256), dtype=F32)

    out_rope = rope_partial(x, cos, sin, 64)
    x1, x2, rest = x[:, :32], x[:, 32:64], x[:, 64:]
    expected_rope = jnp.concatenate(
        (
            round_bf16(round_bf16(x1 * cos) - round_bf16(x2 * sin)),
            round_bf16(round_bf16(x2 * cos) + round_bf16(x1 * sin)),
            rest,
        ),
        axis=-1,
    )
    np.testing.assert_array_equal(
        np.asarray(out_rope).view(np.uint32),
        np.asarray(expected_rope).view(np.uint32),
    )

    out_swiglu = silu_mul(x, up)
    expected_swiglu = (round_bf16(jax.nn.silu(x)) * up).astype(BF16)
    np.testing.assert_array_equal(
        np.asarray(out_swiglu).view(np.uint16),
        np.asarray(expected_swiglu).view(np.uint16),
    )


def test_col_chunk():
    """Verify `col_chunk` selects the largest 128-multiple divisor `<= cap`."""
    assert col_chunk(512, 1024) == 512
    assert col_chunk(1024, 1024) == 1024
    assert col_chunk(1536, 1024) == 768
    assert col_chunk(2048, 1024) == 1024
    assert col_chunk(1024, 384) == 256
    for bad in (0, -128, 100, 192):
        with pytest.raises(ValueError, match="positive multiple of 128"):
            col_chunk(bad)


def test_gdn_gate_act_and_validate_gdn_gate():
    """Verify `validate_gdn_gate` and `gdn_gate_act` for valid and invalid gates."""
    assert validate_gdn_gate("silu") == "silu"
    assert validate_gdn_gate("sigmoid") == "sigmoid"
    assert gdn_gate_act("silu") is jax.nn.silu
    assert gdn_gate_act("sigmoid") is jax.nn.sigmoid

    x = jnp.array([-1.0, 0.0, 2.0], dtype=F32)
    np.testing.assert_allclose(gdn_gate_act("silu")(x), jax.nn.silu(x))
    np.testing.assert_allclose(gdn_gate_act("sigmoid")(x), jax.nn.sigmoid(x))

    for bad in ("relu", "", None, 123):
        with pytest.raises(ValueError, match="'silu' or 'sigmoid'"):
            validate_gdn_gate(bad)
        with pytest.raises(MegakernelConfigError, match="'silu' or 'sigmoid'"):
            gdn_gate_act(bad, err_cls=MegakernelConfigError)
