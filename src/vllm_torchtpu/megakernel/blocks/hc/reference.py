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
"""Pure-JAX reference for a gated-residual hyper-connection (HC).

The residual is `NHC` streams of width `h`, stored flat as `[T, NHC*h]`.
`mix` reduces the streams to one `[T, h]` sublayer input; `combine` adds the
sublayer output back into every stream with a per-stream injection gate.
BF16 rounding points follow the checkpoint's PyTorch definition: every
linear output and the normalized streams are BF16; the SiLU, sigmoid and
stream reduction are computed in float32 between those boundaries.

Parameters (MK layout, `[in, out]`):
  norm  `[NHC*h]`       per-stream zero-centered RMSNorm weight (shared shape)
  down  `[NHC*h, LR]`   low-rank gate down projection
  up    `[LR, NHC*h]`   low-rank gate up projection
  inj   `[NHC*h, NHC]`  block injection projection
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ...ops import F32, rmsnorm, rnd


def hc_normalize(x: jax.Array, norm_w: jax.Array, nhc: int, eps: float) -> jax.Array:
    """Per-stream zero-centered RMSNorm of `x [T, nhc*h]`, BF16-rounded, f32."""
    h = x.shape[-1] // nhc
    x = x.astype(F32)
    norm_w = norm_w.astype(F32)
    return jnp.concatenate(
        [
            rmsnorm(
                x[:, b * h : (b + 1) * h],
                norm_w[b * h : (b + 1) * h],
                eps,
                zero_centered=True,
            )
            for b in range(nhc)
        ],
        axis=-1,
    )


def _mm(x: jax.Array, w: jax.Array) -> jax.Array:
    """BF16 matmul with f32 accumulation, output rounded to BF16 (as f32)."""
    y = jnp.dot(
        x.astype(jnp.bfloat16),
        w.astype(jnp.bfloat16),
        preferred_element_type=F32,
    )
    return rnd(y)


def hc_gate_down(xn: jax.Array, down: jax.Array, nhc: int) -> jax.Array:
    """`silu(xn @ down / nhc)` with the BF16 boundaries of the reference."""
    g = rnd(_mm(xn, down) / nhc)
    return rnd(jax.nn.silu(g))


def hc_mix_streams(xn: jax.Array, up_logits: jax.Array, nhc: int) -> jax.Array:
    """`sum_b sigmoid(up_logits)_b * xn_b / nhc`, BF16-rounded, f32."""
    h = xn.shape[-1] // nhc
    s = jax.nn.sigmoid(up_logits) * xn
    mixed = s[:, :h]
    for b in range(1, nhc):
        mixed = mixed + s[:, b * h : (b + 1) * h]
    return rnd(mixed / nhc)


def hc_injection(xn: jax.Array, inj: jax.Array, nhc: int) -> jax.Array:
    """`2 * sigmoid((xn @ inj) / nhc)` in f32, `[T, nhc]`."""
    return 2.0 * jax.nn.sigmoid(_mm(xn, inj) / nhc)


def hc_mix(
    x: jax.Array, p: dict[str, Any], *, nhc: int, eps: float
) -> tuple[jax.Array, jax.Array]:
    """Return `(mixed [T, h], xn [T, nhc*h])` for residual `x [T, nhc*h]`."""
    xn = hc_normalize(x, p["norm"], nhc, eps)
    g = hc_gate_down(xn, p["down"], nhc)
    u = _mm(g, p["up"])
    return hc_mix_streams(xn, u, nhc), xn


def hc_combine(
    block: jax.Array,
    x: jax.Array,
    xn: jax.Array,
    p: dict[str, Any],
    *,
    nhc: int,
) -> jax.Array:
    """Return the new residual `x_b + block * inj_b` per stream, BF16-rounded."""
    h = block.shape[-1]
    inj = hc_injection(xn, p["inj"], nhc)
    blk = rnd(block.astype(F32))
    x = x.astype(F32)
    return jnp.concatenate(
        [
            rnd(x[:, b * h : (b + 1) * h] + blk * inj[:, b : b + 1])
            for b in range(nhc)
        ],
        axis=-1,
    )


def hc_sublayer_reference(
    x: jax.Array,
    p: dict[str, Any],
    sublayer: Any,
    *,
    nhc: int,
    eps: float,
) -> jax.Array:
    """`combine(sublayer(mix(x)), x)`: one HC-wrapped sublayer on residual `x`."""
    mixed, xn = hc_mix(x, p, nhc=nhc, eps=eps)
    return hc_combine(sublayer(mixed), x, xn, p, nhc=nhc)
