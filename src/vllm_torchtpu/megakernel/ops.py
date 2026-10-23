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
"""Shared numeric, normalization, rotary, and routing primitives."""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
from jax import lax

BF16 = jnp.bfloat16
F32 = jnp.float32
I32 = jnp.int32
U32 = jnp.uint32
FP8 = jnp.float8_e4m3fn


def rup(n: int, k: int) -> int:
    """Round `n` up to the next multiple of `k`."""
    return (n + k - 1) // k * k


def rnd(x: jax.Array) -> jax.Array:
    """Round float32 tensor through bfloat16 back to float32."""
    return x.astype(BF16).astype(F32)


def bswitch(idx: jax.Array, fns: list):
    """Balanced binary tree of `lax.cond` (O(log N) select depth)."""
    if len(fns) == 1:
        return fns[0]()

    def rec(lo: int, hi: int):
        if hi - lo == 1:
            return fns[lo]()
        mid = (lo + hi) // 2
        return lax.cond(idx < mid, lambda: rec(lo, mid), lambda: rec(mid, hi))

    return rec(0, len(fns))


def rmsnorm(
    x: jax.Array,
    w: jax.Array,
    eps: float = 1e-6,
    *,
    zero_centered: bool = False,
) -> jax.Array:
    """RMSNorm in float32 with BF16 rounding (`1.0 + w` if `zero_centered`)."""
    v = x * lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    scale = (1.0 + w) if zero_centered else w
    return rnd(v * scale)


def rms_zc(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Zero-centered RMSNorm (`1.0 + w`) with BF16 intermediate rounding."""
    return rmsnorm(x, w, eps, zero_centered=True)


def rope_partial(
    x: jax.Array, cos: jax.Array, sin: jax.Array, rot_dim: int
) -> jax.Array:
    """Partial rotary positional embedding on the leading `rot_dim` columns."""
    half = rot_dim // 2
    x1, x2, rest = x[:, :half], x[:, half:rot_dim], x[:, rot_dim:]
    return jnp.concatenate(
        (
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
            rest,
        ),
        axis=-1,
    )


def silu_mul(gate: jax.Array, up: jax.Array) -> jax.Array:
    """SwiGLU activation `silu(gate) * up` rounded to BF16."""
    return (jax.nn.silu(gate) * up).astype(BF16)


def scope_fn(enabled: bool):
    """Return a context manager factory for `jax.named_scope` when `enabled=True`."""
    return jax.named_scope if enabled else lambda _: contextlib.nullcontext()
