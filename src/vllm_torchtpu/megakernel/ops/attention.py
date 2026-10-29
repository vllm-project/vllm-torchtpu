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
"""Stateless online softmax LSE and `(max, sum)` merge primitives for GQA/MLA."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from .base import BF16, F32


def split_bf16_lse(lse: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Encode FP32 `lse` as `(hi_bf16, lo_bf16)` (~16-bit mantissa over BF16 ICI)."""
    lse_f32 = lse.astype(F32)
    hi = lse_f32.astype(BF16)
    lo = jnp.where(jnp.isfinite(lse_f32), lse_f32 - hi.astype(F32), F32(0.0)).astype(
        BF16
    )
    return hi, lo


def combine_bf16_lse(hi: jax.Array, lo: jax.Array) -> jax.Array:
    """Reconstruct FP32 `lse` from `(hi_bf16, lo_bf16)`."""
    return hi.astype(F32) + lo.astype(F32)


def online_softmax_lse_merge(
    partials: jax.Array | Sequence[jax.Array],
    lse: jax.Array | Sequence[jax.Array],
    *,
    lse_lo: jax.Array | Sequence[jax.Array] | None = None,
    init_out: jax.Array | None = None,
    init_lse: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Merge normalized `[G, rows, D]` outputs and `[G, rows, 1]` LSEs in VREGs."""
    n_peers = len(partials)
    if n_peers <= 0:
        raise ValueError("online_softmax_lse_merge requires at least 1 peer")

    p0 = partials[0].astype(F32)
    rows, dim = p0.shape
    if init_out is not None and init_lse is not None:
        top = init_lse.astype(F32).reshape(rows, 1)
        valid0 = jnp.isfinite(top)
        acc = jnp.where(valid0, init_out.astype(F32), jnp.zeros((rows, dim), F32))
        den = jnp.where(valid0, jnp.ones((rows, 1), F32), jnp.zeros((rows, 1), F32))
        start = 0
    else:
        lse0 = lse[0].astype(F32)
        if lse_lo is not None:
            lse0 = lse0 + lse_lo[0].astype(F32)
        top = lse0.reshape(rows, 1)
        valid0 = jnp.isfinite(top)
        acc = jnp.where(valid0, p0, jnp.zeros((rows, dim), F32))
        den = jnp.where(valid0, jnp.ones((rows, 1), F32), jnp.zeros((rows, 1), F32))
        start = 1

    for g in range(start, n_peers):
        pg = partials[g].astype(F32)
        lg = lse[g].astype(F32)
        if lse_lo is not None:
            lg = lg + lse_lo[g].astype(F32)
        lg = lg.reshape(rows, 1)
        new_top = jnp.maximum(top, lg)
        safe_top = jnp.where(jnp.isfinite(new_top), new_top, F32(0.0))
        keep = jnp.where(jnp.isfinite(top), jnp.exp(top - safe_top), F32(0.0))
        weight = jnp.where(jnp.isfinite(lg), jnp.exp(lg - safe_top), F32(0.0))
        acc = acc * keep + weight * pg
        den = den * keep + weight
        top = new_top

    merged_out = acc / jnp.where(den > 0.0, den, F32(1.0))
    merged_lse = jnp.where(
        den > 0.0, top + jnp.log(jnp.maximum(den, F32(1e-38))), F32(-jnp.inf)
    )
    return merged_out, merged_lse


def online_softmax_stats_merge(
    accs: jax.Array | Sequence[jax.Array],
    maxs: jax.Array | Sequence[jax.Array],
    sums: jax.Array | Sequence[jax.Array],
    *,
    normalize: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Merge unnormalized `(acc, m, l)` softmax shards across `G` peers."""
    n_peers = len(accs)
    if n_peers <= 0:
        raise ValueError("online_softmax_stats_merge requires at least 1 peer")

    rows = accs[0].shape[0]
    m_list = [maxs[g].astype(F32).reshape(rows, 1) for g in range(n_peers)]
    m = m_list[0]
    for mg in m_list[1:]:
        m = jnp.maximum(m, mg)
    safe_m = jnp.where(jnp.isfinite(m), m, F32(0.0))
    l_acc = jnp.zeros_like(m)
    o_acc = jnp.zeros(accs[0].shape, dtype=F32)
    for g in range(n_peers):
        mg = m_list[g]
        lg = sums[g].astype(F32).reshape(rows, 1)
        w = jnp.where(jnp.isfinite(mg), jnp.exp(mg - safe_m), F32(0.0))
        l_acc = l_acc + w * lg
        o_acc = o_acc + w * accs[g].astype(F32)
    if normalize:
        o_acc = o_acc / jnp.where(l_acc > 0.0, l_acc, F32(1.0))
    return o_acc, m, l_acc
