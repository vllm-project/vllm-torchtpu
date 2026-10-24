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
"""QSA decode indexer on the TensorCore: score, exact top-K, compaction (no scalar loop).

Stock semantics (`kernels/qwen4_exp`, `streamindex_topk` decode path):

  iq      = index_rope(gemma_norm(mixed @ W_idx[:, :512] per head, idx_q_ln))   [rows, 4, 128]
  score_g = sum_h relu(iq_h . ck_g) * 128^-0.5     over complete groups g < n_vis = len // 4
  sel     = top-K of score (ties -> lower group id); n_vis <= K selects every visible group

Selection (measured exact on every row, ~2 µs/row, P1b j-988c4e1c): bisection on the f32
bit pattern with per-row counts as a vector, a tie-break bisection on ids, then compaction
by matmuls: in-row ranks `P = sel @ U`, row offsets `O = c @ Us`, the row of output slot j by
a compare-and-count, that row's ranks by a one-hot matmul, the lane by another count.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..ops import BF16, F32, I32, rnd, rmsnorm

IDX_HEADS, IDX_DIM = 4, 128


def _next_toward(x: jax.Array, up: jax.Array, bitcast: Any) -> jax.Array:
    """`nextafter(x, +inf if up else -inf)` for finite nonzero f32 `x`, by bit arithmetic."""
    b = bitcast(x, I32)
    inc = jnp.where(x > 0, up, ~up)  # magnitude grows toward +inf for x > 0
    return bitcast(jnp.where(inc, b + 1, b - 1), F32)


def fma_bf16(a: jax.Array, b: jax.Array, c: jax.Array, bitcast: Any) -> jax.Array:
    """Stock `_fma`: BF16 of the exact `a * b + c` (two-sum residual breaks BF16 ties)."""
    product = a * b
    high = product + c
    z = high - product
    residual = (product - (high - z)) + (c - z)
    tie = (bitcast(high, I32) & 0xFFFF) == 0x8000
    adj = jnp.where(tie & (residual != 0), _next_toward(high, residual > 0, bitcast), high)
    return rnd(adj)


def index_rope(x: jax.Array, cos: jax.Array, sin: jax.Array, rot: int, bitcast: Any) -> jax.Array:
    """Stock `index_rope`: `[B(a c - B(b s)), B(a s + B(b c)), x[rot:]]` on rows of `x`."""
    half = rot // 2
    a, b = x[:, :half], x[:, half:rot]
    y1 = fma_bf16(a, cos, -rnd(b * sin), bitcast)
    y2 = fma_bf16(a, sin, rnd(b * cos), bitcast)
    return jnp.concatenate((y1, y2, x[:, rot:]), axis=-1)


def index_query(y: jax.Array, iqn: jax.Array, cos, sin, rot: int, eps: float, bitcast) -> list:
    """Per head `[rows, 128]` BF16-valued queries from the projection `y [rows, >=512]`."""
    out = []
    for h in range(IDX_HEADS):
        q = rmsnorm(y[:, h * IDX_DIM : (h + 1) * IDX_DIM], iqn, eps, zero_centered=True)
        out.append(index_rope(q, cos, sin, rot, bitcast))
    return out


def score_rows(qr: jax.Array, ck: jax.Array) -> jax.Array:
    """`[T, G]` scores of `T` rows: `qr [T*4, 128]` (row-major over heads), `ck [G, 128]`."""
    st = jax.lax.dot_general(qr, ck, (((1,), (1,)), ((), ())), preferred_element_type=F32)
    T = qr.shape[0] // IDX_HEADS
    return (jnp.maximum(st, 0.0) * IDX_DIM**-0.5).reshape(T, IDX_HEADS, -1).sum(axis=1)


def thresholds(bits: jax.Array, ids: jax.Array, nv: jax.Array, K: int, G: int):
    """Per-row `(thr, tmax)` [R, 1, 1] for bit patterns `bits [R, S, 128]` (-1 = invisible)."""
    cnt = lambda m: jnp.sum(m.astype(I32), axis=(1, 2), keepdims=True)  # noqa: E731
    over = nv > K
    lo = jnp.zeros(nv.shape, I32)
    for b in range(30, -1, -1):
        cand = lo | (1 << b)
        lo = jnp.where(cnt(bits >= cand) >= K, cand, lo)
    thr = jnp.where(over, lo, 0)
    eq = bits == thr
    need = K - cnt(bits > thr)
    t = jnp.zeros(nv.shape, I32)
    for b in range(int(G - 1).bit_length() - 1, -1, -1):
        cand = t | (1 << b)
        t = jnp.where(cnt(eq & (ids < cand)) < need, cand, t)
    tmax = jnp.where(over & (cnt(eq) > need), t, G)
    return thr, tmax


def compact_row(b, th, tm, nv, K: int, consts: tuple) -> jax.Array:
    """Selected group ids of one row as a lane-major `[1, K]` int32 (slots past the count: 0).

    `b [S, 128]` bit patterns (-1 = invisible), `th / tm / nv` `[1, 1]`.
    """
    U, Us, jcol, srow, lid = consts
    S = b.shape[0]
    ov = nv > K
    sel = (ov & ((b > th) | ((b == th) & (lid <= tm)))) | (~ov & (b >= 0))
    P = jnp.dot(sel.astype(F32).astype(BF16), U, preferred_element_type=F32)
    c_row = jnp.transpose(jnp.broadcast_to(P[:, 127:128], (S, 128)))[0:1, :]
    O_row = jnp.dot(c_row.astype(BF16), Us, preferred_element_type=F32)
    E_row = O_row + c_row
    jf = jcol.astype(F32)
    r_j = jnp.sum((E_row <= jf).astype(I32), axis=1, keepdims=True)
    oh = srow == r_j
    Pg = jnp.dot(oh.astype(F32).astype(BF16), P.astype(BF16), preferred_element_type=F32)
    O_rj = jnp.sum(jnp.where(oh, O_row, 0.0), axis=1, keepdims=True)
    lane_j = jnp.sum((Pg <= jf - O_rj).astype(I32), axis=1, keepdims=True)
    # Slots past the row's count hold group 0, so DMAs issued for them stay in bounds.
    idc = jnp.where(jcol < jnp.minimum(nv, K), r_j * 128 + lane_j, 0).astype(F32)
    return jnp.transpose(jnp.broadcast_to(idc, (K, 128)))[0:1, :].astype(I32)


def compact_consts(S: int, K: int) -> tuple:
    li = jax.lax.broadcasted_iota(I32, (128, 128), 0)
    lj = jax.lax.broadcasted_iota(I32, (128, 128), 1)
    si = jax.lax.broadcasted_iota(I32, (S, S), 0)
    sj = jax.lax.broadcasted_iota(I32, (S, S), 1)
    return (
        (li <= lj).astype(BF16),
        (si < sj).astype(BF16),
        jax.lax.broadcasted_iota(I32, (K, 1), 0),
        jax.lax.broadcasted_iota(I32, (1, S), 1),
        jax.lax.broadcasted_iota(I32, (S, 128), 0) * 128
        + jax.lax.broadcasted_iota(I32, (S, 128), 1),
    )


# ---------------------------------------------------------------- reference
def topk_reference(scores: np.ndarray, nvis: int, K: int) -> list[int]:
    """Stock selection: top-K by score, ties to the lower id, over ids < nvis."""
    s = np.asarray(scores[:nvis], np.float32)
    return np.lexsort((np.arange(nvis), -s))[:K].tolist()
