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
"""`attend_row_mapped`: `blocks/qsa_attn.attend_row` reading request-local group ids.

A copy, not an edit: `attend_row`'s signature is shared with the segment kernel's
Stage B (owned elsewhere), so the whole-step kernel carries its own variant. The only
differences are the id translation in `copies` and the two extra arguments.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.qsa_attn.block import MICRO, TILE, _unpack_kv, lax_rem
from ..ops import BF16, F32, I32


def attend_row_mapped(
    r: Any,
    g0: Any = 0,
    pre: Any = 0,
    *,
    nrows: int | None = None,
    pool: Any,
    ring: Any,
    sems: Any,
    pages: Any,
    kv_len: Any,
    table: Any,
    rows_per_req: int,
    q: jax.Array,
    gate: jax.Array,
    off: Any,
    scale_k: Any,
    scale_v: Any,
    sm_scale: float,
    interpret: bool,
    fold: bool = False,
    ablate: str = "",
    recent: tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, Any, Any]:
    """`attend_row` over GROUP ids: `pages[row, col]` holds a request-local group id `v`
    (what the in-kernel indexer selects), mapped at DMA-issue time to the pool's micro-page
    `table[row // rows_per_req, v >> log2(per_page)] * per_page + (v & (per_page - 1))`.
    One extra SMEM load per copy instead of a separate id-translation pass.

    `recent = (k [n, 256], v [n, 256], valid [1, n], valid [n, 1])`: tokens held in VMEM (the step's own
    tokens and the old members of their groups, already dequantized) merged into the same
    online softmax after the pool tiles; a row with no pool tokens still attends them.

    Gated sparse attention of row `r`: `([nh, 256]` F32, next `g0`, next `pre`).

    Ring slots follow a tile counter carried across rows (`g0`: this row's
    first tile), so the ring never drains at a row boundary: once a row has
    issued its own tiles, its last iterations issue row `r + 1`'s first tiles
    into the slots that row will read. `pre` is how many of this row's tiles a
    previous row already issued. Rows run in order `0 .. nrows - 1`.

    `fold`: apply `scale_k * sm_scale` to q and `scale_v` to the result instead
    of to every K / V tile. `ablate` (timing only, wrong results): "dma" skips
    the copies, "unpack" uses zero K / V, "matmul" skips QK and PV.

    `q`, `gate`: `[nh, 256]`. `ring`: `[2, TILE, 1, 4, 256]` FP8 VMEM; `sems`: DMA
    semaphores `[2]`. `pages` / `kv_len`: SMEM refs.
    """
    per_page = pool.shape[1] // MICRO
    nrows = pages.shape[0] if nrows is None else nrows
    nslots = ring.shape[0]

    def n_tiles(length: Any) -> Any:
        return (length + TILE - 1) // TILE

    length = kv_len[r]
    ntiles = n_tiles(length)
    rn = jnp.minimum(r + 1, nrows - 1)
    ntiles_next = jnp.where(r + 1 < nrows, n_tiles(kv_len[rn]), 0)
    # Cross-row prefetch only when this row covers the ring, so the next row's
    # pre-issued tiles are always its first ones, contiguous from 0.
    cross = ntiles >= nslots - 1
    nh, dim = q.shape

    ncols = pages.shape[1]
    assert per_page & (per_page - 1) == 0, per_page
    shift = per_page.bit_length() - 1

    def copies(row: Any, t: Any, s: Any) -> list[Any]:
        """Tile `t` of `row`: `TILE // MICRO` micro-page copies, issued unconditionally.

        No per-copy branch: a straight-line issue loop costs a few ns per DMA,
        a `pl.when` per copy several times that. Columns past `budget + 1` clamp
        to the last one; tokens past the row's length are masked below.
        """
        out = []
        rq = row // rows_per_req
        for j in range(TILE // MICRO):
            gid = pages[row, jnp.minimum(t * (TILE // MICRO) + j, ncols - 1)]
            v = table[rq, gid >> shift] * per_page + (gid & (per_page - 1))
            out.append(
                pltpu.make_async_copy(
                    pool.at[v >> shift, pl.ds((v & (per_page - 1)) * MICRO, MICRO)],
                    ring.at[s, pl.ds(j * MICRO, MICRO)],
                    sems.at[s],
                )
            )
        return out

    def start(row: Any, t: Any, s: Any) -> None:
        if ablate == "dma":
            return
        for cp in copies(row, t, s):
            cp.start()

    def wait(t: Any, s: Any) -> None:
        if ablate == "dma":
            return
        if interpret:
            for cp in copies(r, t, s):
                cp.wait()
        else:
            # One wait for the whole slot: the copies above add up to its bytes.
            pltpu.make_async_copy(ring.at[s], ring.at[s], sems.at[s]).wait()

    def slot(t: Any) -> Any:
        return lax_rem(g0 + t, nslots)

    for k in range(nslots - 1):  # this row's first S-1 tiles not yet in flight

        @pl.when((k >= pre) & (k < ntiles))
        def _(k=k):
            start(r, k, slot(k))

    qf = q.astype(F32) * (scale_k * sm_scale if fold else 1.0)
    k_mul, v_mul, s_mul = (1.0, 1.0, 1.0) if fold else (scale_k, scale_v, sm_scale)
    static = lambda x: isinstance(x, (int, float))  # noqa: E731

    def tile(t: Any, carry: Any) -> Any:
        m, l, acc = carry
        s = slot(t)
        wait(t, s)
        ahead = t + nslots - 1  # into the slot tile t - 1 has freed

        @pl.when(ahead < ntiles)
        def _():
            start(r, ahead, slot(ahead))

        k_next = ahead - ntiles  # the next row's tile k lands in slot(ahead)

        @pl.when(cross & (k_next >= 0) & (k_next < ntiles_next))
        def _():
            start(rn, k_next, slot(ahead))

        if ablate == "unpack":
            k8 = v8 = jnp.zeros((TILE, dim), BF16)
        else:
            k8, v8 = _unpack_kv(ring.at[s], off, interpret)
        tok = t * TILE + jax.lax.broadcasted_iota(I32, (1, TILE), 1)
        valid = tok < length
        key = k8.astype(F32) if static(k_mul) and k_mul == 1.0 else k8.astype(F32) * k_mul
        # A column iota, not `valid.reshape`: Mosaic has no i1 shape cast.
        tok_col = t * TILE + jax.lax.broadcasted_iota(I32, (TILE, 1), 0)
        vf = v8.astype(F32) if static(v_mul) and v_mul == 1.0 else v8.astype(F32) * v_mul
        value = jnp.where(tok_col < length, vf, 0.0)
        if ablate == "matmul":
            score = jnp.zeros((nh, TILE), F32)
        else:
            score = jax.lax.dot_general(
                qf, key, (((1,), (1,)), ((), ())), preferred_element_type=F32
            )
            if not (static(s_mul) and s_mul == 1.0):
                score = score * s_mul
        score = jnp.where(valid, score, -jnp.inf)
        m_new = jnp.maximum(m, jnp.max(score, axis=-1, keepdims=True))
        p = jnp.where(valid, jnp.exp(score - m_new), 0.0)
        alpha = jnp.exp(m - m_new)
        l_new = l * alpha + jnp.sum(p, axis=-1, keepdims=True)
        pv = (
            jnp.zeros((nh, dim), F32)
            if ablate == "matmul"
            else jnp.dot(p, value, preferred_element_type=F32)
        )
        acc_new = acc * alpha + pv
        return m_new, l_new, acc_new

    init = (
        jnp.full((nh, 1), -jnp.inf, F32),
        jnp.zeros((nh, 1), F32),
        jnp.zeros((nh, dim), F32),
    )
    m, l, acc = jax.lax.fori_loop(0, ntiles, tile, init)
    has = length > 0
    if recent is not None:
        kr, vr, ok, ok_col = recent  # row and column masks (Mosaic has no i1 transpose)
        sc = jax.lax.dot_general(qf, kr, (((1,), (1,)), ((), ())), preferred_element_type=F32)
        sc = jnp.where(ok, sc * (1.0 if fold else sm_scale), -jnp.inf)
        m_new = jnp.maximum(m, jnp.max(sc, axis=-1, keepdims=True))
        safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)
        p = jnp.where(ok, jnp.exp(sc - safe), 0.0)
        alpha = jnp.where(m == -jnp.inf, 0.0, jnp.exp(m - safe))
        l = l * alpha + jnp.sum(p, axis=-1, keepdims=True)
        acc = acc * alpha + jnp.dot(p, jnp.where(ok_col, vr, 0.0), preferred_element_type=F32)
        has = has | (jnp.sum(ok.astype(I32)) > 0)
    attn = acc / jnp.where(l > 0, l, 1.0) * (scale_v if fold else 1.0)
    gated = (
        attn.astype(BF16).astype(F32) * jax.nn.sigmoid(gate.astype(F32)).astype(BF16).astype(F32)
    ).astype(BF16).astype(F32)
    pre_next = jnp.where(cross, jnp.minimum(nslots - 1, ntiles_next), 0)
    return jnp.where(has, gated, 0.0), g0 + ntiles, pre_next


