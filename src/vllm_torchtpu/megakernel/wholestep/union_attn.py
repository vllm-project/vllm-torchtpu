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
"""Per-request union attention: a request's `T` verify rows read one union of their selections.

`attend_row` streams every row's own selected groups, so the `T` rows of a request read their
shared groups `T` times. Here the request's pool-path selections are merged into one ascending
union of groups, each tagged with a `T`-bit code (bit `t`: row `t` selected it); the union is
streamed once through the ring and scored for all `T * nh` query rows in one matmul per tile,
each row masked to its own groups. Every row attends exactly the tokens it attends in
`attend_row`, in the same ascending order; only the tile partition of the online softmax differs
(a row's groups sit in fewer, fuller tiles), so results agree to f32 rounding, not bit for bit.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.qsa_attn.block import MICRO, TILE, _unpack_kv
from ..ops import BF16, F32, I32

GROUPS_PER_TILE = TILE // MICRO  # one group = one micro-page of 4 tokens


UNION_CHUNK = 256  # output slots compacted per pass (bounds the [slots, S] temporaries)


def union_consts(S: int, KU: int) -> tuple:
    """Constants of `compact_union` (`compact_consts` at the per-pass slot width)."""
    W = min(KU, UNION_CHUNK)
    li = jax.lax.broadcasted_iota(I32, (128, 128), 0)
    lj = jax.lax.broadcasted_iota(I32, (128, 128), 1)
    si = jax.lax.broadcasted_iota(I32, (S, S), 0)
    sj = jax.lax.broadcasted_iota(I32, (S, S), 1)
    return (
        (li <= lj).astype(BF16),
        (si < sj).astype(BF16),
        jax.lax.broadcasted_iota(I32, (W, 1), 0),
        jax.lax.broadcasted_iota(I32, (1, S), 1),
        jax.lax.broadcasted_iota(I32, (W, 128), 1),
    )


def compact_union(code: jax.Array, KU: int, consts: tuple) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Ascending union of the groups with a nonzero row code, as lane-major `[1, KU]` ids and
    codes (slots past the count: 0) and the count `[1, 1]`.

    `code [S, 128]` int32 (lane = group id, value = row bits < 2^T). The compaction of
    `indexer.compact_row` with every coded lane selected, `UNION_CHUNK` output slots per pass
    (the `[slots, S]` one-hot temporaries overflow VMEM: all 2048 slots at once 62.8 of 62.1 MiB at
    16 rows, j-c89e9978; 512 per pass still 62.4, j-bfd3a7f8); each slot's code is gathered from its group's sublane row by the
    same one-hot matmul (codes <= 15 are exact in BF16).
    """
    U, Us, jcol0, srow, lane = consts
    S, W = code.shape[0], jcol0.shape[0]
    sel = code > 0
    P = jnp.dot(sel.astype(F32).astype(BF16), U, preferred_element_type=F32)
    c_row = jnp.transpose(jnp.broadcast_to(P[:, 127:128], (S, 128)))[0:1, :]
    O_row = jnp.dot(c_row.astype(BF16), Us, preferred_element_type=F32)
    E_row = O_row + c_row
    Pb, codeb = P.astype(BF16), code.astype(F32).astype(BF16)
    n = jnp.sum(sel.astype(I32), axis=(0, 1), keepdims=True)
    ids_out, codes_out = [], []
    for j0 in range(0, KU, W):
        jcol = jcol0 + j0
        jf = jcol.astype(F32)
        r_j = jnp.sum((E_row <= jf).astype(I32), axis=1, keepdims=True)
        hit = srow == r_j
        oh = hit.astype(F32).astype(BF16)
        Pg = jnp.dot(oh, Pb, preferred_element_type=F32)
        O_rj = jnp.sum(jnp.where(hit, O_row, 0.0), axis=1, keepdims=True)
        lane_j = jnp.sum((Pg <= jf - O_rj).astype(I32), axis=1, keepdims=True)
        Cg = jnp.dot(oh, codeb, preferred_element_type=F32)
        code_j = jnp.sum(jnp.where(lane == lane_j, Cg, 0.0), axis=1, keepdims=True)
        live = jcol < n
        ids = jnp.where(live, r_j * 128 + lane_j, 0).astype(F32)
        codes = jnp.where(live, code_j, 0.0)
        ids_out.append(jnp.transpose(jnp.broadcast_to(ids, (W, 128)))[0:1, :].astype(I32))
        codes_out.append(jnp.transpose(jnp.broadcast_to(codes, (W, 128)))[0:1, :].astype(I32))
    cat = lambda xs: xs[0] if len(xs) == 1 else jnp.concatenate(xs, axis=1)  # noqa: E731
    return cat(ids_out), cat(codes_out), n


def attend_request(
    *,
    T: int,
    nh: int,
    pool: Any,
    ring: Any,
    sems: Any,
    ids: Any,
    codes: Any,
    n_union: Any,
    table: Any,
    table_row: Any,
    qs: list[jax.Array],
    gates: list[jax.Array],
    has: list[Any],
    off: Any,
    scale_k: Any,
    scale_v: Any,
    sm_scale: float,
    interpret: bool,
    recent: tuple[jax.Array, jax.Array, list[jax.Array], jax.Array],
) -> list[jax.Array]:
    """Gated sparse attention of one request's `T` rows over their union: `T` `[nh, 256]` F32.

    `ids` SMEM `[1, KU + 128]` request-local group ids of the union (ascending), `codes` a
    `[1, KU]` VMEM ref of their row bits, `n_union` the count; `table[table_row, v >> shift]`
    maps a group to its micro-page as `attend_row`'s `table` does. `qs` / `gates` `[nh, 256]` and
    `has` (the row has pool tokens) per row. `recent = (k [8, 256], v [8, 256], ok_rows, ok_any
    [8, 1])` in pool units, `ok_rows` `T` `[1, 8]` masks. Each tile is copied and unpacked once and
    scored per row (`[nh, 256]` queries, as `attend_row`); scaling follows `attend_row` without
    `fold`: K by `scale_k`, scores by `sm_scale`, V by `scale_v`, per tile.
    """
    per_page = pool.shape[1] // MICRO
    nslots = ring.shape[0]
    ncols = ids.shape[1]
    assert per_page & (per_page - 1) == 0, per_page
    shift = per_page.bit_length() - 1
    length = n_union * MICRO
    ntiles = (length + TILE - 1) // TILE

    def copies(t: Any, s: Any) -> list[Any]:
        out = []
        for j in range(GROUPS_PER_TILE):
            v = ids[0, jnp.minimum(t * GROUPS_PER_TILE + j, ncols - 1)]
            v = table[table_row, v >> shift] * per_page + (v & (per_page - 1))
            out.append(
                pltpu.make_async_copy(
                    pool.at[v >> shift, pl.ds((v & (per_page - 1)) * MICRO, MICRO)],
                    ring.at[s, pl.ds(j * MICRO, MICRO)],
                    sems.at[s],
                )
            )
        return out

    def start(t: Any, s: Any) -> None:
        for cp in copies(t, s):
            cp.start()

    def wait(t: Any, s: Any) -> None:
        if interpret:
            for cp in copies(t, s):
                cp.wait()
        else:
            pltpu.make_async_copy(ring.at[s], ring.at[s], sems.at[s]).wait()

    for k in range(nslots - 1):

        @pl.when(k < ntiles)
        def _(k=k):
            start(k, k)

    # A tile's 64 group codes sit in one half of an aligned 128-lane window; expanding the whole
    # window with the half's one-hot map gives each of the tile's 256 tokens its group's code
    # (token i: group slot i // 4), with no sub-vreg lane slice.
    gi = jax.lax.broadcasted_iota(I32, (2 * GROUPS_PER_TILE, TILE), 0)
    ti = jax.lax.broadcasted_iota(I32, (2 * GROUPS_PER_TILE, TILE), 1)
    expand = [(ti // MICRO == gi - h * GROUPS_PER_TILE).astype(F32).astype(BF16) for h in (0, 1)]
    qf = [q.astype(F32) for q in qs]

    def tile(t: Any, carry: Any) -> Any:
        s = jax.lax.rem(t, nslots)
        wait(t, s)
        ahead = t + nslots - 1

        @pl.when(ahead < ntiles)
        def _():
            start(ahead, jax.lax.rem(ahead, nslots))

        win = codes[:, pl.ds(pl.multiple_of((t // 2) * 128, 128), 128)].astype(F32).astype(BF16)
        tok_code = jnp.where(
            jax.lax.rem(t, 2) == 0,
            jnp.dot(win, expand[0], preferred_element_type=F32),
            jnp.dot(win, expand[1], preferred_element_type=F32),
        ).astype(I32)
        tok = t * TILE + jax.lax.broadcasted_iota(I32, (1, TILE), 1)
        k8, v8 = _unpack_kv(ring.at[s], off, interpret)
        key = k8.astype(F32) * scale_k
        tok_col = t * TILE + jax.lax.broadcasted_iota(I32, (TILE, 1), 0)
        value = jnp.where(tok_col < length, v8.astype(F32) * scale_v, 0.0)
        out = []
        for r in range(T):
            m, l, acc = carry[r]
            valid = (tok < length) & ((tok_code & (1 << r)) != 0)
            score = jax.lax.dot_general(
                qf[r], key, (((1,), (1,)), ((), ())), preferred_element_type=F32
            )
            score = jnp.where(valid, score * sm_scale, -jnp.inf)
            m_new = jnp.maximum(m, jnp.max(score, axis=-1, keepdims=True))
            safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)  # no token of this row so far
            p = jnp.where(valid, jnp.exp(score - safe), 0.0)
            alpha = jnp.where(
                m == -jnp.inf, jnp.where(m_new == -jnp.inf, 1.0, 0.0), jnp.exp(m - safe)
            )
            out.append(
                (
                    m_new,
                    l * alpha + jnp.sum(p, axis=-1, keepdims=True),
                    acc * alpha + jnp.dot(p, value, preferred_element_type=F32),
                )
            )
        return tuple(out)

    dim = qs[0].shape[-1]
    init = tuple(
        (jnp.full((nh, 1), -jnp.inf, F32), jnp.zeros((nh, 1), F32), jnp.zeros((nh, dim), F32))
        for _ in range(T)
    )
    state = jax.lax.fori_loop(0, ntiles, tile, init)
    rk, rv, ok_rows, ok_any = recent
    rkey = rk * scale_k
    rval = jnp.where(ok_any, rv * scale_v, 0.0)
    out = []
    for r in range(T):
        m, l, acc = state[r]
        ok = ok_rows[r]
        sc = jax.lax.dot_general(qf[r], rkey, (((1,), (1,)), ((), ())), preferred_element_type=F32)
        sc = jnp.where(ok, sc * sm_scale, -jnp.inf)
        m_new = jnp.maximum(m, jnp.max(sc, axis=-1, keepdims=True))
        safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)
        p = jnp.where(ok, jnp.exp(sc - safe), 0.0)
        alpha = jnp.where(m == -jnp.inf, 0.0, jnp.exp(m - safe))
        l = l * alpha + jnp.sum(p, axis=-1, keepdims=True)
        acc = acc * alpha + jnp.dot(p, rval, preferred_element_type=F32)
        attn = acc / jnp.where(l > 0, l, 1.0)
        gated = (
            attn.astype(BF16).astype(F32)
            * jax.nn.sigmoid(gates[r].astype(F32)).astype(BF16).astype(F32)
        ).astype(BF16).astype(F32)
        row_has = has[r] | (jnp.max(jnp.where(ok, 1, 0)) > 0)
        out.append(jnp.where(row_has, gated, 0.0))
    return out
