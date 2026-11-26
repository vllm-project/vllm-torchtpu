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
"""Per-request union attention: a request's `T` verify rows read one union of their
selections.

`attend_row` streams every row's own selected groups, so the `T` rows of a request read
their
shared groups `T` times. Here the request's pool-path selections are merged into one
ascending
union of groups, each tagged with a `T`-bit code (bit `t`: row `t` selected it); the
union is
streamed once through the ring and scored for all `T * nh` query rows in one matmul per
tile,
each row masked to its own groups. Every row attends exactly the tokens it attends in
`attend_row`, in the same ascending order; only the tile partition of the online softmax
differs
(a row's groups sit in fewer, fuller tiles), so results agree to f32 rounding, not bit
for bit.
"""

from __future__ import annotations

import contextlib
import os

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.qsa_attn.block import MICRO, TILE, _unpack_kv
from ..ops import BF16, F32, I32
from .indexer import exclusive_prefix, sum_rows

GROUPS_PER_TILE = TILE // MICRO  # one group = one micro-page of 4 tokens


UNION_CHUNK = 256  # output slots compacted per pass (bounds the [slots, S] temporaries)


# VLLM_TPU_MK_UNION_STACK (default on): stack a request's rows in union attention.
STACK_ROWS = os.environ.get("VLLM_TPU_MK_UNION_STACK", "1") != "0"


def union_consts(S: int, KU: int, lane_major: bool = False) -> tuple:
    """Constants of `compact_union` (`compact_consts` at the per-pass slot width)."""
    W = min(KU, UNION_CHUNK)
    li = jax.lax.broadcasted_iota(I32, (128, 128), 0)
    lj = jax.lax.broadcasted_iota(I32, (128, 128), 1)
    if lane_major:  # `compact_union(..., lane_major=True)`
        return (
            (li >= lj).astype(BF16),
            (li < lj).astype(BF16),  # `indexer.exclusive_prefix`'s 128-lane strict triangle
            (li == lj).astype(BF16),
            jax.lax.broadcasted_iota(I32, (1, W), 1),
            jax.lax.broadcasted_iota(I32, (S, 1), 0),
            jax.lax.broadcasted_iota(I32, (128, 1), 0),
        )
    return (
        (li <= lj).astype(BF16),
        (li < lj).astype(BF16),  # `indexer.exclusive_prefix`'s 128-lane strict triangle
        jax.lax.broadcasted_iota(I32, (W, 1), 0),
        jax.lax.broadcasted_iota(I32, (1, S), 1),
        jax.lax.broadcasted_iota(I32, (W, 128), 1),
    )


def union_chunk_consts(CR: int, KU: int, lane_major: bool = False) -> tuple:
    """Constants of `compact_union_chunk` for `CR`-row chunks."""
    W = min(KU, UNION_CHUNK)
    li = jax.lax.broadcasted_iota(I32, (128, 128), 0)
    lj = jax.lax.broadcasted_iota(I32, (128, 128), 1)
    si = jax.lax.broadcasted_iota(I32, (CR, CR), 0)
    sj = jax.lax.broadcasted_iota(I32, (CR, CR), 1)
    if lane_major:  # `compact_union_chunk(..., lane_major=True)`
        return (
            (li >= lj).astype(BF16),
            (si < sj).astype(BF16),
            (li == lj).astype(BF16),
            jax.lax.broadcasted_iota(I32, (1, W), 1),
            jax.lax.broadcasted_iota(I32, (CR, 1), 0),
            jax.lax.broadcasted_iota(I32, (128, 1), 0),
        )
    return (
        (li <= lj).astype(BF16),
        (si < sj).astype(BF16),
        jax.lax.broadcasted_iota(I32, (W, 1), 0),
        jax.lax.broadcasted_iota(I32, (1, CR), 1),
        jax.lax.broadcasted_iota(I32, (W, 128), 1),
    )


def compact_union(
    code: jax.Array, KU: int, consts: tuple, lane_major: bool = False
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Ascending union of the groups with a nonzero row code, as lane-major `[1, KU]`
    ids and
    codes (slots past the count: 0) and the count `[1, 1]`.

    `code [S, 128]` int32 (lane = group id, value = row bits < 2^T). The compaction of
    `indexer.compact_row` with every coded lane selected, `UNION_CHUNK` output slots per
    pass
    (the `[slots, S]` one-hot temporaries overflow VMEM: all 2048 slots at once 62.8 of
    62.1 MiB at
    16 rows, j-c89e9978; 512 per pass still 62.4, j-bfd3a7f8); each slot's code is
    gathered from its group's sublane row by the
    same one-hot matmul (codes <= 15 are exact in BF16). `lane_major`
    (VLLM_TPU_MK_QSA_LANE_MAJOR_SELECT): `_compact_union_lm`, same ids, codes and count.
    """
    if lane_major:
        return _compact_union_lm(code, KU, consts)
    U, Us, jcol0, srow, lane = consts
    S, W = code.shape[0], jcol0.shape[0]
    sel = code > 0
    P = jnp.dot(sel.astype(F32).astype(BF16), U, preferred_element_type=F32)
    c_row = jnp.transpose(jnp.broadcast_to(P[:, 127:128], (S, 128)))[0:1, :]
    O_row = exclusive_prefix(c_row, Us)
    E_row = O_row + c_row
    Pb, codeb = P.astype(BF16), code.astype(F32).astype(BF16)
    n = jnp.sum(sel.astype(I32), axis=(0, 1), keepdims=True)
    ids_out, codes_out = [], []
    assert KU % W == 0, f"union window {W} must divide KU={KU} (a partial last window overruns the store)"
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
        ids_out.append(
            jnp.transpose(jnp.broadcast_to(ids, (W, 128)))[0:1, :].astype(I32)
        )
        codes_out.append(
            jnp.transpose(jnp.broadcast_to(codes, (W, 128)))[0:1, :].astype(I32)
        )
    cat = lambda xs: xs[0] if len(xs) == 1 else jnp.concatenate(xs, axis=1)  # noqa: E731
    return cat(ids_out), cat(codes_out), n


def _union_lhs(code: jax.Array, consts: tuple, O_row=None) -> tuple:
    """Lane-major union compaction operands of `code [S, 128]`: the transposed in-row ranks
    `PT` and codes `CT` (`[128, S]` from the MXU), the per-row count `c_row [1, S]`, the
    per-row end slots `E [S, W]` (each row's value broadcast over the slot lanes), and the
    one-hot gather's LHS `[PT; CT; O_hi; O_lo; 0] [272, S]` (row offsets split into bytes:
    exact in BF16). `O_row` (f32 `[1, S]`): the rows' exclusive offsets, else
    `exclusive_prefix` of the counts."""
    Ut, Us, eye, jcol0 = consts[:4]
    S, W = code.shape[0], jcol0.shape[1]
    dn = (((1,), (1,)), ((), ()))
    PT = jax.lax.dot_general(  # PT[l, r]: coded lanes <= l of row r
        Ut, (code > 0).astype(F32).astype(BF16), dn, preferred_element_type=F32
    )
    CT = jax.lax.dot_general(eye, code.astype(F32).astype(BF16), dn, preferred_element_type=F32)
    c_row = PT[127:128, :]
    if O_row is None:
        O_row = exclusive_prefix(c_row, Us)
    else:
        O_row = O_row(c_row)
    E_col = jnp.transpose(jnp.broadcast_to(O_row + c_row, (128, S)))  # [S, 128]: E[r]
    E = jnp.concatenate([E_col] * -(-W // 128), axis=1)[:, :W]
    O_hi = jnp.floor(O_row * (1.0 / 256.0))
    lhs = jnp.concatenate(
        [PT, CT, O_hi, O_row - 256.0 * O_hi, jnp.zeros((14, S), F32)], axis=0
    ).astype(BF16)  # [272, S]
    return c_row, E, lhs


def _union_slots(jf: jax.Array, E: jax.Array, lhs: jax.Array, consts: tuple) -> tuple:
    """Sublane row `r_j`, lane `lane_j` and code `code_j` (`[1, W]`) of union slots `jf`."""
    scol, lcol = consts[4], consts[5]
    r_j = sum_rows((E <= jf).astype(I32))
    oh = (scol == r_j).astype(F32).astype(BF16)  # [S, W]
    g = jnp.dot(lhs, oh, preferred_element_type=F32)  # [272, W]
    O_rj = 256.0 * g[256:257] + g[257:258]
    lane_j = sum_rows((g[:128] <= jf - O_rj).astype(I32))
    code_j = sum_rows(jnp.where(lcol == lane_j, g[128:256], 0.0))
    return r_j, lane_j, code_j


def _compact_union_lm(code: jax.Array, KU: int, consts: tuple) -> tuple:
    """`compact_union` lane-major over the slots: the in-row ranks and codes come transposed
    from the MXU, each slot's row and lane are sums over sublanes, and its in-row ranks, code
    and row offset are gathered by one one-hot matmul (no per-slot lane reductions, no
    transposes of the outputs). Same integers: same ids, codes and count."""
    W = consts[3].shape[1]
    c_row, E, lhs = _union_lhs(code, consts)
    n = jnp.sum(c_row, axis=1, keepdims=True)
    ids_out, codes_out = [], []
    assert KU % W == 0, f"union window {W} must divide KU={KU} (a partial last window overruns the store)"
    for j0 in range(0, KU, W):
        jf = (consts[3] + j0).astype(F32)
        r_j, lane_j, code_j = _union_slots(jf, E, lhs, consts)
        live = jf < n
        ids_out.append(jnp.where(live, r_j * 128 + lane_j, 0))
        codes_out.append(jnp.where(live, code_j, 0.0).astype(I32))
    cat = lambda xs: xs[0] if len(xs) == 1 else jnp.concatenate(xs, axis=1)  # noqa: E731
    return cat(ids_out), cat(codes_out), n.astype(I32)


def compact_union_chunk(code: jax.Array, tot: jax.Array, row_of, KU: int, consts: tuple,
                        store, id_of=None, lane_major: bool = False) -> tuple:
    """One row chunk of `compact_union` (`VLLM_TPU_MK_QSA_SELECT_NVIS`): the coded lanes of
    `code [CR, 128]` take union slots `[tot, tot + count)` (`tot`: scalar int32 count of the
    chunks before it, which hold lower ids). For each `UNION_CHUNK` slot window that meets
    that range, `store(j0, ids, codes)` gets the window's `[1, W]` int32 ids / codes (0
    outside the range; chunks add up). `row_of(r)`: sublane row of the chunk's row `r`.
    `id_of(r, l)` (optional): group id of the chunk's row `r`, lane `l` (default `row_of(r)
    * 128 + l`), for chunks whose lanes are not their groups' lanes (`_nvis_union`'s lane
    pack). Same integer arithmetic as `compact_union`. Returns the new `tot`. `lane_major`
    (VLLM_TPU_MK_QSA_LANE_MAJOR_SELECT; `union_chunk_consts(..., lane_major=True)`): the
    lane-major compaction of `_compact_union_lm` (`row_of` / `id_of` get `[1, W]` rows /
    lanes), same ids and codes."""
    if lane_major:
        return _compact_union_chunk_lm(code, tot, row_of, KU, consts, store, id_of)
    U, tri, jcol0, srow, lane = consts
    CR, W = code.shape[0], jcol0.shape[0]
    sel = code > 0
    P = jnp.dot(sel.astype(F32).astype(BF16), U, preferred_element_type=F32)
    c_row = jnp.transpose(jnp.broadcast_to(P[:, 127:128], (CR, 128)))[0:1, :]
    totf = jnp.zeros((1, 1), F32) + tot.astype(F32)
    O_row = jnp.dot(c_row.astype(BF16), tri, preferred_element_type=F32) + totf
    E_row = O_row + c_row
    Pb, codeb = P.astype(BF16), code.astype(F32).astype(BF16)
    n = jnp.sum(sel.astype(I32))
    assert KU % W == 0, f"union window {W} must divide KU={KU} (a partial last window overruns the store)"
    for j0 in range(0, KU, W):

        @pl.when((j0 < tot + n) & (j0 + W > tot))
        def _(j0=j0):
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
            live = (jcol >= tot) & (jcol < tot + n)
            gid = row_of(r_j) * 128 + lane_j if id_of is None else id_of(r_j, lane_j)
            ids = jnp.where(live, gid, 0).astype(F32)
            codes = jnp.where(live, code_j, 0.0)
            store(
                j0,
                jnp.transpose(jnp.broadcast_to(ids, (W, 128)))[0:1, :].astype(I32),
                jnp.transpose(jnp.broadcast_to(codes, (W, 128)))[0:1, :].astype(I32),
            )

    return tot + n


def _compact_union_chunk_lm(code, tot, row_of, KU: int, consts: tuple, store, id_of) -> Any:
    """`compact_union_chunk`, lane-major over the slots (`_compact_union_lm` with the chunk's
    offsets `tot + c_row . tri`)."""
    tri, jcol0 = consts[1], consts[3]
    W = jcol0.shape[1]
    totf = jnp.zeros((1, 1), F32) + tot.astype(F32)
    c_row, E, lhs = _union_lhs(
        code, consts,
        lambda c: jnp.dot(c.astype(BF16), tri, preferred_element_type=F32) + totf,
    )
    n = jnp.sum((code > 0).astype(I32))
    assert KU % W == 0, f"union window {W} must divide KU={KU} (a partial last window overruns the store)"
    for j0 in range(0, KU, W):

        @pl.when((j0 < tot + n) & (j0 + W > tot))
        def _(j0=j0):
            jcol = jcol0 + j0
            r_j, lane_j, code_j = _union_slots(jcol.astype(F32), E, lhs, consts)
            live = (jcol >= tot) & (jcol < tot + n)
            gid = row_of(r_j) * 128 + lane_j if id_of is None else id_of(r_j, lane_j)
            store(j0, jnp.where(live, gid, 0), jnp.where(live, code_j, 0.0).astype(I32))

    return tot + n


def lane_pack(ev: jax.Array, od: jax.Array, r: int, gpp: int, roll) -> jax.Array:
    """Main-KV DCP2 lane pack (VLLM_TPU_MK_QSA_UNION_LANEPACK): pair rank `r` owns the lanes
    `l` with `(l // gpp) % 2 == r` of every code row (`128 % (2 * gpp) == 0`). Returns the
    `[H, 128]` rows whose lanes `[0, 64)` are `ev`'s owned lanes and `[64, 128)` `od`'s, each
    in ascending lane order (`ev` / `od`: code rows 2p / 2p + 1). `roll`: `pltpu.roll` in a
    kernel (`jnp.roll` semantics). Integer selects only: exact."""
    lane = jax.lax.broadcasted_iota(I32, ev.shape, 1)
    out = None
    for off, x in ((0, ev), (64, od)):
        for b in range(64 // gpp):  # owned block b: lanes [(2b + r) gpp, +gpp) -> off + b gpp
            sh = (off + b * gpp - (2 * b + r) * gpp) % 128
            y = roll(x, sh, 1) if sh else x
            lo = off + b * gpp
            inb = (lane >= lo) & (lane < lo + gpp)
            out = jnp.where(inb, y, 0) if out is None else jnp.where(inb, y, out)
    return out


def lane_pack_id(row: jax.Array, lane: jax.Array, r: int, gpp: int) -> jax.Array:
    """Group id of `lane_pack` lane `lane` of the packed row whose even source row is
    `row`: source row `row + lane // 64`, owned block `(lane % 64) // gpp` of rank `r`."""
    lw = jax.lax.rem(lane, 64)
    src = (2 * (lw // gpp) + r) * gpp + jax.lax.rem(lw, gpp)
    return (row + lane // 64) * 128 + src


def unpack_union_ids(ids: jax.Array, n: jax.Array, r: int, gpp: int) -> jax.Array:
    """Group ids of a `compact_union` over pair rank `r`'s packed rows (`ids [1, KU]` packed
    positions, `n` the count): packed row p is code row 2p + r (`gpp` = 128) or `lane_pack`'s
    rows 2p / 2p + 1. Slots past `n` stay 0."""
    prow, plane = ids // 128, jax.lax.rem(ids, 128)
    gid = (2 * prow + r) * 128 + plane if gpp == 128 else lane_pack_id(2 * prow, plane, r, gpp)
    return jnp.where(jax.lax.broadcasted_iota(I32, ids.shape, 1) < n, gid, 0)


def physical_ids(
    ids: jax.Array, tab: Any, per_page: int, dcp: int = 1, n_tab: Any = None,
    table_bytes: int | None = None,
) -> jax.Array:
    """Physical micro-page ids `[1, 128]` of 128 request-local group ids `ids [1, 128]`: what
    `attend_request` computes per copy (`tab[v >> shift] * per_page + (v & (per_page - 1))`), as
    one vector pass against the request's page-table row `tab [1, pps]` (lane select + lane sum:
    exact int32), so each copy issues from a single SMEM load (RPAd's precomputed page ids).
    `dcp` = 2: a table column covers `2 * per_page` groups (main-KV DCP; the row is unchanged).
    `n_tab` (traced; `wholephysmxu`): `tab` is a `[1, pps]` VMEM ref and only its first `n_tab`
    128-lane chunks hold the ids' columns, so only those are read. Each chunk is gathered on the
    MXU with the ids kept lane-major (no transposes or lane reductions): a one-hot `[column,
    id]` matrix times the chunk's entries split into `table_bytes` bytes (entries < 2^(8 *
    table_bytes)). Every product sum has at most one nonzero term, a byte (exact in BF16 and
    F32), so the result equals the vector pass for every id whose column is below `n_tab * 128`;
    an id past them maps to page 0, as an id past the row does in the vector pass.
    """
    shift = per_page.bit_length() - 1
    cshift = shift + dcp.bit_length() - 1
    if n_tab is not None:
        assert table_bytes is not None and 1 <= table_bytes <= 4, table_bytes
        cols = ids >> cshift
        byte = jax.lax.broadcasted_iota(I32, (8, 128), 0)  # LHS row k: byte k of each entry

        def chunk(c: Any, acc: Any) -> Any:
            lo = pl.multiple_of(c * 128, 128)
            row = jax.lax.broadcasted_iota(I32, (128, 128), 0) + lo
            hot = (cols == row).astype(F32).astype(BF16)  # [column, id]
            t = jnp.broadcast_to(tab[:, pl.ds(lo, 128)], (8, 128))
            b = jnp.where(byte < table_bytes, (t >> (8 * byte)) & 255, 0)
            return acc + jnp.dot(b.astype(F32).astype(BF16), hot, preferred_element_type=F32)

        acc = jax.lax.fori_loop(0, n_tab, chunk, jnp.zeros((8, 128), F32)).astype(I32)
        page = acc[0:1]
        for k in range(1, table_bytes):
            page = page + (acc[k : k + 1] << (8 * k))
        return page * per_page + (ids & (per_page - 1))
    col = jnp.transpose(jnp.broadcast_to(ids, (128, 128)))[:, 0:1]
    lane = jax.lax.broadcasted_iota(I32, (1, tab.shape[1]), 1)
    page = jnp.sum(jnp.where((col >> cshift) == lane, tab, 0), axis=1, keepdims=True)
    phys = page * per_page + (col & (per_page - 1))
    return jnp.transpose(jnp.broadcast_to(phys, (128, 128)))[0:1, :]


def table_chunks(n_vis: Any, per_page: int, dcp: int, width: int) -> Any:
    """128-lane chunks of a `width`-column page-table row holding the columns of groups
    `< n_vis` (at least one)."""
    assert width % 128 == 0, width
    cshift = per_page.bit_length() - 1 + dcp.bit_length() - 1
    last_col = (jnp.maximum(n_vis, 1) - 1) >> cshift
    return jnp.minimum(last_col // 128 + 1, width // 128)


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
    phys: bool = False,
    dcp: int = 1,
    sc: Any = None,
    ids_row: Any = 0,
    stream: Any = None,
    lite: bool = False,
    phys_rows: bool = False,
) -> list[Any]:
    """Gated sparse attention of one request's `T` rows over their union: `T` `[nh, 256]` F32.

    `ids` SMEM `[1, KU + 128]` request-local group ids of the union (ascending), `codes` a
    `[1, KU]` VMEM ref of their row bits, `n_union` the count; `table[table_row, v >> shift]`
    maps a group to its micro-page as `attend_row`'s `table` does. `qs` / `gates` `[nh, 256]` and
    `phys`: `ids` already holds physical micro-page ids (`physical_ids`) for every slot a tile
    reads (`< ceil(n_union / 64) * 64`); each copy is then one SMEM load, no table, no clamp.
    `dcp` = 2 (main-KV DCP): a table column covers `2 * per_page` groups; the caller zeroes the
    codes of groups this rank does not own (their slots then hold the pair rank's rows: read,
    sanitised, never scored), and each row returns its ungated partial `(attn [nh, 256], lse
    [nh, 1])` (`lse = -inf`: no owned token) for the pair's LSE combine.
    `has` (the row has pool tokens) per row. `recent = (k [8, 256], v [8, 256], ok_rows, ok_any
    [8, 1])` in pool units, `ok_rows` `T` `[1, 8]` masks. Each tile is copied and unpacked once and
    scored per row (`[nh, 256]` queries, as `attend_row`); scaling follows `attend_row` without
    `fold`: K by `scale_k`, scores by `sm_scale`, V by `scale_v`, per tile.
    `ids_row`: the row of `ids` holding this request's ids.
    `stream` (cross-request prefetch) = `(base, first, nxt)`: the request's tiles are positions
    `[0, ntiles)` of one tile stream over the request loop, ring slot `(base + position) %
    nslots`; positions from `ntiles` on are the next request's tiles, issued into the slots this
    request's last tiles free. `first`: no tile of this request was issued by the previous one
    (else its first `min(ntiles, nslots - 1)` were). `nxt = (live, ids_row, table_row, n_union)`
    of the next request (its ids already staged in `ids[ids_row]`). Every issued tile is waited
    by its own request; the caller carries `base + ntiles` to the next request.
    `lite` (VLLM_TPU_MK_UATTN_LITE): fewer per-tile ops, same bytes. One code-expand matmul per
    tile (the tile's half picks its expand matrix first), and under `dcp` > 1 one select for
    V's NaN sanitize and scale, with no length mask: the sanitize leaves every V finite, and a
    slot past `n_union` has `p` = 0 exactly, so its `p * v` adds an exact zero as before.
    `phys_rows` (with `phys`; `wholephysmxu`): on TPU each copy reads the pool viewed as rows of
    micro-pages, so its source address is one id times the row stride (same bytes as
    `pool.at[v >> shift, ...]`; CPU interpret cannot index a reshaped ref and keeps the per-page
    form).
    """
    sc = sc or (lambda *_: contextlib.nullcontext())
    per_page = pool.shape[1] // MICRO
    nslots = ring.shape[0]
    ncols = ids.shape[1]
    assert per_page & (per_page - 1) == 0, per_page
    shift = per_page.bit_length() - 1
    cshift = shift + dcp.bit_length() - 1
    length = n_union * MICRO
    ntiles = (length + TILE - 1) // TILE
    assert not phys_rows or phys, "micro-page rows: physical ids"
    micro_pages = (
        pool.reshape(pool.shape[0] * per_page, MICRO, *pool.shape[2:])
        if phys_rows and not interpret
        else None
    )

    def copies(t: Any, s: Any, row: Any = None, trow: Any = None) -> list[Any]:
        out = []
        base = t * GROUPS_PER_TILE
        row = ids_row if row is None else row
        trow = table_row if trow is None else trow
        for j in range(GROUPS_PER_TILE):
            if micro_pages is not None:  # same bytes as `pool.at[v >> shift, ...]`
                out.append(
                    pltpu.make_async_copy(
                        micro_pages.at[ids[row, base + j]],
                        ring.at[s, pl.ds(j * MICRO, MICRO)],
                        sems.at[s],
                    )
                )
                continue
            if phys:
                v = ids[row, base + j]
            else:
                v = ids[row, jnp.minimum(base + j, ncols - 1)]
                v = table[trow, v >> cshift] * per_page + (v & (per_page - 1))
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

    if stream is None:
        for k in range(nslots - 1):

            @pl.when(k < ntiles)
            def _(k=k):
                start(k, k)

        slot = lambda t: jax.lax.rem(t, nslots)  # noqa: E731

        def issue(p: Any) -> None:
            @pl.when(p < ntiles)
            def _():
                start(p, slot(p))

    else:
        sbase, first, (nlive, nrow, ntrow, n_next) = stream
        nt_next = jnp.where(nlive, (n_next * MICRO + TILE - 1) // TILE, 0)
        slot = lambda p: jax.lax.rem(sbase + p, nslots)  # noqa: E731

        def issue(p: Any, own_ok: Any = True) -> None:
            # Position p: this request's tile p, or the next request's tile p - ntiles.
            @pl.when((p < ntiles) & own_ok)
            def _():
                start(p, slot(p))

            @pl.when((p >= ntiles) & (p - ntiles < nt_next))
            def _():
                for cp in copies(p - ntiles, slot(p), nrow, ntrow):
                    cp.start()

        for k in range(nslots - 1):
            issue(k, first)

    # A tile's 64 group codes sit in one half of an aligned 128-lane window; expanding
    # the whole
    # window with the half's one-hot map gives each of the tile's 256 tokens its group's
    # code
    # (token i: group slot i // 4), with no sub-vreg lane slice.
    gi = jax.lax.broadcasted_iota(I32, (2 * GROUPS_PER_TILE, TILE), 0)
    ti = jax.lax.broadcasted_iota(I32, (2 * GROUPS_PER_TILE, TILE), 1)
    expand = [
        (ti // MICRO == gi - h * GROUPS_PER_TILE).astype(F32).astype(BF16)
        for h in (0, 1)
    ]
    qf = [q.astype(F32) for q in qs]
    # The request's T rows stacked: one [T * nh, 256] score / PV matmul and one masked online
    # softmax per tile (each output row is the same dot and elementwise math as per row).
    stack = STACK_ROWS
    qcat = jnp.concatenate(qf, axis=0) if stack else None
    rowbit = jax.lax.broadcasted_iota(I32, (T * nh, 1), 0) // nh

    def tile(t: Any, carry: Any) -> Any:
        s = slot(t)
        with sc("u_wait", 3):
            wait(t, s)
        issue(t + nslots - 1)
        return _tile_body(t, s, carry)

    def _tile_body(t: Any, s: Any, carry: Any) -> Any:
      with sc("u_unpack", 3):
        win = codes[:, pl.ds(pl.multiple_of((t // 2) * 128, 128), 128)].astype(F32).astype(BF16)
        if lite:
            ex = jnp.where(jax.lax.rem(t, 2) == 0, expand[0], expand[1])
            tok_code = jnp.dot(win, ex, preferred_element_type=F32).astype(I32)
        else:
            tok_code = jnp.where(
                jax.lax.rem(t, 2) == 0,
                jnp.dot(win, expand[0], preferred_element_type=F32),
                jnp.dot(win, expand[1], preferred_element_type=F32),
            ).astype(I32)
        tok = t * TILE + jax.lax.broadcasted_iota(I32, (1, TILE), 1)
        k8, v8 = _unpack_kv(ring.at[s], off, interpret)
        key = k8.astype(F32) * scale_k
        tok_col = t * TILE + jax.lax.broadcasted_iota(I32, (TILE, 1), 0)
        value = v8.astype(F32)
        if lite and dcp > 1:  # sanitize + scale in one select; no length mask (see `lite`)
            value = jnp.where(value == value, value * scale_v, 0.0)
        else:
            if dcp > 1:  # unowned slots hold the pair rank's rows (FP8: may be NaN)
                value = jnp.where(value == value, value, 0.0)
            value = jnp.where(tok_col < length, value * scale_v, 0.0)
      with sc("u_rows", 3):
        if stack:
            m, l, acc = carry
            valid = (tok < length) & ((jnp.right_shift(tok_code, rowbit) & 1) != 0)
            score = jax.lax.dot_general(
                qcat, key, (((1,), (1,)), ((), ())), preferred_element_type=F32
            )
            score = jnp.where(valid, score * sm_scale, -jnp.inf)
            m_new = jnp.maximum(m, jnp.max(score, axis=-1, keepdims=True))
            safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)
            p = jnp.where(valid, jnp.exp(score - safe), 0.0)
            alpha = jnp.where(
                m == -jnp.inf, jnp.where(m_new == -jnp.inf, 1.0, 0.0), jnp.exp(m - safe)
            )
            return (
                m_new,
                l * alpha + jnp.sum(p, axis=-1, keepdims=True),
                acc * alpha + jnp.dot(p, value, preferred_element_type=F32),
            )
        out = []
        for r in range(T):
            m, l_sum, acc = carry[r]
            valid = (tok < length) & ((tok_code & (1 << r)) != 0)
            score = jax.lax.dot_general(
                qf[r], key, (((1,), (1,)), ((), ())), preferred_element_type=F32
            )
            score = jnp.where(valid, score * sm_scale, -jnp.inf)
            m_new = jnp.maximum(m, jnp.max(score, axis=-1, keepdims=True))
            safe = jnp.where(
                m_new == -jnp.inf, 0.0, m_new
            )  # no token of this row so far
            p = jnp.where(valid, jnp.exp(score - safe), 0.0)
            alpha = jnp.where(
                m == -jnp.inf, jnp.where(m_new == -jnp.inf, 1.0, 0.0), jnp.exp(m - safe)
            )
            out.append(
                (
                    m_new,
                    l_sum * alpha + jnp.sum(p, axis=-1, keepdims=True),
                    acc * alpha + jnp.dot(p, value, preferred_element_type=F32),
                )
            )
        return tuple(out)

    dim = qs[0].shape[-1]
    init = tuple(
        (
            jnp.full((nh, 1), -jnp.inf, F32),
            jnp.zeros((nh, 1), F32),
            jnp.zeros((nh, dim), F32),
        )
        for _ in range(T)
    )
    if stack:
        init = (
            jnp.full((T * nh, 1), -jnp.inf, F32),
            jnp.zeros((T * nh, 1), F32),
            jnp.zeros((T * nh, dim), F32),
        )
    state = jax.lax.fori_loop(0, ntiles, tile, init)
    if stack:
        state = tuple(
            tuple(x[r * nh : (r + 1) * nh] for x in state) for r in range(T)
        )
    rk, rv, ok_rows, ok_any = recent
    rkey = rk * scale_k
    rval = jnp.where(ok_any, rv * scale_v, 0.0)
    if dcp > 1:
        partial = []
    out = []
    for r in range(T):
        m, l_sum, acc = state[r]
        ok = ok_rows[r]
        sc = jax.lax.dot_general(
            qf[r], rkey, (((1,), (1,)), ((), ())), preferred_element_type=F32
        )
        sc = jnp.where(ok, sc * sm_scale, -jnp.inf)
        m_new = jnp.maximum(m, jnp.max(sc, axis=-1, keepdims=True))
        safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)
        p = jnp.where(ok, jnp.exp(sc - safe), 0.0)
        alpha = jnp.where(m == -jnp.inf, 0.0, jnp.exp(m - safe))
        l_sum = l_sum * alpha + jnp.sum(p, axis=-1, keepdims=True)
        acc = acc * alpha + jnp.dot(p, rval, preferred_element_type=F32)
        attn = acc / jnp.where(l_sum > 0, l_sum, 1.0)
        if dcp > 1:
            lse = jnp.where(l_sum > 0, m_new + jnp.log(jnp.where(l_sum > 0, l_sum, 1.0)), -jnp.inf)
            partial.append((jnp.where(l_sum > 0, attn, 0.0), lse))
            continue
        gated = (
            (
                attn.astype(BF16).astype(F32)
                * jax.nn.sigmoid(gates[r].astype(F32)).astype(BF16).astype(F32)
            )
            .astype(BF16)
            .astype(F32)
        )
        row_has = has[r] | (jnp.max(jnp.where(ok, 1, 0)) > 0)
        out.append(jnp.where(row_has, gated, 0.0))
    return partial if dcp > 1 else out
