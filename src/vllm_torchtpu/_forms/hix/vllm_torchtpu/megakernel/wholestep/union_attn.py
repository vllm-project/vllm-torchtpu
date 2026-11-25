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
from .indexer import exclusive_prefix

GROUPS_PER_TILE = TILE // MICRO  # one group = one micro-page of 4 tokens


UNION_CHUNK = 256  # output slots compacted per pass (bounds the [slots, S] temporaries)


# VLLM_TPU_MK_UNION_STACK (default on): stack a request's rows in union attention.
STACK_ROWS = os.environ.get("VLLM_TPU_MK_UNION_STACK", "1") != "0"


def union_consts(S: int, KU: int) -> tuple:
    """Constants of `compact_union` (`compact_consts` at the per-pass slot width)."""
    W = min(KU, UNION_CHUNK)
    li = jax.lax.broadcasted_iota(I32, (128, 128), 0)
    lj = jax.lax.broadcasted_iota(I32, (128, 128), 1)
    return (
        (li <= lj).astype(BF16),
        (li < lj).astype(BF16),  # `indexer.exclusive_prefix`'s 128-lane strict triangle
        jax.lax.broadcasted_iota(I32, (W, 1), 0),
        jax.lax.broadcasted_iota(I32, (1, S), 1),
        jax.lax.broadcasted_iota(I32, (W, 128), 1),
    )


def compact_union(
    code: jax.Array, KU: int, consts: tuple
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
    same one-hot matmul (codes <= 15 are exact in BF16).
    """
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


def physical_ids(ids: jax.Array, tab: jax.Array, per_page: int, dcp: int = 1) -> jax.Array:
    """Physical micro-page ids `[1, 128]` of 128 request-local group ids `ids [1, 128]`: what
    `attend_request` computes per copy (`tab[v >> shift] * per_page + (v & (per_page - 1))`), as
    one vector pass against the request's page-table row `tab [1, pps]` (lane select + lane sum:
    exact int32), so each copy issues from a single SMEM load (RPAd's precomputed page ids).
    `dcp` = 2: a table column covers `2 * per_page` groups (main-KV DCP; the row is unchanged).
    """
    shift = per_page.bit_length() - 1
    cshift = shift + dcp.bit_length() - 1
    col = jnp.transpose(jnp.broadcast_to(ids, (128, 128)))[:, 0:1]
    lane = jax.lax.broadcasted_iota(I32, (1, tab.shape[1]), 1)
    page = jnp.sum(jnp.where((col >> cshift) == lane, tab, 0), axis=1, keepdims=True)
    phys = page * per_page + (col & (per_page - 1))
    return jnp.transpose(jnp.broadcast_to(phys, (128, 128)))[0:1, :]


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
    # q38 L2-UAISSUE begin
    fast: bool = False,
    # q38 L2-UAISSUE end
    # q38 L2-UNPACK begin
    unpack: bool = False,
    # q38 L2-UNPACK end
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
    # q38 L2-UNPACK begin
    if unpack:  # X-UA-UNPACK (union_unpack.py): the tile body reads its slot with full-vreg strided loads, pair 0 only
        from .union_unpack import check_offset, unpack_kv as _unpack_kv

        check_offset(off)
    else:
        from ..blocks.qsa_attn.block import _unpack_kv
    # q38 L2-UNPACK end
    stock_kv = _kern_stock_kv()  # q38 HIX-KERN K6 (default off)

    def copies(t: Any, s: Any) -> list[Any]:
        # q38 L2-UAISSUE begin
        if fast:  # X-UA-ISSUE stage (b), MM D12 F0a's issue recipe on physical ids (union_issue.fast_copies)
            from . import union_issue

            assert phys, "stage (b) issues from physical micro-page ids"
            return union_issue.fast_copies(
                pool=pool, ring=ring, sems=sems, ids=ids, t=t, s=s, n_union=n_union, interpret=interpret
            )
        # q38 L2-UAISSUE end
        out = []
        base = t * GROUPS_PER_TILE
        for j in range(GROUPS_PER_TILE):
            if phys:
                v = ids[0, base + j]
            else:
                v = ids[0, jnp.minimum(base + j, ncols - 1)]
                v = table[table_row, v >> cshift] * per_page + (v & (per_page - 1))
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
        # q38 L2-UAISSUE begin
        if fast:  # one wait sized to the bytes tile t's live copies move
            from . import union_issue

            return union_issue.fast_wait(ring=ring, sems=sems, t=t, s=s, n_union=n_union)
        # q38 L2-UAISSUE end
        if interpret:
            for cp in copies(t, s):
                cp.wait()
        else:
            pltpu.make_async_copy(ring.at[s], ring.at[s], sems.at[s]).wait()

    for k in range(nslots - 1):

        @pl.when(k < ntiles)
        def _(k=k):
            start(k, k)

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
        s = jax.lax.rem(t, nslots)
        with sc("u_wait", 3):
            wait(t, s)
        ahead = t + nslots - 1

        @pl.when(ahead < ntiles)
        def _():
            start(ahead, jax.lax.rem(ahead, nslots))

        return _tile_body(t, s, carry)

    def _tile_body(t: Any, s: Any, carry: Any) -> Any:
      with sc("u_unpack", 3):
        win = codes[:, pl.ds(pl.multiple_of((t // 2) * 128, 128), 128)].astype(F32).astype(BF16)
        tok_code = jnp.where(
            jax.lax.rem(t, 2) == 0,
            jnp.dot(win, expand[0], preferred_element_type=F32),
            jnp.dot(win, expand[1], preferred_element_type=F32),
        ).astype(I32)
        tok = t * TILE + jax.lax.broadcasted_iota(I32, (1, TILE), 1)
        k8, v8 = _unpack_kv(ring.at[s], off, interpret)
        tok_col = t * TILE + jax.lax.broadcasted_iota(I32, (TILE, 1), 0)
        if stock_kv:  # q38 HIX-KERN K6: FP8 operands straight to the MXU, scales after the dots (stock RPA order; kern_stock_kv below)
            key, value = k8, _kern_value_f8(v8, tok_col < length, dcp > 1)
            if interpret:  # CPU interpret has no mixed f32 x f8 dot: the exact f32 values (the device pushes the FP8 codes)
                key, value = key.astype(F32), value.astype(F32)
        else:
            key = k8.astype(F32) * scale_k
            value = v8.astype(F32)
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
            score = jnp.where(valid, (score * sm_scale) * scale_k if stock_kv else score * sm_scale, -jnp.inf)
            m_new = jnp.maximum(m, jnp.max(score, axis=-1, keepdims=True))
            safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)
            p = jnp.where(valid, jnp.exp(score - safe), 0.0)
            alpha = jnp.where(
                m == -jnp.inf, jnp.where(m_new == -jnp.inf, 1.0, 0.0), jnp.exp(m - safe)
            )
            pv = jnp.dot(p, value, preferred_element_type=F32)
            return (
                m_new,
                l * alpha + jnp.sum(p, axis=-1, keepdims=True),
                acc * alpha + (pv * scale_v if stock_kv else pv),
            )
        out = []
        for r in range(T):
            m, l_sum, acc = carry[r]
            valid = (tok < length) & ((tok_code & (1 << r)) != 0)
            score = jax.lax.dot_general(
                qf[r], key, (((1,), (1,)), ((), ())), preferred_element_type=F32
            )
            score = jnp.where(valid, (score * sm_scale) * scale_k if stock_kv else score * sm_scale, -jnp.inf)
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
                    acc * alpha + (jnp.dot(p, value, preferred_element_type=F32) * scale_v if stock_kv
                                   else jnp.dot(p, value, preferred_element_type=F32)),
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


# ===================================================================== q38 L2-UAPIPE (X-UA-PIPE parts (i) and (ii))
# Everything below is new and is reached only with a nonzero `union_pipe` (VLLM_TPU_MK_UNION_PIPE). It is appended after
# MM's code on purpose: Pallas op locations carry full tracebacks, so no line that the flag-off programs trace may move.
#
# `attend_request` streams a request's union through an `nslots`-slot ring with `D = nslots - 1` tiles in flight: tile t
# waits its slot, then starts tile t + D under `pl.when(t + D < ntiles)`, then runs the tile body; the ring restarts for
# every request, so each request pays the fill of its first D tiles and each tile pays a scalar branch around its 64
# copy starts. `attend_request_pipe` keeps the copies (same pool rows, same 64 descriptors per tile), the tile order and
# the tile body's arithmetic, and changes only when the starts are issued:
#
# (i)  PIPE_COISSUE: the tile loop is split at `t_a = clip(end - D, 0, ntiles)` (`end` = the last stream position + 1).
#      Tiles [0, t_a) start position t + D unconditionally (no branch, so the 64 starts sit in the same basic block as
#      the tile's unpack and matmuls and can co-issue with them); tiles [t_a, ntiles) start nothing. The clamp lives in
#      the trip counts, so no copy is ever issued past the stream's end (nothing to drain, no wasted DMA).
#      PIPE_POST (with (i), census variant): the starts sit after the tile's ring unpack instead of right after its wait.
# (ii) PIPE_XREQ: one stream per layer instead of one per request. Stream position p of request q is its tile p for
#      p < ntiles and request q + 1's tile p - ntiles for ntiles <= p < ntiles + m, m = min(D, tiles of q + 1); position
#      p lives in ring slot (base + p) % nslots, base = the tiles of requests 0 .. q - 1 (mod nslots). So request q + 1's
#      first m tiles are started behind request q's last D tiles (or in q's prologue when q has fewer than D tiles), and
#      request q + 1 finds them in flight. With nslots = D + 1 the slot a start writes held the position consumed one
#      iteration earlier, exactly as in `attend_request`. Request q + 1's union ids must be in SMEM while request q
#      runs: `pipe_request_loop` keeps PIPE_IDS_BUFS rows (q and q + 1 resident, q + 2 in flight; D08 S1 B3's double
#      buffer plus one row), each with its own DMA semaphore.
#
# Exactness: every tile still lands the same 64 pool rows (the same ids, table rows and address math) in a whole slot
# before its wait returns, tiles are consumed in the same order and `_pipe_tile_body` is `attend_request`'s tile body
# operation for operation, so the outputs are bitwise those of `attend_request`; only the slot a tile occupies ((ii))
# and the issue points of the starts change.
#
# q38 L2-UAMERGE (q38-l2-uamerge): X-UA-ISSUE and X-UA-UNPACK run inside the pipe, in any combination with it.
#   `phys` (X-UA-ISSUE stage (a)): `ids` rows hold physical micro-page ids (union_issue.prep's `qx_pv`), so each copy
#        issues from one SMEM load of its stream position's own row, with no table and no clamp (`attend_request`'s
#        `phys` copies).
#   `fast` (stage (b), needs `phys`): the copies are `union_issue.fast_copies` on the position's own ids row
#        (`ids.at[pl.ds(row, 1)]`) and its own request's union count (`nu`: request q + 1's count for the positions
#        that belong to it), so the last tile of either request moves no padded rows; the wait is
#        `union_issue.fast_wait`, sized to the bytes tile t's live copies move (tile t is always this request's own).
#   `unpack` (X-UA-UNPACK): the tile body takes K / V from `union_unpack.unpack_kv` (static offset 0, checked), not
#        `_unpack_kv`.
#   Each is bitwise against the same arm without the pipe for the reasons given in union_issue.py / union_unpack.py; the
#   pipe moves only the issue points and the ring slots, so the composition is bitwise (tests/megakernel/
#   test_l2_uamerge.py).
PIPE_COISSUE = 1
PIPE_XREQ = 2
PIPE_POST = 4
PIPE_MODES = (1, 2, 3, 5, 7)
PIPE_IDS_BUFS = 3


def pipe_mode(v: Any) -> int:
    """The `union_pipe` bit mask (0 = off), validated: PIPE_POST needs PIPE_COISSUE."""
    v = int(v or 0)
    if v and v not in PIPE_MODES:
        raise ValueError(f"union_pipe={v}: one of {PIPE_MODES} (1 co-issue, 2 cross-request, 4 starts after the unpack)")
    return v


class PipeNext:
    """(ii): what request q's stream needs about request q + 1 (`has` False: q is the layer's last request)."""

    def __init__(self, *, ids_row: Any, n_union: Any, table_row: Any, has: Any) -> None:
        self.ids_row, self.n_union, self.table_row, self.has = ids_row, n_union, table_row, has


def _pipe_wait(pool: Any, ring: Any, sems: Any, s: Any, interpret: bool) -> None:
    """Wait for slot `s`'s 64 copies: one whole-slot wait on device (`attend_request`'s), per copy in interpret mode
    with stand-in sources of the copies' shape (a wait depends only on the semaphore and the byte count)."""
    if interpret:
        for j in range(GROUPS_PER_TILE):
            pltpu.make_async_copy(
                pool.at[0, pl.ds(0, MICRO)], ring.at[s, pl.ds(j * MICRO, MICRO)], sems.at[s]
            ).wait()
    else:
        pltpu.make_async_copy(ring.at[s], ring.at[s], sems.at[s]).wait()


def attend_request_pipe(
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
    pipe: int,
    ids_row: Any = 0,
    base: Any = 0,
    inflight: Any = False,
    nxt: PipeNext | None = None,
    dcp: int = 1,
    sc: Any = None,
    phys: bool = False,
    fast: bool = False,
    unpack: bool = False,
) -> tuple[list[Any], Any]:
    """`attend_request` with the X-UA-PIPE issue schedule (`pipe`, module note above): the same
    outputs bit for bit, plus this request's tile count (the caller's stream position advance under (ii)).

    `ids[ids_row]` holds this request's union ids (`attend_request`'s `ids[0]`). Under (ii): `base` = the ring position
    of this request's tile 0, `inflight` = its first min(D, ntiles) tiles were started by the previous request (every
    request but the layer's first), `nxt` = request q + 1 (`PipeNext`). `phys` / `fast` / `unpack`: X-UA-ISSUE stage
    (a) / stages (a) + (b) and X-UA-UNPACK (module note, q38 L2-UAMERGE)."""
    pipe = pipe_mode(pipe)
    assert pipe, "attend_request_pipe needs a nonzero union_pipe"
    xreq, coissue, post = bool(pipe & PIPE_XREQ), bool(pipe & PIPE_COISSUE), bool(pipe & PIPE_POST)
    assert xreq == (nxt is not None), (pipe, nxt)
    sc = sc or (lambda *_: contextlib.nullcontext())
    per_page = pool.shape[1] // MICRO
    nslots = ring.shape[0]
    D = nslots - 1  # tiles in flight ahead of the one being read
    assert D >= 1, nslots
    ncols = ids.shape[1]
    assert per_page & (per_page - 1) == 0, per_page
    shift = per_page.bit_length() - 1
    cshift = shift + dcp.bit_length() - 1
    length = n_union * MICRO
    ntiles = (length + TILE - 1) // TILE
    assert phys or not fast, "X-UA-ISSUE stage (b) issues from physical micro-page ids"
    if unpack:  # q38 L2-UAMERGE: X-UA-UNPACK's strided reader (static offset 0) in the pipe's tile body
        from . import union_unpack

        union_unpack.check_offset(off)

        def read_kv(slot_ref: Any) -> tuple[jax.Array, jax.Array]:
            return union_unpack.unpack_kv(slot_ref, off, interpret)

    else:

        def read_kv(slot_ref: Any) -> tuple[jax.Array, jax.Array]:
            return _unpack_kv(slot_ref, off, interpret)

    if xreq:
        n_next = (nxt.n_union * MICRO + TILE - 1) // TILE
        m = jnp.where(nxt.has, jnp.minimum(n_next, D), 0)
        end = ntiles + m  # stream positions [0, end): this request's tiles, then request q + 1's first m
    else:
        end = ntiles

    def slot(p: Any) -> Any:
        return jax.lax.rem(base + p, nslots) if xreq else jax.lax.rem(p, nslots)

    def copies(tile: Any, s: Any, row: Any, trow: Any, nu: Any) -> list[Any]:
        """`attend_request.copies` for tile `tile` of the request whose ids sit in `ids[row]`, whose page-table row is
        `trow` and whose union count is `nu`, into slot `s`."""
        if fast:  # q38 L2-UAMERGE: X-UA-ISSUE stage (b) on the position's own ids row and union count
            from . import union_issue

            cps = union_issue.fast_copies(
                pool=pool, ring=ring, sems=sems, ids=ids.at[pl.ds(row, 1)], t=tile, s=s, n_union=nu, interpret=interpret
            )
            return cps  # each copy keeps X-UA-ISSUE's live-slot predicate (union_issue.When)
        out = []
        col0 = tile * GROUPS_PER_TILE
        for j in range(GROUPS_PER_TILE):
            if phys:  # q38 L2-UAMERGE: X-UA-ISSUE stage (a), a physical micro-page id (no table, no clamp)
                v = ids[row, col0 + j]
            else:
                v = ids[row, jnp.minimum(col0 + j, ncols - 1)]
                v = table[trow, v >> cshift] * per_page + (v & (per_page - 1))
            out.append(
                pltpu.make_async_copy(
                    pool.at[v >> shift, pl.ds((v & (per_page - 1)) * MICRO, MICRO)],
                    ring.at[s, pl.ds(j * MICRO, MICRO)],
                    sems.at[s],
                )
            )
        return out

    def start_pos(p: Any) -> None:
        """Start stream position `p` (< `end`)."""
        if xreq:
            own = p < ntiles
            tile = jnp.where(own, p, p - ntiles)
            row = jnp.where(own, ids_row, nxt.ids_row)
            trow = jnp.where(own, table_row, nxt.table_row)
            nu = jnp.where(own, n_union, nxt.n_union)  # q38 L2-UAMERGE: the live count of the position's request
        else:
            tile, row, trow, nu = p, ids_row, table_row, n_union
        for cp in copies(tile, slot(p), row, trow, nu):
            cp.start()

    # Prologue: stream positions [0, D) that are not in flight yet (a request's own first tiles are, under (ii), unless
    # it is the layer's first; positions past its own tiles belong to request q + 1).
    for k in range(D):
        if xreq:
            need = (k < end) & jnp.logical_not(jnp.logical_and(inflight, k < ntiles))
        else:
            need = k < ntiles

        @pl.when(need)
        def _(k=k):
            start_pos(k)

    # A tile's 64 group codes sit in one half of an aligned 128-lane window (`attend_request`).
    gi = jax.lax.broadcasted_iota(I32, (2 * GROUPS_PER_TILE, TILE), 0)
    ti = jax.lax.broadcasted_iota(I32, (2 * GROUPS_PER_TILE, TILE), 1)
    expand = [
        (ti // MICRO == gi - h * GROUPS_PER_TILE).astype(F32).astype(BF16)
        for h in (0, 1)
    ]
    qf = [q.astype(F32) for q in qs]
    stack = STACK_ROWS
    qcat = jnp.concatenate(qf, axis=0) if stack else None
    rowbit = jax.lax.broadcasted_iota(I32, (T * nh, 1), 0) // nh

    def tile_body(t: Any, s: Any, carry: Any, mid: Any = None) -> Any:
        """`attend_request._tile_body`, operation for operation; `mid()` (PIPE_POST) runs after the unpack."""
        with sc("u_unpack", 3):
            win = codes[:, pl.ds(pl.multiple_of((t // 2) * 128, 128), 128)].astype(F32).astype(BF16)
            tok_code = jnp.where(
                jax.lax.rem(t, 2) == 0,
                jnp.dot(win, expand[0], preferred_element_type=F32),
                jnp.dot(win, expand[1], preferred_element_type=F32),
            ).astype(I32)
            tok = t * TILE + jax.lax.broadcasted_iota(I32, (1, TILE), 1)
            k8, v8 = read_kv(ring.at[s])  # q38 L2-UAMERGE: X-UA-UNPACK's reader with `unpack`, else `_unpack_kv`
            key = k8.astype(F32) * scale_k
            tok_col = t * TILE + jax.lax.broadcasted_iota(I32, (TILE, 1), 0)
            value = v8.astype(F32)
            if dcp > 1:  # unowned slots hold the pair rank's rows (FP8: may be NaN)
                value = jnp.where(value == value, value, 0.0)
            value = jnp.where(tok_col < length, value * scale_v, 0.0)
        if mid is not None:
            mid()
        with sc("u_rows", 3):
            if stack:
                m_, l_, acc = carry
                valid = (tok < length) & ((jnp.right_shift(tok_code, rowbit) & 1) != 0)
                score = jax.lax.dot_general(
                    qcat, key, (((1,), (1,)), ((), ())), preferred_element_type=F32
                )
                score = jnp.where(valid, score * sm_scale, -jnp.inf)
                m_new = jnp.maximum(m_, jnp.max(score, axis=-1, keepdims=True))
                safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)
                p = jnp.where(valid, jnp.exp(score - safe), 0.0)
                alpha = jnp.where(
                    m_ == -jnp.inf, jnp.where(m_new == -jnp.inf, 1.0, 0.0), jnp.exp(m_ - safe)
                )
                return (
                    m_new,
                    l_ * alpha + jnp.sum(p, axis=-1, keepdims=True),
                    acc * alpha + jnp.dot(p, value, preferred_element_type=F32),
                )
            out = []
            for r in range(T):
                m_, l_sum, acc = carry[r]
                valid = (tok < length) & ((tok_code & (1 << r)) != 0)
                score = jax.lax.dot_general(
                    qf[r], key, (((1,), (1,)), ((), ())), preferred_element_type=F32
                )
                score = jnp.where(valid, score * sm_scale, -jnp.inf)
                m_new = jnp.maximum(m_, jnp.max(score, axis=-1, keepdims=True))
                safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)
                p = jnp.where(valid, jnp.exp(score - safe), 0.0)
                alpha = jnp.where(
                    m_ == -jnp.inf, jnp.where(m_new == -jnp.inf, 1.0, 0.0), jnp.exp(m_ - safe)
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

    def waited(t: Any) -> Any:
        s = slot(t)
        with sc("u_wait", 3):
            if fast:  # q38 L2-UAMERGE: X-UA-ISSUE stage (b)'s wait, sized to the bytes tile t's live copies move
                from . import union_issue

                union_issue.fast_wait(ring=ring, sems=sems, t=t, s=s, n_union=n_union)
            else:
                _pipe_wait(pool, ring, sems, s, interpret)
        return s

    if coissue:
        t_a = jnp.minimum(jnp.maximum(end - D, 0), ntiles)  # tiles whose start position t + D is < end

        def tile_a(t: Any, carry: Any) -> Any:
            s = waited(t)
            if post:
                return tile_body(t, s, carry, mid=lambda: start_pos(t + D))
            start_pos(t + D)
            return tile_body(t, s, carry)

        def tile_b(t: Any, carry: Any) -> Any:
            return tile_body(t, waited(t), carry)

        state = jax.lax.fori_loop(0, t_a, tile_a, init)
        state = jax.lax.fori_loop(t_a, ntiles, tile_b, state)
    else:  # (ii) alone: `attend_request`'s guarded start, over the stream

        def tile(t: Any, carry: Any) -> Any:
            s = waited(t)

            @pl.when(t + D < end)
            def _():
                start_pos(t + D)

            return tile_body(t, s, carry)

        state = jax.lax.fori_loop(0, ntiles, tile, init)
    if stack:
        state = tuple(
            tuple(x[r * nh : (r + 1) * nh] for x in state) for r in range(T)
        )
    rk, rv, ok_rows, ok_any = recent
    rkey = rk * scale_k
    rval = jnp.where(ok_any, rv * scale_v, 0.0)
    partial, out = [], []
    for r in range(T):
        m_, l_sum, acc = state[r]
        ok = ok_rows[r]
        sc_ = jax.lax.dot_general(
            qf[r], rkey, (((1,), (1,)), ((), ())), preferred_element_type=F32
        )
        sc_ = jnp.where(ok, sc_ * sm_scale, -jnp.inf)
        m_new = jnp.maximum(m_, jnp.max(sc_, axis=-1, keepdims=True))
        safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)
        p = jnp.where(ok, jnp.exp(sc_ - safe), 0.0)
        alpha = jnp.where(m_ == -jnp.inf, 0.0, jnp.exp(m_ - safe))
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
    return (partial if dcp > 1 else out), ntiles


def pipe_ids_copies(src: Any, zero_tail: Any, rows: Any, sems: Any, k: Any) -> list[Any]:
    """(ii): one request's union ids `src [1, KU]` and the 128-lane zero tail (`attend_request`'s clamp reads past the
    union land on zeros) -> SMEM row `k` of `rows [PIPE_IDS_BUFS, KU + 128]`, both on row k's own semaphore `sems[k]`
    (D08 S1 B3's two copies per row; one semaphore per row, never shared between rows)."""
    KU = src.shape[-1]
    return [
        pltpu.make_async_copy(src, rows.at[pl.ds(k, 1), pl.ds(0, KU)], sems.at[k]),
        pltpu.make_async_copy(zero_tail, rows.at[pl.ds(k, 1), pl.ds(KU, 128)], sems.at[k]),
    ]


def pipe_request_loop(
    *,
    nreq: int,
    nslots: int,
    pipe: int,
    ids_copies: Any,
    body: Any,
    init: Any,
    sc: Any = None,
) -> Any:
    """A layer's request loop under (ii) (`fori_loop` over `nreq` requests; `pipe` without PIPE_XREQ: a plain loop,
    the caller keeps its own ids handling).

    `ids_copies(q, k)`: the async copies that land request q's union ids row (and its zero tail) in SMEM row k of the
    PIPE_IDS_BUFS-row buffer, all on row k's own DMA semaphore. `body(q, ctl, carry) -> (carry, ntiles)` runs request
    q with `ctl` = dict(ids_row, base, inflight, next_row, has_next) (None without (ii)).

    Rows: request q in row q % 3. Before the loop requests 0 and 1 are started and request 0 is waited. At request q:
    wait request q + 1 (started a whole request earlier), then start request q + 2 into row (q + 2) % 3 = (q - 1) % 3,
    which request q - 1's tile loop no longer reads. Ring position: base(q + 1) = (base(q) + ntiles(q)) % nslots."""
    sc = sc or (lambda *_: contextlib.nullcontext())
    if not pipe & PIPE_XREQ:
        return jax.lax.fori_loop(0, nreq, lambda q, c: body(q, None, c)[0], init)
    NB = PIPE_IDS_BUFS
    with sc("qsa_u_ids", 2):
        for c in ids_copies(0, 0):
            c.start()
        if nreq > 1:
            for c in ids_copies(1, 1):
                c.start()
        for c in ids_copies(0, 0):
            c.wait()

    def req(q: Any, carry: Any) -> Any:
        inner, base = carry
        k, k1, k2 = jax.lax.rem(q, NB), jax.lax.rem(q + 1, NB), jax.lax.rem(q + 2, NB)
        with sc("qsa_u_ids", 2):

            @pl.when(q + 1 < nreq)
            def _():
                for c in ids_copies(q + 1, k1):
                    c.wait()

            @pl.when(q + 2 < nreq)
            def _():
                for c in ids_copies(q + 2, k2):
                    c.start()

        ctl = dict(ids_row=k, base=base, inflight=q > 0, next_row=k1, has_next=q + 1 < nreq)
        inner, nt = body(q, ctl, inner)
        return inner, jax.lax.rem(base + nt, nslots)

    return jax.lax.fori_loop(0, nreq, req, (init, jnp.int32(0)))[0]


# q38 HIX-KERN K6 begin (VLLM_TPU_MK_KERN_STOCK_KV, default off; appended, so no line above moves except the marked edits).
# The union tile body dequantizes FP8 K / V before the dots (key = f32(k8) * scale_k, value = f32(v8) * scale_v, NaN / past-length rows zeroed
# in f32) and Mosaic pushes that f32 right operand as bf16 in one pass (U1B j-9b68df9c). With the key the FP8 tiles go to the dots as FP8
# (stock RPA's f32 x f8E4M3FN form, batched_rpa flash_attention.py:65-70 / :165-166) and the scales apply after: score = (dot(q, k8) *
# sm_scale) * scale_k, acc += dot(p, v8) * scale_v. No operand is quantized: K / V are already FP8 in the pool and q / p stay f32. The value
# mask moves to the integer domain before the bitcast (byte 0 = +0.0 for a NaN code when dcp > 1, or a row at / past the request's length),
# which is the f32 path's +0.0. At the served scales (1.0 / 1.0, enforced by KV-SCALE-GUARD) the f32 path's operands are exactly the FP8
# values, so outputs are bitwise IF the f8-push dot equals the bf16-push dot on the MXU: U on CPU (interpret mode upcasts), decided by the
# deviceless LLO leg (vmatpush1.f8e4m3 on the right operand, left operand not cast to f8) and the DEV car's device bitwise row.
def _kern_stock_kv() -> bool:
    from vllm_torchtpu.megakernel_runtime.qwen38_kern import stock_kv_on

    return stock_kv_on()


def _kern_value_f8(v8: Any, keep_rows: Any, nan_mask: bool) -> Any:
    """v8 [TILE, 256] FP8 with rows past the length (and NaN codes when `nan_mask`) set to byte 0 (+0.0), in the integer domain."""
    vi = jax.lax.bitcast_convert_type(v8, jnp.uint8).astype(jnp.int32)  # the select runs on 32-bit lanes (zero-extended bytes), then truncates back
    keep = jnp.broadcast_to(keep_rows, vi.shape)
    if nan_mask:  # e4m3fn NaN = S.1111.111 (0x7F / 0xFF)
        keep = jnp.logical_and(keep, jnp.bitwise_and(vi, 0x7F) != 0x7F)
    return jax.lax.bitcast_convert_type(jnp.where(keep, vi, 0).astype(jnp.uint8), v8.dtype)
# q38 HIX-KERN K6 end
