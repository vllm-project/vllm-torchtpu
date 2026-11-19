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
"""Experimental small-batch FP8 MoE: one Pallas call, no sort or permute.

Each token's routing is a dense weight table over experts. The kernel collects
the shard's hit experts, streams their weights through a ring of VMEM slots in
units (the gate|up halves and the down projection of one expert), and applies
each expert to every row with that row's routing weight as the mask. The
output is the shard's partial sum; the combine across shards stays the
caller's. Activations are quantized per row to FP8 before each matmul (W8A8).

The routing arrives as a table built from the caller's local top-k ids
(`routing_table`), as the router's softmax scores, or as its raw logits
(`from_logits=True`); with scores or logits the kernel runs the top-k
selection, and with a copies placement also the copy-aware holder selection
over the placement's holder tables. The other inputs are the rows, the FP8
expert weights and their scales, and the shard's first expert.
"""

import functools
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# The most rows per call the kernel is built for; moe_lowtok and the routed-ids
# cap (fused_moe_gmm.set_lowtok_max_tokens) refuse more.
MOE_LOWTOK_MAX_TOKENS = 448
HOT_SLOTS = 128
_MAX_LOCAL_EXPERTS = 128
_TABLE_LANES = 512
# The copy-aware entry's prefix scan over the hot table covers a power of two
# at or above the layer's copy count; any such width counts every active copy.
# It covers at least this many lanes, the width the 4-copies-per-rank layout
# over 8 ranks (32 copies) needs.
_MIN_HOT_SCAN = 32

_ROW_TILE = 8  # sublane row alignment (avoids padding T=8,16 to 32)
_NEG = -3.0e38  # router scores at or below this never win a pass
LANE = 128
_VMEM_FRACTION = 0.75
_VMEM_MARGIN = 1.50  # over declared scratch and operands, for compiler tiles
FP8_DTYPE = jnp.float8_e4m3fn
_FP8_MAX = 448.0


def dot_f32(a, b):
    """``a @ b`` for 2D operands, accumulated in float32."""
    return jax.lax.dot_general(
        a, b, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )


def can_use_moe_lowtok(hidden, w1, w2, s1, s2, b1, b2, *, activation, rhs_quant_dtype):
    """Check arithmetic and layout support, rows included (at most
    MOE_LOWTOK_MAX_TOKENS). The chip is not checked: the kernel computes with
    FP8 activations (W8A8), so it is armed only where the chip has native FP8."""
    if activation != "silu" or rhs_quant_dtype is not None:
        return False
    if b1 is not None or b2 is not None or s1 is None or s2 is None:
        return False
    if hidden.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return False
    if hidden.shape[0] > MOE_LOWTOK_MAX_TOKENS:
        return False
    h = hidden.shape[1]
    e, i, out = w2.shape
    if not 0 < e <= _MAX_LOCAL_EXPERTS or out != h or w1.shape != (e, h, 2 * i):
        return False
    if (2 * i) % 512:
        return False
    if (
        hidden.dtype != jnp.bfloat16
        or w1.dtype != jnp.float8_e4m3fn
        or w2.dtype != jnp.float8_e4m3fn
    ):
        return False
    return s1.shape == (e, 1, 1, 2 * i) and s2.shape == (e, 1, 1, h)


def _vmem_budget(rows, hidden, inter, w13_parts, lanes):
    """Declared scratch and VMEM operands plus a compiler margin, capped.

    A shape-dependent reservation leaves room for neighbouring ops to
    prefetch their weights.
    """
    weights = w13_parts * (hidden * 2 * (inter // w13_parts)) + inter * hidden
    scales = w13_parts * 4 * (2 * (inter // w13_parts)) + 4 * hidden
    accumulators = rows * 4 * (inter + hidden)
    # x in and out, scores, tables, one-hot.
    operands = (
        rows * 2 * hidden * 2 + rows * 4 * lanes + 4 * lanes * 8 + 2 * lanes * lanes
    )
    need = int((weights + scales + accumulators + operands) * _VMEM_MARGIN)
    capacity = int(pltpu.get_tpu_info().vmem_capacity_bytes * _VMEM_FRACTION)
    return min(need, capacity)


def quantize_rows(v):
    """Per-row FP8 quantization of an f32 block: values and restoring scale."""
    amax = jnp.max(jnp.abs(v), axis=1, keepdims=True)
    scale = jnp.maximum(amax, 1e-12) / _FP8_MAX
    return (v / scale).astype(FP8_DTYPE), scale


def _stream(
    gid_s,
    hits_s,
    wtab_ref,
    x_ref,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    out_ref,
    ids,
    w13_buf,
    w2_buf,
    s13_buf,
    s2_buf,
    w13_sem,
    w2_sem,
    h_acc,
    acc,
    *,
    n_experts,
    inter,
    w13_parts,
    owned=None,
    choice=None,
    gid_of=None,
):
    """Stream the hit experts of this shard and accumulate their outputs.

    gid_s [1, E] (SMEM) is the table lane of each local expert (or `gid_of`
    maps local slot `e` to its global lane); hits_s [1, lanes] (SMEM) is 1 on
    lanes some row routes to; wtab_ref [rows, lanes] holds the routing weights.
    Each expert is `w13_parts` gate|up units then one down unit; the ring has
    one slot per w13 unit plus a dedicated w2 slot.
    """
    if gid_of is None:
        gid_of = lambda e: gid_s[0, e]  # noqa: E731
    rows = wtab_ref.shape[0]
    units = w13_parts + 1
    lane_128 = jax.lax.broadcasted_iota(jnp.int32, (rows, LANE), 1)

    def collect(e, n):
        h = hits_s[0, gid_of(e)]

        @pl.when(h == 1)
        def _():
            ids[n] = e

        return n + h

    n_collect = owned if (choice is not None and owned is not None) else n_experts
    n_hit = jax.lax.fori_loop(0, n_collect, collect, jnp.int32(0))
    n_units = n_hit * units

    def unit_copies(e_idx, part):
        expert = ids[e_idx]
        slot = jnp.minimum(part, w13_parts - 1)
        col = pl.multiple_of(slot * inter, inter)
        w13_copies = (
            pltpu.make_async_copy(
                w13_hbm.at[expert, :, pl.ds(col, inter)],
                w13_buf.at[slot],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                s13_hbm.at[expert, :, pl.ds(col, inter)],
                s13_buf.at[slot],
                w13_sem.at[slot],
            ),
        )
        w2_copies = (
            pltpu.make_async_copy(w2_hbm.at[expert], w2_buf.at[0], w2_sem.at[0]),
            pltpu.make_async_copy(s2_hbm.at[expert], s2_buf.at[0], w2_sem.at[0]),
        )
        return part, w13_copies, w2_copies

    def advance(e_idx, part, by):
        """The unit `by` (at most `units`) ahead, without a division."""
        p = part + by
        wrap = p >= units
        return jnp.where(wrap, e_idx + 1, e_idx), jnp.where(wrap, p - units, p)

    def start(e_idx, part):
        part, w13_copies, w2_copies = unit_copies(e_idx, part)

        @pl.when(part < w13_parts)
        def _():
            for c in w13_copies:
                c.start()

        @pl.when(part == w13_parts)
        def _():
            for c in w2_copies:
                c.start()

    def wait(e_idx, part):
        part, w13_copies, w2_copies = unit_copies(e_idx, part)

        @pl.when(part < w13_parts)
        def _():
            for c in w13_copies:
                c.wait()

        @pl.when(part == w13_parts)
        def _():
            for c in w2_copies:
                c.wait()

    for ahead in range(units - 1):

        @pl.when(ahead < n_units)
        def _prime(ahead=ahead):
            start(jnp.int32(0), jnp.int32(ahead))

    if choice is not None:
        n_before = n_hit
        n_hit = choice(n_hit)
        n_units = n_hit * units
        for ahead in range(units - 1):

            @pl.when((n_before == 0) & (ahead < n_units))
            def _prime_late(ahead=ahead):
                start(jnp.int32(0), jnp.int32(ahead))

    acc[...] = jnp.zeros_like(acc)
    x_q, x_s = quantize_rows(x_ref[...].astype(jnp.float32))

    def body(u, carry):
        e_idx, part = carry
        expert = ids[e_idx]

        @pl.when(u + units - 1 < n_units)
        def _prefetch():
            start(*advance(e_idx, part, units - 1))

        wait(e_idx, part)

        @pl.when(part == 0)
        def _gate_half():
            h = dot_f32(x_q, w13_buf[0])
            gate = h * x_s * s13_buf[0]
            h_acc[...] = gate * jax.nn.sigmoid(gate)

        @pl.when(part == 1)
        def _up_half():
            h = dot_f32(x_q, w13_buf[1])
            up = h * x_s * s13_buf[1]
            h_acc[...] = h_acc[...] * up

        @pl.when(part == 2)
        def _down():
            a = h_acc[...]
            a_q, a_s = quantize_rows(a)
            g = gid_of(expert)
            blk = pl.multiple_of((g >> 7) << 7, LANE)
            w_slice = wtab_ref[:, pl.ds(blk, LANE)]
            w = jnp.sum(
                jnp.where(lane_128 + blk == g, w_slice, 0.0),
                axis=1,
                keepdims=True,
            )
            row_scale = a_s * w
            o = dot_f32(a_q, w2_buf[0])
            acc[...] += (o * s2_buf[0]) * row_scale

        return advance(e_idx, part, 1)

    jax.lax.fori_loop(0, n_units, body, (jnp.int32(0), jnp.int32(0)))
    out_ref[...] = acc[...].astype(out_ref.dtype)


def select_topk(scores, topk, *, from_logits: bool = False):
    """The served router as a routing table over global expert ids.

    `topk` max-and-mask passes over the softmax scores or raw router logits,
    weights renormalized and rounded through bf16. Ties go to the lowest id
    at every row count: each pass takes the row max, then the lowest id equal
    to it. A single argmax would leave the tie to Mosaic's reduction order,
    which on TPU does not keep the lowest id.
    """
    lane = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    scores_f32 = scores.astype(jnp.float32) if from_logits else scores
    x0 = jnp.where(scores_f32 > _NEG, scores_f32, _NEG)
    x = x0
    masked = jnp.finfo(jnp.float32).min
    wsum = jnp.zeros((x.shape[0], 1), jnp.float32)
    m0 = None
    for r in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        # The other ids fill with the last one, so the min never picks an id
        # below the max.
        idx = jnp.min(jnp.where(x == m, lane, x.shape[1] - 1), axis=1, keepdims=True)
        if from_logits:
            if r == 0:
                m0 = m
            wsum = wsum + jnp.where(m > _NEG, jnp.exp(m - cast("jax.Array", m0)), 0.0)
        else:
            wsum = wsum + m
        x = jnp.where(lane == idx, masked, x)
    num = jnp.exp(x0 - cast("jax.Array", m0)) if from_logits else x0
    wt = jnp.where((x == masked) & (wsum > 0), num / wsum, 0.0)
    return wt.astype(jnp.bfloat16).astype(jnp.float32)


def _kernel(
    gid_s,
    x_ref,
    route_ref,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    out_ref,
    ids,
    hits_v,
    hits_s,
    hsem,
    *scratch,
    n_experts,
    inter,
    w13_parts,
    topk,
    from_logits=False,
):
    if topk is None:
        wtab_ref = route_ref
    else:
        wtab_ref, *scratch = scratch
        wtab_ref[...] = select_topk(route_ref[...], topk, from_logits=from_logits)
    # Lanes some row routes to; `collect` keeps the ones in this shard's gid.
    hit = jnp.any(wtab_ref[...] != 0, axis=0, keepdims=True)
    hits_v[...] = jnp.where(hit, 1, 0).astype(jnp.int32)
    copy = pltpu.make_async_copy(hits_v, hits_s, hsem.at[0])
    copy.start()
    copy.wait()
    _stream(
        gid_s,
        hits_s,
        wtab_ref,
        x_ref,
        s13_hbm,
        s2_hbm,
        w13_hbm,
        w2_hbm,
        out_ref,
        ids,
        *scratch,
        n_experts=n_experts,
        inter=inter,
        w13_parts=w13_parts,
    )


def _kernel_copies(
    tab_s,
    prim_ref,
    copy_ref,
    hot1h_ref,
    x_hbm,
    scores_ref,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    out_ref,
    ids,
    wtab_ref,
    hits_v,
    hits_s,
    cmp_v,
    cmp_s,
    hsem,
    x_vmem,
    xsem,
    *scratch,
    n_experts,
    inter,
    w13_parts,
    topk,
    ranks,
    owned,
    offsets,
    from_logits=False,
):
    """Scores entry with hot-expert copy slots and greedy holder assignment."""
    rows, lanes = scores_ref.shape
    x_copy = pltpu.make_async_copy(x_hbm, x_vmem, xsem.at[0])
    x_copy.start()
    o_gid, o_hot, o_hprim, o_hcopy, o_slot = offsets
    my_rank = tab_s[0] // owned

    wt = select_topk(scores_ref[...], topk, from_logits=from_logits)
    wtab_ref[...] = wt
    hit = jnp.any(wt != 0, axis=0, keepdims=True)
    prim = prim_ref[...]
    copy = copy_ref[...]
    has_copy = copy >= 0
    fixed = hit & jnp.logical_not(has_copy)

    hits_v[...] = jnp.where(fixed & (prim == my_rank), 1, 0).astype(jnp.int32)
    cp1 = pltpu.make_async_copy(hits_v, hits_s, hsem.at[1])
    cp1.start()

    r_iota = jax.lax.broadcasted_iota(jnp.int32, (ranks, lanes), 0)
    prim_1h_t = jnp.transpose((prim == r_iota).astype(jnp.bfloat16))
    fixed8 = jnp.broadcast_to(
        jnp.where(fixed, 1.0, 0.0).astype(jnp.bfloat16), (8, lanes)
    )
    loads_mxu = jnp.dot(fixed8, prim_1h_t, preferred_element_type=jnp.float32)[
        :1
    ].astype(jnp.int32)

    nc = hot1h_ref.shape[1]
    max_copies = (n_experts - owned) * ranks if n_experts > owned else nc
    n_hot_bound = min(nc, max(_MIN_HOT_SCAN, 1 << (max_copies - 1).bit_length()))
    hit8 = jnp.broadcast_to(jnp.where(hit, 1.0, 0.0).astype(jnp.bfloat16), (8, lanes))
    hh = jnp.dot(hit8, hot1h_ref[...], preferred_element_type=jnp.float32)[:1]
    lane_c = jax.lax.broadcasted_iota(jnp.int32, (1, nc), 1)
    ps = hh
    k = 1
    while k < n_hot_bound:
        ps = ps + jnp.where(lane_c >= k, pltpu.roll(ps, k, axis=1), 0.0)
        k *= 2
    n_active = ps[:, n_hot_bound - 1 : n_hot_bound].astype(jnp.int32)
    pos = jnp.where((hh > 0.5) & (lane_c < n_hot_bound), ps - 1.0, -1.0)
    posm = jnp.transpose(jnp.broadcast_to(pos, (nc, nc)))[:n_hot_bound, :]
    row_i = jax.lax.broadcasted_iota(jnp.int32, (n_hot_bound, nc), 0)
    lane_p = jax.lax.broadcasted_iota(jnp.int32, (n_hot_bound, nc), 1).astype(
        jnp.float32
    )
    comp = jnp.sum(jnp.where(posm == lane_p, row_i, 0), axis=0, keepdims=True)

    cmp_v[...] = jnp.zeros_like(cmp_v)
    cmp_v[0:1, :] = comp
    cmp_v[1:2, :] = jnp.broadcast_to(n_active, (1, nc))
    cmp_v[2:3, :] = jnp.pad(loads_mxu, ((0, 0), (0, nc - ranks)))
    cp2 = pltpu.make_async_copy(cmp_v, cmp_s, hsem.at[2])
    cp2.start()
    cp1.wait()

    def choice(n_fixed):
        cp2.wait()

        def assign(j, n):
            i = cmp_s[0, j]
            e = tab_s[o_hot + i]
            o = tab_s[o_hprim + i]
            c = tab_s[o_hcopy + i]
            pick = jnp.where(cmp_s[2, c] < cmp_s[2, o], c, o)
            cmp_s[2, pick] = cmp_s[2, pick] + 1
            mine = pick == my_rank

            @pl.when(mine)
            def _():
                ids[n] = tab_s[o_slot + e]

            return n + jnp.where(mine, 1, 0)

        return jax.lax.fori_loop(0, cmp_s[1, 0], assign, n_fixed)

    x_copy.wait()
    _stream(
        tab_s,
        hits_s,
        wtab_ref,
        x_vmem,
        s13_hbm,
        s2_hbm,
        w13_hbm,
        w2_hbm,
        out_ref,
        ids,
        *scratch,
        n_experts=n_experts,
        inter=inter,
        w13_parts=w13_parts,
        owned=owned,
        choice=choice,
        gid_of=lambda e: tab_s[o_gid + e],
    )


def weight_layout(x, w13):
    """(experts, intermediate width, w13 weight units, padded rows) of a call."""
    tokens, _ = x.shape
    n_experts, _, two_inter = w13.shape
    inter = two_inter // 2
    if inter % (2 * LANE):
        # can_use_moe_lowtok admits gate|up widths that are multiples of 512;
        # the kernel streams the gate columns and the up columns as two units.
        raise ValueError(
            f"the intermediate width {inter} must be a multiple of {2 * LANE}"
        )
    w13_parts = 2
    rows = pl.cdiv(tokens, _ROW_TILE) * _ROW_TILE
    return n_experts, inter, w13_parts, rows


def _stream_scratch(rows, hidden, inter, w13_parts):
    half = inter // w13_parts
    return (
        pltpu.VMEM((w13_parts, hidden, 2 * half), FP8_DTYPE),
        pltpu.VMEM((1, inter, hidden), FP8_DTYPE),
        pltpu.VMEM((w13_parts, 1, 2 * half), jnp.float32),
        pltpu.VMEM((1, 1, hidden), jnp.float32),
        pltpu.SemaphoreType.DMA((w13_parts,)),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.VMEM((rows, inter), jnp.float32),
        pltpu.VMEM((rows, hidden), jnp.float32),
    )


def table_offsets(n_experts, nc=HOT_SLOTS):
    """Flat packed table offsets `(o_gid, o_hot, o_hprim, o_hcopy, o_slot, length)`."""
    o_gid = 8
    o_hot = o_gid + n_experts
    o_hprim = o_hot + nc
    o_hcopy = o_hprim + nc
    o_slot = o_hcopy + nc
    length = pl.cdiv(o_slot + _TABLE_LANES, LANE) * LANE
    return o_gid, o_hot, o_hprim, o_hcopy, o_slot, length


def pack_tables(experts_start, gid, hot, hprim, hcopy, slot):
    """Pack the scalar tables for a copy-enabled layer into one flat int32 array.

    `experts_start` is the rank's first primary id, rank * owned; the kernel
    recovers the rank from it (`tab_s[0] // owned`).
    """
    gid = np.asarray(gid, np.int32).reshape(-1)
    hot, hprim, hcopy = (
        np.asarray(t, np.int32).reshape(-1) for t in (hot, hprim, hcopy)
    )
    slot = np.asarray(slot, np.int32).reshape(-1)
    if slot.size != _TABLE_LANES:
        raise ValueError(
            f"the slot table is {_TABLE_LANES} lanes wide, got {slot.size}"
        )
    if not (hprim.size == hot.size == hcopy.size):
        raise ValueError("hot, hprim and hcopy must be the same length")
    o_gid, o_hot, o_hprim, o_hcopy, o_slot, length = table_offsets(gid.size, hot.size)
    tab = np.full((length,), -1, np.int32)
    tab[0] = int(experts_start)
    tab[o_gid : o_gid + gid.size] = gid
    tab[o_hot : o_hot + hot.size] = hot
    tab[o_hprim : o_hprim + hot.size] = hprim
    tab[o_hcopy : o_hcopy + hot.size] = hcopy
    tab[o_slot : o_slot + _TABLE_LANES] = slot
    return tab


def slot_table(gid, n_experts):
    """This rank's local slot for each of the 512 global experts (`-1` where
    absent); `gid` lists the global expert in each local slot."""
    ids = np.asarray(gid, np.int64).reshape(-1)
    if ids.size and (ids.min() < 0 or ids.max() >= _TABLE_LANES):
        raise ValueError(
            f"slot table: expert ids must be in 0..{_TABLE_LANES - 1} (the copy "
            f"tables are {_TABLE_LANES} lanes wide), got {ids.min()}..{ids.max()}"
        )
    if len(set(ids.tolist())) != ids.size:
        raise ValueError("slot table: an expert is listed in two local slots")
    gid = jnp.asarray(gid, jnp.int32).reshape(-1)
    return (
        jnp.full((_TABLE_LANES,), -1, jnp.int32)
        .at[gid]
        .set(jnp.arange(n_experts, dtype=jnp.int32))
    )


def hot_onehot(hot):
    """One-hot gather matrix `[512, len(hot)]` bf16 for a hot expert table
    (`-1` entries are padding and get an empty column)."""
    ids = np.asarray(hot, np.int64).reshape(-1)
    if ((ids < -1) | (ids >= _TABLE_LANES)).any():
        raise ValueError(
            f"hot table: expert ids must be -1 (padding) or in "
            f"0..{_TABLE_LANES - 1}, got {sorted(set(ids.tolist()))}"
        )
    hot = jnp.asarray(hot, jnp.int32).reshape(-1)
    lanes = jax.lax.broadcasted_iota(jnp.int32, (_TABLE_LANES, hot.shape[0]), 0)
    return jnp.where(lanes == hot.reshape(1, -1), 1.0, 0.0).astype(jnp.bfloat16)


def hot_tables(prim, copy, order=None, slots=HOT_SLOTS):
    """Return `(hot, hprim, hcopy)` padded with `-1` to `slots` entries."""
    prim = jnp.asarray(prim, jnp.int32).reshape(-1)
    copy = jnp.asarray(copy, jnp.int32).reshape(-1)
    if order is not None:
        o = [int(e) for e in order]
        if len(o) > slots:
            raise ValueError(
                f"{len(o)} copied experts exceed the {slots}-entry hot table"
            )
        if len(set(o)) != len(o) or any(e < 0 or e >= prim.shape[0] for e in o):
            raise ValueError("hot order must contain unique, valid expert ids")
        uncopied = [e for e in o if int(copy[e]) < 0]
        if uncopied:
            # The kernel would serve such an expert on its primary and assign
            # it a holder as well, adding its contribution twice.
            raise ValueError(f"hot order lists experts with no copy: {uncopied}")
        hot = np.full(slots, -1, np.int32)
        hot[: len(o)] = o
        hot = jnp.asarray(hot)
    else:
        hot = jnp.nonzero(copy >= 0, size=slots, fill_value=-1)[0].astype(jnp.int32)
    safe = jnp.where(hot >= 0, hot, 0)
    hprim = jnp.where(hot >= 0, prim[safe], -1).astype(jnp.int32)
    hcopy = jnp.where((hot >= 0) & (copy[safe] >= 0), copy[safe], -1).astype(jnp.int32)
    return hot, hprim, hcopy


def routing_table(topk_ids_local, topk_weights, n_experts):
    """[tokens, n_experts] routing weights from local top-k ids (-1: remote)."""
    valid = topk_ids_local >= 0
    onehot = jax.nn.one_hot(
        jnp.where(valid, topk_ids_local, 0), n_experts, dtype=jnp.float32
    )
    weights = jnp.where(valid, topk_weights, 0).astype(jnp.float32)
    return jnp.einsum("tke,tk->te", onehot, weights)


@functools.partial(
    jax.jit,
    static_argnames=("topk", "ranks", "owned", "has_copies", "from_logits"),
)
def moe_lowtok(
    x,
    route,
    w13,
    s13,
    w2,
    s2,
    experts_start=0,
    tables=None,
    prim=None,
    copy=None,
    hot1h=None,
    *,
    topk=None,
    ranks=8,
    owned=64,
    has_copies=False,
    from_logits=False,
):
    """This shard's MoE output for a few tokens.

    With `topk` None, `route` is the routing table over local experts (see
    routing_table). With `topk` and `has_copies=False`, `route` is the router's
    softmax scores (or raw router logits when `from_logits=True`) over all
    experts, the top-k runs in the kernel, and this shard holds experts
    [experts_start, experts_start + E). With `has_copies=True`, `tables`,
    `prim`, `copy`, and `hot1h` provide the placement's holder tables and the
    kernel assigns each hit copied expert to the less-loaded holder; the rank
    then comes from the packed tables (`pack_tables`' `experts_start`) and the
    `experts_start` argument is not read. At most MOE_LOWTOK_MAX_TOKENS rows.
    """
    tokens, hidden = x.shape
    if tokens > MOE_LOWTOK_MAX_TOKENS:
        raise ValueError(
            f"moe_lowtok is built for at most {MOE_LOWTOK_MAX_TOKENS} rows, "
            f"got {tokens}"
        )
    if from_logits and topk is None:
        raise ValueError(
            "moe_lowtok: from_logits=True needs topk (the kernel runs the top-k "
            "on the router logits); pass a routing table with from_logits=False"
        )
    n_experts, inter, w13_parts, rows = weight_layout(x, w13)
    route_cols = route.shape[1]
    lanes = pl.cdiv(route_cols, LANE) * LANE
    if topk is None and lanes < n_experts:
        # Every weight slot looks up its routing-table lane.
        raise ValueError(
            f"the routing table's {route_cols} columns ({lanes} lanes) do not "
            f"cover the {n_experts} weight slots"
        )
    x_rows = jnp.pad(x, ((0, rows - tokens), (0, 0)))
    if from_logits:
        route_dtype = (
            route.dtype if route.dtype in (jnp.bfloat16, jnp.float32) else jnp.float32
        )
        pad_val = jnp.asarray(
            -jnp.inf if route_dtype == jnp.bfloat16 else _NEG, dtype=route_dtype
        )
        route = jnp.pad(
            route.astype(route_dtype),
            ((0, rows - tokens), (0, lanes - route.shape[1])),
            constant_values=pad_val,
        )
    else:
        route = jnp.pad(
            route.astype(jnp.float32), ((0, rows - tokens), (0, lanes - route.shape[1]))
        )
    s13 = s13.reshape(n_experts, 1, 2 * inter).astype(jnp.float32)
    s2 = s2.reshape(n_experts, 1, hidden).astype(jnp.float32)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    if has_copies:
        if topk is None:
            raise ValueError("has_copies=True requires topk to be set")
        if tables is None or prim is None or copy is None or hot1h is None:
            raise ValueError(
                "has_copies=True requires tables, prim, copy, and hot1h tensors"
            )
        nc = int(hot1h.shape[-1])
        if route_cols != _TABLE_LANES:
            raise ValueError(
                f"has_copies=True serves {_TABLE_LANES} global experts (the "
                f"copy tables' width), got {route_cols}"
            )
        if not 0 < ranks <= nc or not 0 < owned <= n_experts:
            raise ValueError(
                f"has_copies=True needs 0 < ranks <= {nc} (the hot table's "
                f"width) and 0 < owned <= {n_experts} (the weight slots), got "
                f"ranks={ranks}, owned={owned}"
            )
        if ranks * owned != _TABLE_LANES:
            raise ValueError(
                f"has_copies=True needs ranks x owned = {_TABLE_LANES} primaries, "
                f"got {ranks} x {owned}"
            )
        o_gid, o_hot, o_hprim, o_hcopy, o_slot, length = table_offsets(n_experts, nc)
        tab = jnp.asarray(tables, jnp.int32)
        if tab.shape != (length,):
            raise ValueError(
                f"packed tables for {n_experts} slots and {nc} hot entries are "
                f"[{length}], got {tab.shape}"
            )
        prim_v = jnp.asarray(prim, jnp.int32).reshape(1, lanes)
        copy_v = jnp.asarray(copy, jnp.int32).reshape(1, lanes)
        hot1h_v = jnp.asarray(hot1h, jnp.bfloat16).reshape(lanes, nc)
        out = pl.pallas_call(
            functools.partial(
                _kernel_copies,
                n_experts=n_experts,
                inter=inter,
                w13_parts=w13_parts,
                topk=topk,
                ranks=ranks,
                owned=owned,
                offsets=(o_gid, o_hot, o_hprim, o_hcopy, o_slot),
                from_logits=from_logits,
            ),
            out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[vmem, vmem, vmem, hbm, vmem, hbm, hbm, hbm, hbm],
                out_specs=vmem,
                scratch_shapes=(
                    pltpu.SMEM((n_experts,), jnp.int32),
                    pltpu.VMEM((rows, lanes), jnp.float32),
                    pltpu.VMEM((1, lanes), jnp.int32),
                    pltpu.SMEM((1, lanes), jnp.int32),
                    pltpu.VMEM((8, nc), jnp.int32),
                    pltpu.SMEM((8, nc), jnp.int32),
                    pltpu.SemaphoreType.DMA((3,)),
                    pltpu.VMEM((rows, hidden), x_rows.dtype),
                    pltpu.SemaphoreType.DMA((1,)),
                    *_stream_scratch(rows, hidden, inter, w13_parts),
                ),
                grid=(),
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=_vmem_budget(rows, hidden, inter, w13_parts, lanes)
            ),
            name="moe_lowtok_copies",
        )(tab, prim_v, copy_v, hot1h_v, x_rows, route, s13, s2, w13, w2)
        return out[:tokens]

    gid = (experts_start + jnp.arange(n_experts, dtype=jnp.int32)).reshape(1, n_experts)
    wtab = () if topk is None else (pltpu.VMEM((rows, lanes), jnp.float32),)
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            n_experts=n_experts,
            inter=inter,
            w13_parts=w13_parts,
            topk=topk,
            from_logits=from_logits,
        ),
        out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, vmem, hbm, hbm, hbm, hbm],
            out_specs=vmem,
            scratch_shapes=(
                pltpu.SMEM((n_experts,), jnp.int32),
                pltpu.VMEM((1, lanes), jnp.int32),
                pltpu.SMEM((1, lanes), jnp.int32),
                pltpu.SemaphoreType.DMA((1,)),
                *wtab,
                *_stream_scratch(rows, hidden, inter, w13_parts),
            ),
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=_vmem_budget(rows, hidden, inter, w13_parts, lanes)
        ),
        name="moe_lowtok",
    )(gid, x_rows, route, s13, s2, w13, w2)
    return out[:tokens]
