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

The routing arrives either as a table built from the caller's local top-k ids
(`routing_table`) or as the router's softmax scores, with the top-k selection
run in the kernel.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Rows per call at or below which the MoE entry selects this kernel: the
# measured crossover vs dense-expert MoE (j-a6af2846, j-d8af40dc). At 128 rows
# it wins 1.14x (scores) / 1.09x (ids) on recorded skew; at 256 it is parity
# on recorded skew (1.01x / 0.93x) and loses on uniform routing (0.95x / 0.90x).
MOE_LOWTOK_MAX_TOKENS = 128
_MAX_LOCAL_EXPERTS = 128

_ROW_TILE = 32  # FP8 sublane packing
_NEG = -3.0e38  # router scores at or below this never win a pass
_LANE = 128
_VMEM_FRACTION = 0.75
_VMEM_MARGIN = 1.25  # over declared scratch and operands, for compiler tiles
_FP8 = jnp.float8_e4m3fn
_FP8_MAX = 448.0
_HOT_SLOTS = 128  # copied experts per layer, one lane tile


class Holders(NamedTuple):
    """One shard's copy placement for the scores routing, in router-row ids."""

    gid: jax.Array  # [E] the expert in each local slot: primaries, then copies
    slot: jax.Array  # [lanes] this shard's slot of each expert, -1 if none
    copy: jax.Array  # [lanes] the shard holding each expert's copy, -1 if none
    hot: jax.Array  # [_HOT_SLOTS] the copied experts, hottest first, -1 padded
    hprim: jax.Array  # [_HOT_SLOTS] each hot expert's primary shard
    hcopy: jax.Array  # [_HOT_SLOTS] each hot expert's copy shard
    hot1h: jax.Array  # [lanes, _HOT_SLOTS] 1 at (hot[i], i)


def holder_tables(gid, copy, hot, owned):
    """Host-built Holders from a shard's slot list, each expert's copy shard
    (-1 without one) and the copied experts in hot order."""
    n_global = len(copy)
    lanes = pl.cdiv(n_global, _LANE) * _LANE
    if len(hot) > _HOT_SLOTS:
        raise ValueError(f"{len(hot)} copied experts exceed {_HOT_SLOTS}")
    slot = np.full(lanes, -1, np.int32)
    slot[np.asarray(gid)] = np.arange(len(gid), dtype=np.int32)
    copy_lanes = np.full(lanes, -1, np.int32)
    copy_lanes[:n_global] = copy
    hot_slots = np.full(_HOT_SLOTS, -1, np.int32)
    hot_slots[: len(hot)] = hot
    real = hot_slots >= 0
    if not np.all(copy_lanes[hot_slots[real]] >= 0):
        raise ValueError("every hot expert must have a copy")
    return Holders(
        gid=np.asarray(gid, np.int32),
        slot=slot,
        copy=copy_lanes,
        hot=hot_slots,
        hprim=np.where(real, hot_slots // owned, -1).astype(np.int32),
        hcopy=np.where(real, copy_lanes[np.maximum(hot_slots, 0)], -1).astype(np.int32),
        hot1h=(np.arange(lanes)[:, None] == hot_slots[None, :]).astype(np.float32),
    )


def can_use_moe_lowtok(hidden, w1, w2, s1, s2, b1, b2, *, activation, rhs_quant_dtype):
    """Check arithmetic/layout support; the caller applies the token cap."""
    if activation != "silu" or rhs_quant_dtype is not None:
        return False
    if b1 is not None or b2 is not None or s1 is None or s2 is None:
        return False
    if hidden.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
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


def _vmem_budget(rows, hidden, inter, w13_parts, lanes, extra=0):
    """Declared scratch and VMEM operands plus a compiler margin, capped.

    A shape-dependent reservation leaves room for neighbouring ops to
    prefetch their weights.
    """
    unit = w13_parts + 1
    weights = unit * (hidden * 2 * (inter // w13_parts) + inter * hidden)
    scales = unit * 4 * (2 * (inter // w13_parts) + hidden)
    accumulators = rows * 4 * (2 * inter + hidden)
    # x in and out, scores, tables, one-hot.
    operands = (
        rows * 2 * hidden * 2 + rows * 4 * lanes + 4 * lanes * 8 + 2 * lanes * lanes
    )
    need = int((weights + scales + accumulators + operands + extra) * _VMEM_MARGIN)
    capacity = int(pltpu.get_tpu_info().vmem_capacity_bytes * _VMEM_FRACTION)
    return min(need, capacity)


def _quantize_rows(v):
    """Per-row FP8 quantization of an f32 block: values and restoring scale."""
    amax = jnp.max(jnp.abs(v), axis=1, keepdims=True)
    scale = jnp.maximum(amax, 1e-12) / _FP8_MAX
    return (v / scale).astype(_FP8), scale


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
    choice=None,
):
    """Stream the hit experts of this shard and accumulate their outputs.

    gid_s [1, E] (SMEM) is the table lane of each local expert; hits_s
    [1, lanes] (SMEM) is 1 on lanes some row routes to; wtab_ref [rows, lanes]
    holds the routing weights. Each expert is `w13_parts` gate|up units then
    one down unit; the ring has one slot per unit, so slot == part. `choice`,
    if given, takes the hit count after the first units are started, may
    append local slots to `ids`, and returns the new count.
    """
    half = inter // w13_parts
    units = w13_parts + 1
    slots = units
    lane = jax.lax.broadcasted_iota(jnp.int32, wtab_ref.shape, 1)

    def collect(e, n):
        h = hits_s[0, gid_s[0, e]]

        @pl.when(h == 1)
        def _():
            ids[n] = e

        return n + h

    n_hit = jax.lax.fori_loop(0, n_experts, collect, jnp.int32(0))
    n_units = n_hit * units

    def unit_copies(e_idx, part):
        expert = ids[e_idx]
        slot = part
        # The down unit builds w13 descriptors it never starts; clamp them.
        col = pl.multiple_of(jnp.minimum(part, w13_parts - 1) * half, half)
        w13_copies = (
            pltpu.make_async_copy(
                w13_hbm.at[expert, :, pl.ds(col, half)],
                w13_buf.at[slot, :, pl.ds(0, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                w13_hbm.at[expert, :, pl.ds(inter + col, half)],
                w13_buf.at[slot, :, pl.ds(half, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                s13_hbm.at[expert, :, pl.ds(col, half)],
                s13_buf.at[slot, :, pl.ds(0, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                s13_hbm.at[expert, :, pl.ds(inter + col, half)],
                s13_buf.at[slot, :, pl.ds(half, half)],
                w13_sem.at[slot],
            ),
        )
        w2_copies = (
            pltpu.make_async_copy(w2_hbm.at[expert], w2_buf.at[slot], w2_sem.at[slot]),
            pltpu.make_async_copy(s2_hbm.at[expert], s2_buf.at[slot], w2_sem.at[slot]),
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

    acc[...] = jnp.zeros_like(acc)
    x_q, x_s = _quantize_rows(x_ref[...].astype(jnp.float32))

    for ahead in range(slots - 1):

        @pl.when(ahead < n_units)
        def _prime(ahead=ahead):
            start(jnp.int32(0), jnp.int32(ahead))

    if choice is not None:
        n_before = n_hit
        n_hit = choice(n_hit)
        n_units = n_hit * units
        # Nothing was started above when no fixed expert hit this shard.
        for ahead in range(slots - 1):

            @pl.when((n_before == 0) & (ahead < n_units))
            def _prime_late(ahead=ahead):
                start(jnp.int32(0), jnp.int32(ahead))

    def body(u, carry):
        e_idx, part = carry
        slot = part
        expert = ids[e_idx]

        @pl.when(u + slots - 1 < n_units)
        def _prefetch():
            start(*advance(e_idx, part, slots - 1))

        wait(e_idx, part)

        # One static branch per half: the lane offsets must be static.
        for p in range(w13_parts):

            @pl.when(part == p)
            def _gate_up_half(p=p):
                h = jax.lax.dot_general(
                    x_q,
                    w13_buf[slot],
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                scales = s13_buf[slot]
                col = p * half
                h_acc[:, col : col + half] = h[:, :half] * x_s * scales[:, :half]
                h_acc[:, inter + col : inter + col + half] = (
                    h[:, half:] * x_s * scales[:, half:]
                )

        @pl.when(part == w13_parts)
        def _down():
            gate = h_acc[:, pl.ds(0, inter)]
            up = h_acc[:, pl.ds(inter, inter)]
            a = gate * jax.nn.sigmoid(gate) * up
            a_q, a_s = _quantize_rows(a)
            o = jax.lax.dot_general(
                a_q,
                w2_buf[slot],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            o = o * a_s * s2_buf[slot]
            # Zero weight masks the rows this expert does not serve.
            w = jnp.sum(
                jnp.where(lane == gid_s[0, expert], wtab_ref[...], 0.0),
                axis=1,
                keepdims=True,
            )
            acc[...] += w * o

        return advance(e_idx, part, 1)

    jax.lax.fori_loop(0, n_units, body, (jnp.int32(0), jnp.int32(0)))
    out_ref[...] = acc[...].astype(out_ref.dtype)


def _select_topk(scores, topk):
    """The served router as a routing table over global expert ids.

    `topk` max-and-mask passes over the softmax scores (ties to the lowest
    id), weights renormalized and rounded through bf16.
    """
    lane = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    x = jnp.where(scores > _NEG, scores, _NEG)
    masked = jnp.finfo(jnp.float32).min
    sel = jnp.zeros_like(x)
    wsum = jnp.zeros((x.shape[0], 1), jnp.float32)
    for _ in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, lane, x.shape[1] - 1), axis=1, keepdims=True)
        pick = lane == idx
        sel = jnp.where(pick, m, sel)
        wsum = wsum + m
        x = jnp.where(pick, masked, x)
    wt = jnp.where(wsum > 0, sel / wsum, 0.0)
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
):
    if topk is None:
        wtab_ref = route_ref
    else:
        wtab_ref, *scratch = scratch
        wtab_ref[...] = _select_topk(route_ref[...], topk)
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
    gid_s,
    rank_s,
    hot_s,
    hprim_s,
    hcopy_s,
    slot_s,
    x_ref,
    scores_ref,
    copy_ref,
    hot1h_ref,
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
    *scratch,
    n_experts,
    inter,
    w13_parts,
    topk,
    ranks,
    owned,
):
    """The scores routing with copies: each hit expert has one server.

    An expert without a copy is served by its primary shard. The hit copied
    experts go, hottest first, to whichever holder has fewer hit experts so
    far (ties to the primary); every shard computes the same assignment. This
    shard's fixed experts start streaming while the copies are assigned.
    """
    wtab_ref[...] = _select_topk(scores_ref[...], topk)
    lanes = wtab_ref.shape[1]
    lane = jax.lax.broadcasted_iota(jnp.int32, (1, lanes), 1)
    my_rank = rank_s[0] // owned
    hit = jnp.any(wtab_ref[...] != 0, axis=0, keepdims=True)
    prim = lane // owned
    fixed = hit & (copy_ref[...] < 0)
    loads = jnp.zeros_like(lane)
    for r in range(ranks):
        load = jnp.sum(jnp.where(fixed & (prim == r), 1, 0), axis=1, keepdims=True)
        loads = jnp.where(lane == r, load, loads)
    hits_v[...] = jnp.where(fixed & (prim == my_rank), 1, 0).astype(jnp.int32)
    fixed_copy = pltpu.make_async_copy(hits_v, hits_s, hsem.at[0])
    fixed_copy.start()
    # The hit copied experts in hot order: hit[hot[i]] gathered on the MXU, an
    # inclusive lane prefix sum for each one's position, a one-hot scatter.
    nc = hot1h_ref.shape[1]
    hit8 = jnp.broadcast_to(jnp.where(hit, 1.0, 0.0).astype(jnp.bfloat16), (8, lanes))
    hh = jnp.dot(hit8, hot1h_ref[...], preferred_element_type=jnp.float32)[:1]
    lane_c = jax.lax.broadcasted_iota(jnp.int32, (1, nc), 1)
    ps = hh
    k = 1
    while k < nc:
        ps = ps + jnp.where(lane_c >= k, pltpu.roll(ps, k, axis=1), 0.0)
        k *= 2
    pos = jnp.where(hh > 0.5, ps - 1.0, -1.0)
    pos_rows = jnp.transpose(jnp.broadcast_to(pos, (nc, nc)))
    row = jax.lax.broadcasted_iota(jnp.int32, (nc, nc), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (nc, nc), 1).astype(jnp.float32)
    compact = jnp.sum(jnp.where(pos_rows == col, row, 0), axis=0, keepdims=True)
    # One buffer to SMEM: the compact list, its length, the fixed loads.
    cmp_v[...] = jnp.zeros_like(cmp_v)
    cmp_v[0:1, :] = compact
    cmp_v[1:2, :] = jnp.broadcast_to(ps[:, nc - 1 :].astype(jnp.int32), (1, nc))
    cmp_v[2:3, :] = loads[:, :nc]
    table_copy = pltpu.make_async_copy(cmp_v, cmp_s, hsem.at[1])
    table_copy.start()
    fixed_copy.wait()

    def choice(n_fixed):
        table_copy.wait()

        def assign(j, n):
            i = cmp_s[0, j]
            home, other = hprim_s[0, i], hcopy_s[0, i]
            pick = jnp.where(cmp_s[2, other] < cmp_s[2, home], other, home)
            cmp_s[2, pick] = cmp_s[2, pick] + 1
            mine = pick == my_rank

            @pl.when(mine)
            def _():
                ids[n] = slot_s[0, hot_s[0, i]]

            return n + jnp.where(mine, 1, 0)

        return jax.lax.fori_loop(0, cmp_s[1, 0], assign, n_fixed)

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
        choice=choice,
    )


def _layout(x, w13):
    tokens, _ = x.shape
    n_experts, _, two_inter = w13.shape
    inter = two_inter // 2
    if inter % _LANE:
        raise ValueError(
            f"the intermediate width {inter} must be a multiple of {_LANE} lanes"
        )
    w13_parts = 2 if (inter // 2) % _LANE == 0 else 1
    rows = pl.cdiv(tokens, _ROW_TILE) * _ROW_TILE
    return n_experts, inter, w13_parts, rows


def _stream_scratch(rows, hidden, inter, w13_parts):
    slots = w13_parts + 1
    half = inter // w13_parts
    return (
        pltpu.VMEM((slots, hidden, 2 * half), _FP8),
        pltpu.VMEM((slots, inter, hidden), _FP8),
        pltpu.VMEM((slots, 1, 2 * half), jnp.float32),
        pltpu.VMEM((slots, 1, hidden), jnp.float32),
        pltpu.SemaphoreType.DMA((slots,)),
        pltpu.SemaphoreType.DMA((slots,)),
        pltpu.VMEM((rows, 2 * inter), jnp.float32),
        pltpu.VMEM((rows, hidden), jnp.float32),
    )


def _moe_lowtok_copies(
    x, scores, w13, s13, w2, s2, experts_start, holders, topk, ranks, owned
):
    rows, hidden = x.shape
    n_experts, inter, w13_parts, _ = _layout(x, w13)
    lanes = scores.shape[1]
    nc = holders.hot.shape[0]
    tables = holders.hot1h.size * 2 + 4 * lanes + 2 * 8 * nc * 4
    smem = [
        holders.gid.reshape(1, n_experts),
        jnp.reshape(experts_start, (1,)).astype(jnp.int32),
        *(t.reshape(1, -1) for t in (holders.hot, holders.hprim, holders.hcopy)),
        holders.slot.reshape(1, lanes),
    ]
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(
            _kernel_copies,
            n_experts=n_experts,
            inter=inter,
            w13_parts=w13_parts,
            topk=topk,
            ranks=ranks,
            owned=owned,
        ),
        out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(smem),
            in_specs=[vmem, vmem, vmem, vmem, hbm, hbm, hbm, hbm],
            out_specs=vmem,
            scratch_shapes=(
                pltpu.SMEM((n_experts,), jnp.int32),
                pltpu.VMEM((rows, lanes), jnp.float32),
                pltpu.VMEM((1, lanes), jnp.int32),
                pltpu.SMEM((1, lanes), jnp.int32),
                pltpu.VMEM((8, nc), jnp.int32),
                pltpu.SMEM((8, nc), jnp.int32),
                pltpu.SemaphoreType.DMA((2,)),
                *_stream_scratch(rows, hidden, inter, w13_parts),
            ),
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=_vmem_budget(rows, hidden, inter, w13_parts, lanes, tables)
        ),
        name="moe_lowtok",
    )(
        *smem,
        x,
        scores,
        holders.copy.reshape(1, lanes),
        holders.hot1h.astype(jnp.bfloat16),
        s13,
        s2,
        w13,
        w2,
    )


def routing_table(topk_ids_local, topk_weights, n_experts):
    """[tokens, n_experts] routing weights from local top-k ids (-1: remote)."""
    valid = topk_ids_local >= 0
    onehot = jax.nn.one_hot(
        jnp.where(valid, topk_ids_local, 0), n_experts, dtype=jnp.float32
    )
    weights = jnp.where(valid, topk_weights, 0).astype(jnp.float32)
    return jnp.einsum("tke,tk->te", onehot, weights)


@functools.partial(jax.jit, static_argnames=("topk", "ranks"))
def moe_lowtok(
    x, route, w13, s13, w2, s2, experts_start=0, *, topk=None, holders=None, ranks=None
):
    """This shard's MoE output for a few tokens.

    With `topk` None, `route` is the routing table over local experts (see
    routing_table). With `topk`, `route` is the router's softmax scores over
    all experts, the top-k runs in the kernel, and this shard holds
    experts [experts_start, experts_start + E); a zero score row (a padded
    token) selects nothing. With `holders` (see holder_tables) and `ranks`
    shards, this shard's primaries are experts [experts_start, experts_start
    + owned) and its later slots copies, and every hit expert is served by
    exactly one holder. Scales use [E, 1, 1, N]. Callers must first check
    can_use_moe_lowtok.
    """
    tokens, hidden = x.shape
    n_experts, inter, w13_parts, rows = _layout(x, w13)
    # The table's lanes are local expert ids, or global ones for the scores.
    n_route = route.shape[1]
    lanes = pl.cdiv(n_route, _LANE) * _LANE
    x_rows = jnp.pad(x, ((0, rows - tokens), (0, 0)))
    route = jnp.pad(
        route.astype(jnp.float32), ((0, rows - tokens), (0, lanes - route.shape[1]))
    )
    s13 = s13.reshape(n_experts, 1, 2 * inter).astype(jnp.float32)
    s2 = s2.reshape(n_experts, 1, hidden).astype(jnp.float32)
    if holders is not None:
        assert topk is not None and ranks <= _HOT_SLOTS <= lanes
        out = _moe_lowtok_copies(
            x_rows,
            route,
            w13,
            s13,
            w2,
            s2,
            experts_start,
            holders,
            topk,
            ranks,
            n_route // ranks,
        )
        return out[:tokens]
    gid = (experts_start + jnp.arange(n_experts, dtype=jnp.int32)).reshape(1, n_experts)
    wtab = () if topk is None else (pltpu.VMEM((rows, lanes), jnp.float32),)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            n_experts=n_experts,
            inter=inter,
            w13_parts=w13_parts,
            topk=topk,
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
