# Copyright 2025 Google LLC
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
"""A decode-step MoE for a few tokens: one Pallas call per layer.

At a handful of tokens the expert-parallel MoE is bound by reading the hit
experts' weights (about 12.6 MB each in fp8) and by the chain of small
operations around the grouped matmuls: two sorts, a permute, a pad to the
matmul tile, an unpermute. This kernel drops the chain. Every token's routing
is a dense table of weights over the shard's experts, the kernel walks the
hit experts only, streams their weights through a ring of VMEM slots in
equal-sized units (the gate|up halves and the down projection of one expert
are three units of 4 MB at the served shape), and applies each expert to
every token with that token's weight as the mask (zero for tokens the expert
does not serve). Routing requires no sort or permutation. The output is
the shard's partial sum of expert outputs per token, and the combine across
shards stays the caller's all-reduce.

Weights are fp8 with one scale per output channel; activations are
quantised per token to fp8 inside the kernel before each matmul, the whole
row at once, as the grouped matmul does (W8A8). Rows are padded to the fp8
tile (32) by the wrapper; the kernel computes the padded rows and the wrapper
drops them. The hit list is built inside the kernel from a 0/1 mask, so the
wrapper adds no sort.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

ROW_TILE = 32  # fp8 sublane packing: four rows per 32-bit word, eight words
LANES = 512  # the routing table's expert axis: every expert of the model, by global id
NEG = -3.0e38  # the router's sentinel: scores at or below it never win a pass
LANE = 128  # the lane tile; column offsets inside a unit must sit on it
VMEM_FRACTION = 0.75  # of the core's vector memory, read from the device at trace time


def vmem_limit(fraction=VMEM_FRACTION):
    """The kernel's scratch ceiling: `fraction` of the core's vector memory (the weight units,
    the row-sized accumulators and the row buffers all live there)."""
    return int(pltpu.get_tpu_info().vmem_capacity_bytes * fraction)


VMEM_MARGIN = 1.25  # over the scratch and operands the kernel declares, for the compiler's own tiles


def vmem_budget(rows,
                hidden,
                inter,
                w13_parts,
                two_inter,
                n_experts,
                extra_bytes=0):
    """Reserve the scratch and VMEM operands plus a compiler margin.

    A shape-dependent reservation leaves room for surrounding operations to
    prefetch their weights. The device-derived ceiling bounds the reservation.
    """
    unit = w13_parts + 1
    weights = unit * (hidden * 2 * (inter // w13_parts) + inter * hidden
                      )  # fp8 units of the ring
    scales = unit * 4 * (2 * (inter // w13_parts) + hidden)
    rows_f32 = rows * 4 * (two_inter + hidden)  # the accumulators
    operands = rows * 2 * hidden * 2 + rows * 4 * LANES + 4 * LANES * 8 + 2 * LANES * LANES  # x in and out, scores, tables, one-hot
    need = int(
        (weights + scales + rows_f32 + operands + extra_bytes) * VMEM_MARGIN)
    return min(need, vmem_limit())


# per expert: the gate|up columns in w13_parts units (two halves when the half is lane-aligned, else one), then down;
# the ring has one slot per unit, so two units stream ahead of the compute
FP8 = jnp.float8_e4m3fn
FP8_MAX = 448.0


def _quantize_rows(v):
    """Per-row fp8 quantisation of a [rows, cols] f32 block: the value in
    fp8 and the per-row scale that restores it."""
    amax = jnp.max(jnp.abs(v), axis=1, keepdims=True)
    scale = jnp.maximum(amax, 1e-12) / FP8_MAX
    return (v / scale).astype(FP8), scale


def _stream(gid_s,
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
            gid_of=None):
    """The streaming body shared by the entries. gid_s int32[1, n_experts]
    (SMEM): the table lane of each local slot (the global expert id, or the
    local id itself for the ids entry), or `gid_of(e)` when given (the scores
    entry reads its packed table); hits_s int32[1, LANES] (SMEM): 1 in
    the lanes served by this rank; wtab_ref [rows, LANES]
    f32: the routing weight of global expert g in lane g; x_ref [rows,
    hidden] bf16; s13_hbm [E, 1, 2*inter] and s2_hbm [E, 1, hidden] f32
    per-channel scales in HBM (each expert's rows ride with its weight DMAs);
    w13_hbm [E, hidden, 2*inter] and w2_hbm [E, inter, hidden] fp8 in HBM;
    out_ref [rows, hidden].

    Scratch: ids int32[E] (SMEM) the hit list; w13_buf [slots, hidden,
    2*half] a gate|up unit per slot (its gate columns, then its up columns);
    w2_buf [slots, inter, hidden]; s13_buf [slots, 1, 2*half] and s2_buf
    [slots, 1, hidden] the unit's scale rows; the DMA semaphores; h_acc
    [rows, 2*inter] f32 the current expert's gate|up rows; acc the output
    accumulator.

    choice: an optional hook called with the count of hits collected so far
    (the fixed experts, whose first weight units are already in flight);
    it may append local slots to `ids` from that index on and must return
    the new count. The scores entry's holder choice runs there, under the
    first expert's fetch."""
    rows = x_ref.shape[0]
    if gid_of is None:
        gid_of = lambda e: gid_s[0, e]  # noqa: E731
    half = inter // w13_parts
    units = w13_parts + 1
    slots = units
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, LANES), 1)

    # the hit list of this shard's experts, in expert order: a scalar loop
    def collect(e, n):
        h = hits_s[0, gid_of(e)]

        @pl.when(h == 1)
        def _():
            ids[n] = e

        return n + h

    n_hit = jax.lax.fori_loop(0, n_experts, collect, jnp.int32(0))
    n_units = n_hit * units

    def unit_copies(e_idx, part):
        """The DMA descriptors of a unit into its slot (slot = part, since the
        ring has as many slots as an expert has units): parts 0 and 1 are the
        gate|up halves (gate columns [p*half, (p+1)*half) and the up columns
        [inter + p*half, ...) of w13), part 2 is the down projection."""
        expert = ids[e_idx]
        slot = part
        # part 2 (the down unit) builds descriptors it never starts; clamp
        # its column into range
        col = pl.multiple_of(jnp.minimum(part, w13_parts - 1) * half, half)
        w13_copies = (pltpu.make_async_copy(
            w13_hbm.at[expert, :, pl.ds(col, half)],
            w13_buf.at[slot, :, pl.ds(0, half)], w13_sem.at[slot]),
                      pltpu.make_async_copy(
                          w13_hbm.at[expert, :,
                                     pl.ds(inter + col, half)],
                          w13_buf.at[slot, :,
                                     pl.ds(half, half)], w13_sem.at[slot]),
                      pltpu.make_async_copy(
                          s13_hbm.at[expert, :, pl.ds(col, half)],
                          s13_buf.at[slot, :,
                                     pl.ds(0, half)], w13_sem.at[slot]),
                      pltpu.make_async_copy(
                          s13_hbm.at[expert, :,
                                     pl.ds(inter + col, half)],
                          s13_buf.at[slot, :,
                                     pl.ds(half, half)], w13_sem.at[slot]))
        w2_copies = (pltpu.make_async_copy(w2_hbm.at[expert], w2_buf.at[slot],
                                           w2_sem.at[slot]),
                     pltpu.make_async_copy(s2_hbm.at[expert], s2_buf.at[slot],
                                           w2_sem.at[slot]))
        return part, w13_copies, w2_copies

    def advance(e_idx, part, by):
        """The (expert index, part) of the unit `by` units ahead, without a
        division: `by` is at most UNITS."""
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
        def _prime():
            start(jnp.int32(0), jnp.int32(ahead))

    if choice is not None:
        n_before = n_hit
        n_hit = choice(n_hit)
        n_units = n_hit * units
        # no fixed hit on this rank: nothing was primed above, prime now from the appended hits
        for ahead in range(slots - 1):

            @pl.when((n_before == 0) & (ahead < n_units))
            def _prime_late():
                start(jnp.int32(0), jnp.int32(ahead))

    def body(u, carry):
        e_idx, part = carry
        slot = part
        expert = ids[e_idx]

        @pl.when(u + slots - 1 < n_units)
        def _prefetch():
            start(*advance(e_idx, part, slots - 1))

        wait(e_idx, part)

        # this half's gate|up columns for every row, scaled per token and per
        # channel, into h_acc's columns; one static branch per half, since
        # the lane offsets must be static
        for p in range(w13_parts):

            @pl.when(part == p)
            def _gate_up_half(p=p):
                h = jax.lax.dot_general(x_q,
                                        w13_buf[slot],
                                        (((1, ), (0, )), ((), ())),
                                        preferred_element_type=jnp.float32)
                scales = s13_buf[
                    slot]  # [1, 2*half]: this half's gate columns, then its up columns
                col = p * half
                h_acc[:, col:col + half] = h[:, :half] * x_s * scales[:, :half]
                h_acc[:, inter + col:inter + col +
                      half] = h[:, half:] * x_s * scales[:, half:]

        @pl.when(part == w13_parts)
        def _down():
            gate = h_acc[:, pl.ds(0, inter)]
            up = h_acc[:, pl.ds(inter, inter)]
            a = gate * jax.nn.sigmoid(gate) * up
            a_q, a_s = _quantize_rows(a)
            o = jax.lax.dot_general(a_q,
                                    w2_buf[slot], (((1, ), (0, )), ((), ())),
                                    preferred_element_type=jnp.float32)
            o = o * a_s * s2_buf[slot]
            # the routing weight is the mask: zero for rows the expert does not serve
            w = jnp.sum(jnp.where(lane == gid_of(expert), wtab_ref[...], 0.0),
                        axis=1,
                        keepdims=True)  # [rows, 1]
            acc[...] += w * o

        return advance(e_idx, part, 1)

    jax.lax.fori_loop(0, n_units, body, (jnp.int32(0), jnp.int32(0)))
    out_ref[...] = acc[...].astype(out_ref.dtype)


def _kernel_ids(gid_ref, hit_ref, x_ref, wtab_ref, s13_hbm, s2_hbm, w13_hbm,
                w2_hbm, out_ref, ids, w13_buf, w2_buf, s13_buf, s2_buf,
                w13_sem, w2_sem, h_acc, acc, *, n_experts, inter, w13_parts):
    """The ids entry: the routing table (over local ids) and the hit vector
    come from the caller; gid_ref is the identity over local ids."""
    _stream(gid_ref,
            hit_ref,
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
            n_experts=n_experts,
            inter=inter,
            w13_parts=w13_parts)


def _kernel_scores(tab_s,
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
                   topk,
                   ranks,
                   owned=1,
                   has_copies=True,
                   offsets):
    """The scores entry: the router's selection runs here first, then the
    assignment of every hit expert to one holder. prim_ref / copy_ref
    [1, LANES] int32 (VMEM): the rank holding each global expert's primary /
    copy (-1: no copy); tab_s int32[length] (SMEM), the scalar tables packed
    as one flat operand (see pack_tables / table_offsets; `offsets` = (gid,
    hot, hprim, hcopy, slot) as static ints): [0] this rank (or its first
    owned expert id, divided by `owned` here), then the global id in each
    local slot, the copied experts hottest first (-1 padded) with the rank
    of each one's primary and copy, and this rank's local slot of every
    global expert (-1 where it holds none); x_hbm [rows, hidden] (HBM): the rows, copied into x_vmem under the
    selection below; hot1h_ref [LANES, NC] bf16 (VMEM): one-hot, 1 at
    (hot[i], i). Single-holder
    experts are fixed and their first weight units are fetched first; the copied
    experts that are hit on this call are compacted into a short list on the
    vector units (the one-hot gather on the MXU, a lane prefix sum, a one-hot
    scatter) and then assigned one after another in hot order, each to the
    home with fewer experts assigned so far (fixed plus the copies already
    placed), ties to the primary; identical on every rank."""
    rows = x_vmem.shape[0]
    # Copy rows into VMEM while selection and holder assignment run; wait
    # for the copy immediately before streaming the expert weights.
    x_copy = pltpu.make_async_copy(x_hbm, x_vmem, xsem.at[0])
    x_copy.start()
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, LANES), 1)
    # this rank: the scalar as given, or a rank's first owned expert id divided by the owned count (the served
    # layer's experts_start value, packed at load: integer arithmetic on the torch side reaches the kernel as
    # int64 and the compile refuses it)
    o_gid, o_hot, o_hprim, o_hcopy, o_slot = offsets
    my_rank = tab_s[0] // owned

    # the router's selection, in the kernel: topk passes of max-and-mask over
    # the softmax scores (ties to the lowest id, as the served router), the
    # weights renormalised and rounded through bf16 as the served path does,
    # written as the dense routing table [rows, LANES] over global ids
    x = scores_ref[...]
    x = jnp.where(x > NEG, x, NEG)
    masked = jnp.finfo(jnp.float32).min
    sel = jnp.zeros_like(x)
    wsum = jnp.zeros((rows, 1), jnp.float32)
    for _ in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, lane, LANES - 1),
                      axis=1,
                      keepdims=True)
        pick = lane == idx
        sel = jnp.where(pick, m, sel)
        wsum = wsum + m
        x = jnp.where(pick, masked, x)
    wt = jnp.where(wsum > 0, sel / wsum,
                   0.0).astype(jnp.bfloat16).astype(jnp.float32)
    wtab_ref[...] = wt
    # which global experts serve a row this call, as a lane vector [1, LANES]
    hit = jnp.any(wt != 0, axis=0, keepdims=True)
    prim = prim_ref[...]
    copy = copy_ref[...]
    has_copy = copy >= 0
    # the assignment, every lane at once: single-holder experts are fixed and
    # set each rank's base load; a copied expert goes to the holder with the
    # smaller base load, ties to the primary. Identical on every rank.
    fixed = hit & jnp.logical_not(has_copy)
    lane1 = lane[:1]
    if not has_copies:
        # With no copies, every hit expert has one holder. Send this rank's
        # hits to SMEM and start streaming without the holder-assignment loop.
        hits_v[...] = jnp.where(fixed & (prim == my_rank), 1,
                                0).astype(jnp.int32)
        cp1 = pltpu.make_async_copy(hits_v, hits_s, hsem.at[1])
        cp1.start()
        cp1.wait()
        x_copy.wait()
        _stream(tab_s,
                hits_s,
                wtab_ref,
                x_vmem,
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
                n_experts=n_experts,
                inter=inter,
                w13_parts=w13_parts,
                gid_of=lambda e: tab_s[o_gid + e])
        return
    # the fixed load of every rank, in lane r of a lane vector, to SMEM for the scalar loop
    loads = jnp.zeros_like(lane1)
    for r in range(ranks):
        load_r = jnp.sum(jnp.where(fixed & (prim == r), 1, 0),
                         axis=1,
                         keepdims=True)
        loads = jnp.where(lane1 == r, load_r, loads)
    # this rank's fixed experts, per global lane, to SMEM first: the stream primes its first fetch from them
    # while the choice below is computed; the copied experts it serves are appended by the choice
    hits_v[...] = jnp.where(fixed & (prim == my_rank), 1, 0).astype(jnp.int32)
    cp1 = pltpu.make_async_copy(hits_v, hits_s, hsem.at[1])
    cp1.start()
    # the hit copied experts, compacted: hh[i] = hit[hot[i]] through the one-hot on the MXU; an inclusive lane
    # prefix sum gives each hit entry its position; a one-hot scatter puts entry i at column pos[i]
    nc = hot1h_ref.shape[1]
    hit8 = jnp.broadcast_to(
        jnp.where(hit, 1.0, 0.0).astype(jnp.bfloat16), (8, LANES))
    hh = jnp.dot(hit8, hot1h_ref[...],
                 preferred_element_type=jnp.float32)[:1]  # [1, nc]
    lane_c = jax.lax.broadcasted_iota(jnp.int32, (1, nc), 1)
    ps = hh
    k = 1
    while k < nc:
        ps = ps + jnp.where(lane_c >= k, pltpu.roll(ps, k, axis=1), 0.0)
        k *= 2
    n_active = ps[:, nc - 1:nc].astype(jnp.int32)  # [1, 1]
    pos = jnp.where(hh > 0.5, ps - 1.0, -1.0)  # [1, nc]
    posm = jnp.transpose(jnp.broadcast_to(
        pos, (nc, nc)))  # row i holds pos[i] in every lane
    row_i = jax.lax.broadcasted_iota(jnp.int32, (nc, nc), 0)
    lane_p = jax.lax.broadcasted_iota(jnp.int32, (nc, nc),
                                      1).astype(jnp.float32)
    comp = jnp.sum(jnp.where(posm == lane_p, row_i, 0), axis=0,
                   keepdims=True)  # [1, nc]: the hit entries in order
    # one packed buffer to SMEM: row 0 the compact list, row 1 n_active, row 2 the fixed loads per rank
    cmp_v[...] = jnp.zeros_like(cmp_v)
    cmp_v[0:1, :] = comp
    cmp_v[1:2, :] = jnp.broadcast_to(n_active, (1, nc))
    cmp_v[2:3, :] = loads[:, :nc].astype(jnp.int32)
    cp2 = pltpu.make_async_copy(cmp_v, cmp_s, hsem.at[2])
    cp2.start()
    cp1.wait()

    def choice(n_fixed):
        """the hit copied experts, hottest first: each to the home with fewer experts so far, ties to the
        primary; the ones this rank serves are appended to the hit list (their local slot) from n_fixed on"""
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
                hits_s[0, e] = 1
                ids[n] = tab_s[o_slot + e]

            return n + jnp.where(mine, 1, 0)

        return jax.lax.fori_loop(0, cmp_s[1, 0], assign, n_fixed)

    x_copy.wait()
    _stream(tab_s,
            hits_s,
            wtab_ref,
            x_vmem,
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
            n_experts=n_experts,
            inter=inter,
            w13_parts=w13_parts,
            choice=choice,
            gid_of=lambda e: tab_s[o_gid + e])


@functools.partial(jax.jit, static_argnames=("vmem_limit_bytes", ))
def moe_lowtok(x,
               topk_ids_local,
               topk_weights,
               w13,
               s13,
               w2,
               s2,
               *,
               vmem_limit_bytes=None):
    """The shard's MoE output for a few tokens.

    x [tokens, hidden] bf16; topk_ids_local [tokens, k] int32 with -1 for
    experts the shard does not hold; topk_weights [tokens, k]; w13 [E, hidden,
    2*inter] fp8 (gate | up) with s13 [E, 2*inter] scales (any leading
    singleton axes); w2 [E, inter, hidden] fp8 with s2 [E, hidden] scales.
    Returns [tokens, hidden] in x's dtype: the sum over the shard's experts
    of weight * expert(x), zero for tokens routed elsewhere; the all-reduce
    over shards is the caller's.
    """
    tokens, hidden = x.shape
    n_experts, _, two_inter = w13.shape
    inter = two_inter // 2
    if n_experts > LANES:
        raise ValueError(
            f"the routing table holds up to {LANES} experts per shard, not {n_experts}"
        )
    if inter % LANE:
        raise ValueError(
            f"the intermediate width {inter} must be a multiple of the lane tile"
        )
    w13_parts = 2 if (inter // 2) % LANE == 0 else 1
    rows = -(-tokens // ROW_TILE) * ROW_TILE
    # the dense routing-weight table over the shard's experts, [rows, E], then padded to the lane tile
    valid = topk_ids_local >= 0
    onehot = jax.nn.one_hot(jnp.where(valid, topk_ids_local, 0),
                            n_experts,
                            dtype=jnp.float32)
    wtab = jnp.einsum("tke,tk->te", onehot,
                      jnp.where(valid, topk_weights, 0).astype(jnp.float32))
    hit = jnp.pad(
        jnp.any(wtab != 0, axis=0).astype(jnp.int32),
        (0, LANES - n_experts)).reshape(1, LANES)
    gid = jnp.arange(n_experts, dtype=jnp.int32).reshape(1, n_experts)
    wtab = jnp.pad(wtab, ((0, rows - tokens), (0, LANES - n_experts)))
    x_rows = jnp.pad(x, ((0, rows - tokens), (0, 0)))
    s13 = s13.reshape(n_experts, 1, two_inter).astype(jnp.float32)
    s2 = s2.reshape(n_experts, 1, hidden).astype(jnp.float32)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out = pl.pallas_call(
        functools.partial(_kernel_ids,
                          n_experts=n_experts,
                          inter=inter,
                          w13_parts=w13_parts),
        out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[vmem, vmem, hbm, hbm, hbm, hbm],
            out_specs=vmem,
            scratch_shapes=(
                pltpu.SMEM((n_experts, ), jnp.int32),
                pltpu.VMEM((w13_parts + 1, hidden, 2 * (inter // w13_parts)),
                           FP8),
                pltpu.VMEM((w13_parts + 1, inter, hidden), FP8),
                pltpu.VMEM((w13_parts + 1, 1, 2 * (inter // w13_parts)),
                           jnp.float32),
                pltpu.VMEM((w13_parts + 1, 1, hidden), jnp.float32),
                pltpu.SemaphoreType.DMA((w13_parts + 1, )),
                pltpu.SemaphoreType.DMA((w13_parts + 1, )),
                pltpu.VMEM((rows, two_inter), jnp.float32),
                pltpu.VMEM((rows, hidden), jnp.float32),
            ),
            grid=()),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes or vmem_budget(
                rows, hidden, inter, w13_parts, two_inter, n_experts)),
        name="moe_lowtok",
    )(gid, hit, x_rows, wtab, s13, s2, w13, w2)
    return out[:tokens]


@functools.partial(jax.jit,
                   static_argnames=("topk", "ranks", "owned",
                                    "vmem_limit_bytes", "has_copies"))
def moe_lowtok_scores(x,
                      scores,
                      rank,
                      gid,
                      prim,
                      copy,
                      w13,
                      s13,
                      w2,
                      s2,
                      *,
                      topk,
                      ranks,
                      owned=1,
                      vmem_limit_bytes=None,
                      hot=None,
                      hprim=None,
                      hcopy=None,
                      hot1h=None,
                      slot=None,
                      has_copies=None,
                      tables=None):
    """The shard's MoE output for a few tokens, with the router's top-k and the
    holder assignment inside. tables: the scalar tables packed by
    `pack_tables` (the served layer builds them at load); when given, rank,
    gid, hot, hprim, hcopy and slot are not read and hot1h is required. scores [tokens, LANES] f32: the softmax over
    all experts; rank: this rank (int32 scalar), or with owned > 1 the rank's
    first owned expert id (experts_start), divided in the kernel; gid [n_local] int32: the
    global expert in each local slot; prim / copy [LANES] int32: the rank
    holding each global expert's primary / copy (-1: none); hot [NC] int32:
    the copied experts, hottest first (-1 padded), with hprim / hcopy [NC]
    the rank of each one's primary and copy (built from prim / copy when
    not given). A hit expert is served by exactly one rank: its only holder,
    or, with a copy, the home with fewer experts assigned so far when its
    turn comes (hottest first, ties to the primary). Returns [tokens, hidden]
    in x's dtype; the all-reduce over ranks is the caller's."""
    tokens, hidden = x.shape
    n_experts, _, two_inter = w13.shape
    inter = two_inter // 2
    if scores.shape[1] != LANES or prim.shape[0] != LANES or copy.shape[
            0] != LANES:
        raise ValueError(f"the routing table is {LANES} experts wide")
    if inter % LANE:
        raise ValueError(
            f"the intermediate width {inter} must be a multiple of the lane tile"
        )
    w13_parts = 2 if (inter // 2) % LANE == 0 else 1
    rows = -(-tokens // ROW_TILE) * ROW_TILE
    x_rows = jnp.pad(x, ((0, rows - tokens), (0, 0)))
    sc = jnp.pad(scores.astype(jnp.float32), ((0, rows - tokens), (0, 0)))
    s13 = s13.reshape(n_experts, 1, two_inter).astype(jnp.float32)
    s2 = s2.reshape(n_experts, 1, hidden).astype(jnp.float32)
    prim = jnp.asarray(prim, jnp.int32).reshape(1, LANES)
    copy = jnp.asarray(copy, jnp.int32).reshape(1, LANES)
    if has_copies is None:
        # a concrete copy table says whether any expert has a copy; a traced
        # one cannot, and the general path stays
        try:
            has_copies = bool(np.any(np.asarray(copy) >= 0))
        except Exception:
            has_copies = True
    has_copies = bool(has_copies)
    if tables is None:
        # the tables packed here, per call (tests and the reference use);
        # the served layer packs them once at load
        rk = jnp.asarray(rank, jnp.int32).reshape(1)
        gid = jnp.asarray(gid, jnp.int32).reshape(1, n_experts)
        if hot is None:
            hot, hprim, hcopy = hot_tables(prim.reshape(LANES),
                                           copy.reshape(LANES))
        hot = jnp.asarray(hot, jnp.int32).reshape(1, -1)
        hprim = jnp.asarray(hprim, jnp.int32).reshape(1, -1)
        hcopy = jnp.asarray(hcopy, jnp.int32).reshape(1, -1)
        nc = hot.shape[1]
        if hot1h is None:
            hot1h = hot_onehot(hot.reshape(-1))
        # this rank's local slot of every global expert (-1: none), built here from the slot -> global id table
        if slot is None:
            slot = slot_table(gid.reshape(-1), n_experts)
        slot = jnp.asarray(slot, jnp.int32).reshape(1, LANES)
        o_gid, o_hot, o_hprim, o_hcopy, o_slot, length = table_offsets(
            n_experts, nc)
        tab = jnp.full((length, ), -1, jnp.int32)
        tab = tab.at[0].set(rk[0]).at[o_gid:o_gid + n_experts].set(gid[0])
        tab = tab.at[o_hot:o_hot + nc].set(hot[0]).at[o_hprim:o_hprim +
                                                      nc].set(hprim[0])
        tab = tab.at[o_hcopy:o_hcopy + nc].set(hcopy[0]).at[o_slot:o_slot +
                                                            LANES].set(slot[0])
    else:
        if hot1h is None:
            raise ValueError("packed tables need the one-hot table hot1h")
        nc = int(hot1h.shape[-1])
        o_gid, o_hot, o_hprim, o_hcopy, o_slot, length = table_offsets(
            n_experts, nc)
        tab = jnp.asarray(tables, jnp.int32)
        if tab.shape != (length, ):
            raise ValueError(
                f"packed tables for {n_experts} slots and {nc} hot entries are [{length}], got {tab.shape}"
            )
    if nc % LANE:
        raise ValueError(
            f"the hot table's length {nc} must be a multiple of the lane tile")
    hot1h = jnp.asarray(hot1h, jnp.bfloat16).reshape(LANES, nc)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out = pl.pallas_call(
        functools.partial(_kernel_scores,
                          n_experts=n_experts,
                          inter=inter,
                          w13_parts=w13_parts,
                          topk=topk,
                          ranks=ranks,
                          owned=owned,
                          has_copies=has_copies,
                          offsets=(o_gid, o_hot, o_hprim, o_hcopy, o_slot)),
        out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, vmem, vmem, hbm, vmem, hbm, hbm, hbm, hbm],
            out_specs=vmem,
            scratch_shapes=(
                pltpu.SMEM((n_experts, ), jnp.int32),
                pltpu.VMEM((rows, LANES), jnp.float32),
                pltpu.VMEM((1, LANES), jnp.int32),
                pltpu.SMEM((1, LANES), jnp.int32),
                pltpu.VMEM((8, nc), jnp.int32),
                pltpu.SMEM((8, nc), jnp.int32),
                pltpu.SemaphoreType.DMA((3, )),
                pltpu.VMEM((rows, hidden), x_rows.dtype),
                pltpu.SemaphoreType.DMA((1, )),
                pltpu.VMEM((w13_parts + 1, hidden, 2 * (inter // w13_parts)),
                           FP8),
                pltpu.VMEM((w13_parts + 1, inter, hidden), FP8),
                pltpu.VMEM((w13_parts + 1, 1, 2 * (inter // w13_parts)),
                           jnp.float32),
                pltpu.VMEM((w13_parts + 1, 1, hidden), jnp.float32),
                pltpu.SemaphoreType.DMA((w13_parts + 1, )),
                pltpu.SemaphoreType.DMA((w13_parts + 1, )),
                pltpu.VMEM((rows, two_inter), jnp.float32),
                pltpu.VMEM((rows, hidden), jnp.float32),
            ),
            grid=()),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes or vmem_budget(
                rows, hidden, inter, w13_parts, two_inter, n_experts)),
        name="moe_lowtok",
    )(tab, prim, copy, hot1h, x_rows, sc, s13, s2, w13, w2)
    return out[:tokens]


def table_offsets(n_experts, nc):
    """The flat packed table's layout for `n_experts` local slots and `nc`
    hot entries: (gid, hot, hprim, hcopy, slot) offsets and the padded
    length (a multiple of the lane tile)."""
    o_gid = 8
    o_hot = o_gid + n_experts
    o_hprim = o_hot + nc
    o_hcopy = o_hprim + nc
    o_slot = o_hcopy + nc
    length = -(-(o_slot + LANES) // LANE) * LANE
    return o_gid, o_hot, o_hprim, o_hcopy, o_slot, length


def pack_tables(rank, gid, hot, hprim, hcopy, slot):
    """The scores entry's scalar tables as one flat int32 array (see
    table_offsets): [0] the rank (or a rank's first owned expert id, which
    the kernel divides by the owned count), the global id of each local
    slot, the copied experts hottest first with the rank of each one's
    primary and copy (-1 padded), this rank's local slot of every global
    expert (-1: none). Packing the tables uses one scalar-prefetch operand
    for all six tables."""
    gid = np.asarray(gid, np.int32).reshape(-1)
    hot, hprim, hcopy = (np.asarray(t, np.int32).reshape(-1)
                         for t in (hot, hprim, hcopy))
    slot = np.asarray(slot, np.int32).reshape(-1)
    if slot.size != LANES:
        raise ValueError(
            f"the slot table is {LANES} lanes wide, got {slot.size}")
    if not (hprim.size == hot.size == hcopy.size):
        raise ValueError("hot, hprim and hcopy must be the same length")
    o_gid, o_hot, o_hprim, o_hcopy, o_slot, length = table_offsets(
        gid.size, hot.size)
    tab = np.full((length, ), -1, np.int32)
    tab[0] = int(rank)
    tab[o_gid:o_gid + gid.size] = gid
    tab[o_hot:o_hot + hot.size] = hot
    tab[o_hprim:o_hprim + hot.size] = hprim
    tab[o_hcopy:o_hcopy + hot.size] = hcopy
    tab[o_slot:o_slot + LANES] = slot
    return tab


def slot_table(gid, n_experts):
    """This rank's local slot of every global expert, -1 where it holds none: [LANES] int32 from the rank's
    slot -> global id table."""
    gid = jnp.asarray(gid, jnp.int32).reshape(-1)
    return jnp.full(
        (LANES, ), -1,
        jnp.int32).at[gid].set(jnp.arange(n_experts, dtype=jnp.int32))


def hot_onehot(hot):
    """The one-hot gather matrix of a hot table: [LANES, len(hot)] bf16 with a 1 at (hot[i], i) for the real
    entries (-1 padding gives an all-zero column)."""
    hot = jnp.asarray(hot, jnp.int32).reshape(-1)
    lanes = jax.lax.broadcasted_iota(jnp.int32, (LANES, hot.shape[0]), 0)
    return jnp.where(lanes == hot.reshape(1, -1), 1.0,
                     0.0).astype(jnp.bfloat16)


HOT_SLOTS = 128  # the copied-expert tables' fixed length (-1 padded)


def hot_tables(prim, copy, order=None, slots=HOT_SLOTS):
    """The copied experts hottest first (order: table ids; without one, by id) with the rank of each one's
    primary and copy, -1 padded to `slots` entries: the scalar loop's tables. Works on traced arrays (the
    default order inside the jitted entry) and on host arrays (the served tables at load)."""
    prim = jnp.asarray(prim, jnp.int32).reshape(-1)
    copy = jnp.asarray(copy, jnp.int32).reshape(-1)
    if order is not None:
        o = [int(e) for e in order]
        if len(o) > slots:
            raise ValueError(
                f"{len(o)} copied experts exceed the {slots}-entry hot table")
        if len(set(o)) != len(o) or any(e < 0 or e >= prim.shape[0]
                                        for e in o):
            raise ValueError("hot order must contain unique, valid expert ids")
        hot = np.full(slots, -1, np.int32)
        hot[:len(o)] = o
        hot = jnp.asarray(hot)
    else:
        hot = jnp.nonzero(copy >= 0, size=slots,
                          fill_value=-1)[0].astype(jnp.int32)
    safe = jnp.where(hot >= 0, hot, 0)
    hprim = jnp.where(hot >= 0, prim[safe], -1).astype(jnp.int32)
    hcopy = jnp.where((hot >= 0) & (copy[safe] >= 0), copy[safe],
                      -1).astype(jnp.int32)
    return hot, hprim, hcopy


def contiguous_tables(experts_start, n_local, ranks):
    """The holder tables of the served placement: rank r holds experts r*n_local .. (r+1)*n_local-1, no copies."""
    gid = experts_start + jnp.arange(n_local, dtype=jnp.int32)
    prim = jnp.arange(LANES, dtype=jnp.int32) // n_local
    copy = jnp.full((LANES, ), -1, jnp.int32)
    return gid, prim, copy


def router_select_reference(scores, topk):
    """The served router in plain JAX: max-and-mask passes, ties to the lowest id, renormalised, bf16."""
    x = jnp.where(scores > NEG, scores, NEG)
    n = x.shape[1]
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    ws, ids = [], []
    for _ in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
        ws.append(m[:, 0])
        ids.append(idx)
        x = jnp.where(iota == idx[:, None], jnp.finfo(x.dtype).min, x)
    w = jnp.stack(ws, axis=1)
    i = jnp.stack(ids, axis=1)
    w = w / jnp.sum(w, axis=1, keepdims=True)
    return w.astype(jnp.bfloat16), i.astype(jnp.int32)


def moe_lowtok_reference(x, topk_ids_local, topk_weights, w13, s13, w2, s2):
    """Plain JAX with the same arithmetic (fp8 weights, per-channel scales,
    per-token fp8 activations), for the parity test."""
    tokens, hidden = x.shape
    n_experts, _, two_inter = w13.shape
    inter = two_inter // 2
    xf = x.astype(jnp.float32)
    x_q, x_s = _quantize_rows(xf)
    s13 = s13.reshape(n_experts, two_inter).astype(jnp.float32)
    s2 = s2.reshape(n_experts, hidden).astype(jnp.float32)
    out = jnp.zeros((tokens, hidden), jnp.float32)
    for e in range(n_experts):
        w = jnp.sum(jnp.where(topk_ids_local == e, topk_weights,
                              0).astype(jnp.float32),
                    axis=1)  # [tokens]
        h = jnp.dot(x_q, w13[e],
                    preferred_element_type=jnp.float32) * x_s * s13[e][None, :]
        a = h[:, :inter] * jax.nn.sigmoid(h[:, :inter]) * h[:, inter:]
        a_q, a_s = _quantize_rows(a)
        o = jnp.dot(a_q, w2[e],
                    preferred_element_type=jnp.float32) * a_s * s2[e][None, :]
        out = out + w[:, None] * o
    return out.astype(x.dtype)
