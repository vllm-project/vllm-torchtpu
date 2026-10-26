# SPDX-License-Identifier: Apache-2.0
"""The decode routing plan as one Pallas call.

At a decode bucket the sharded routing plan (host.build_routing_tables_sharded
and the shard_* cutters after it) is about a hundred XLA ops over arrays of a
few hundred integers, each paying a fixed launch cost, plus two SparseCore
scatters the TensorCore waits on. Here the same kernel operands come out of
one VMEM-resident pass over the gathered selections.

Every prefix sum is a matmul against a 0/1 matrix. The operands are split
into bytes so that each bf16 product is exact and every f32 partial sum stays
below 2**24, which makes the integer result exact and order independent.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v2.host import (
    HIDDEN_LANE_BLOCK,
    ROWBLK,
    act_scale_slab_rows,
    gather_clamp_bounds,
    pow2_shift,
    token_gather_window_rows,
)

# Table values must stay below 2**16 for the two-byte split below to be exact.
_MAX_TABLE_VALUE = 1 << 16


class DecodePlan(NamedTuple):
    """The kernel operands and combine tables of one shard, as the layer uses them."""

    block_tables: tuple  # run_start, aligned rows, contrib offset, push dest
    true_rows: jax.Array  # [G, ep] rows per (local expert, destination)
    token_gather: jax.Array  # [shard_stride + window] i32
    scale_slab: jax.Array | None  # [scale rows, 128] f32
    expert_rows: jax.Array  # [G]
    slab_base: jax.Array  # [G]
    visit: jax.Array  # [G]
    counts: jax.Array  # [N_COUNTS, ep]
    pos: jax.Array  # [t_local, K]
    mirror_pos: jax.Array  # [t_local, K]


def _iota(shape, dim):
    return lax.broadcasted_iota(jnp.int32, shape, dim)


def _bf16(x):
    return x.astype(jnp.float32).astype(jnp.bfloat16)


def _exact_dot(a, b, dims):
    """Integer dot where exactly one side is a 0/1 matrix and the other is
    in [0, 2**16): split the integer side into bytes, one pass each."""
    int_left = a.dtype == jnp.int32
    x = a if int_left else b
    out = None
    for shift in (8, 0):
        part = _bf16(jnp.right_shift(x, shift) & 255)
        lhs, rhs = (part, b) if int_left else (a, part)
        r = lax.dot_general(
            lhs, rhs, (dims, ((), ())), preferred_element_type=jnp.float32
        )
        out = r if out is None else out * 256.0 + r
    return out.astype(jnp.int32)


def _mm(a, b):
    """[m, k] @ [k, n]."""
    return _exact_dot(a, b, ((1,), (0,)))


def _mm_nt(a, b):
    """[m, k] @ [n, k].T."""
    return _exact_dot(a, b, ((1,), (1,)))


def _plan_kernel(
    me_ref,
    pairs_ref,
    *refs,
    e_total,
    ep,
    t_local,
    topk,
    block,
    tile_m,
    shard_stride,
    row_stride,
    gather_rows,
    scale_lanes,
    gather_hi,
):
    it = iter(refs)
    scale_ref = next(it) if scale_lanes else None
    (rs_ref, al_ref, co_ref, pd_ref, tr_ref, tg_ref) = (next(it) for _ in range(6))
    ss_ref = next(it) if scale_lanes else None
    er_ref, sb_ref, vi_ref, cn_ref, pos_ref, mir_ref = it
    E, G, P = e_total, e_total // ep, t_local * topk
    me = me_ref[0]
    first = me * G
    pairs = pairs_ref[...]  # [ep, P]
    lane_p = _iota((ep, P), 1)

    def column(x, lanes, j):
        """x[:, j:j+1] as a masked lane sum, which every layout supports."""
        return jnp.sum(jnp.where(lanes == j, x, 0), axis=1, keepdims=True)

    lane_e = _iota((ep, E), 1)
    counts = jnp.zeros((ep, E), jnp.int32)
    for j in range(P):
        counts = counts + (column(pairs, lane_p, j) == lane_e).astype(jnp.int32)
    # Rows from source d to expert e, padded to whole ROWBLK runs.
    aligned = (counts + (ROWBLK - 1)) & jnp.int32(-ROWBLK)

    lower_d = _bf16(_iota((ep, ep), 1) < _iota((ep, ep), 0))  # [d, d'] d' < d
    upper_e = _bf16(_iota((E, E), 0) < _iota((E, E), 1))  # [e', e] e' < e
    same_shard = _iota((E, E), 0) // G == _iota((E, E), 1) // G
    upper_seg = _bf16((_iota((E, E), 0) < _iota((E, E), 1)) & same_shard)

    run_start = _mm(lower_d, aligned)  # [d, e] exclusive over sources
    recv_base = _mm(aligned, upper_e)  # [d, e] exclusive over experts
    expert_rows_all = jnp.broadcast_to(jnp.sum(aligned, axis=0, keepdims=True), (8, E))
    local_base = _mm(expert_rows_all, upper_seg)  # [8, E], rows identical
    shift = pow2_shift(tile_m, "tile_m")
    last = run_start + jnp.maximum(aligned, 1) - 1
    tiles = jnp.where(
        aligned > 0,
        jnp.right_shift(last, shift) - jnp.right_shift(run_start, shift) + 1,
        0,
    )

    # This shard's experts as rows: [g, e] picks e = first + g.
    sel = _bf16(_iota((G, E), 1) == first + _iota((G, E), 0))
    sel_t = _bf16(_iota((E, G), 0) == first + _iota((E, G), 1))
    run_start_g = _mm_nt(sel, run_start)  # [G, ep]
    aligned_g = _mm_nt(sel, aligned)
    counts_g = _mm_nt(sel, counts)
    recv_base_g = _mm_nt(sel, recv_base)
    tiles_g = _mm_nt(sel, tiles)
    lower_g = _bf16(_iota((G, G), 1) < _iota((G, G), 0))
    upper_d = _bf16(_iota((ep, ep), 0) < _iota((ep, ep), 1))
    out_total = jnp.broadcast_to(jnp.sum(aligned_g, axis=0, keepdims=True), (8, ep))
    contrib_off = _mm(out_total, upper_d)[0:1] + _mm(lower_g, aligned_g)
    rs_ref[...] = jnp.right_shift(run_start_g, ROWBLK.bit_length() - 1)
    al_ref[...] = jnp.right_shift(aligned_g, ROWBLK.bit_length() - 1)
    co_ref[...] = jnp.right_shift(contrib_off, ROWBLK.bit_length() - 1)
    pd_ref[...] = jnp.right_shift(recv_base_g, ROWBLK.bit_length() - 1)
    tr_ref[...] = counts_g

    rows_row = _mm(expert_rows_all, sel_t)[0:1]  # [1, G]
    er_ref[...] = rows_row
    sb_ref[...] = _mm(local_base, sel_t)[0:1]
    rows_col = _mm_nt(sel, expert_rows_all)[:, 0:1]  # [G, 1]
    key_row = _iota((1, G), 1) - rows_row * G
    key_col = _iota((G, 1), 0) - rows_col * G
    rank_col = jnp.sum((key_row < key_col).astype(jnp.int32), axis=1, keepdims=True)
    perm = jnp.sum(
        jnp.where(rank_col == _iota((G, G), 1), _iota((G, G), 0), 0),
        axis=0,
        keepdims=True,
    )
    vi_ref[...] = jnp.minimum(perm, G - 1)

    ones = _bf16(jnp.ones((8, E), jnp.int32))
    lane_d = _iota((1, ep), 1)
    mine = (lane_d == me).astype(jnp.int32)
    other = 1 - mine

    def over_g(x):
        return jnp.sum(x, axis=0, keepdims=True)

    recv_rows_all = _mm_nt(ones, counts)[0:1]
    recv_total = _mm_nt(ones, aligned)[0:1]
    recv_tiles_all = _mm_nt(ones, tiles)[0:1]
    n_visit = jnp.sum((rows_row > 0).astype(jnp.int32), axis=1, keepdims=True)
    for row, value in enumerate(
        (
            other * over_g(counts_g),
            mine * (recv_rows_all - over_g(counts_g)),
            other * over_g(aligned_g),
            mine * (recv_total - over_g(aligned_g)),
            mine * n_visit,
            other * over_g(tiles_g),
            mine * (recv_tiles_all - over_g(tiles_g)),
            mine * over_g(tiles_g),
        )
    ):
        cn_ref[row : row + 1, :] = value

    # This shard's own pairs: where each arrives and its mirror element.
    mine_d = _iota((ep, E), 0) == me
    recv_base_me = jnp.sum(jnp.where(mine_d, recv_base, 0), axis=0, keepdims=True)
    run_start_me = jnp.sum(jnp.where(mine_d, run_start, 0), axis=0, keepdims=True)
    my_pairs = pairs_ref[pl.ds(me, 1), :]  # [1, P]
    # One bin past the experts holds the id E the layer gives a row that
    # routes nowhere: such a pair is in no table, so its lookups are zero
    # and its rank is block-local, as in the sharded builder.
    bins = _iota((E + 8, P), 0)
    onehot_ext = (bins == my_pairs).astype(jnp.int32)  # [e, p]
    before = _iota((P, P), 0) < _iota((P, P), 1)
    same_block = _iota((P, P), 0) // block == _iota((P, P), 1) // block

    def rank_among(earlier):
        prior = _mm(onehot_ext, _bf16(earlier))
        return jnp.sum(onehot_ext * prior, axis=0, keepdims=True)

    valid = (my_pairs >= 0) & (my_pairs < E)
    rank = jnp.where(valid, rank_among(before), rank_among(before & same_block))
    onehot_b = _bf16(onehot_ext[:E])

    def lookup(table_row):
        return _mm(jnp.broadcast_to(table_row, (8, E)), onehot_b)[0:1]

    lane_e1 = _iota((1, E), 1)
    pos_ref[...] = lookup(recv_base_me) + rank
    mir_ref[...] = (
        lookup(jnp.right_shift(recv_base_me, shift) + 2 * lane_e1) * tile_m
        + lookup(run_start_me & (tile_m - 1))
        + rank
    )

    # Every pair computing on this shard: its slab row, then the token it
    # reads and its activation scale scattered onto that row.
    slab_base = local_base[0:1] + run_start  # [d, e]
    seen = jnp.zeros((ep, E), jnp.int32)
    lane_g = _iota((ep, gather_rows), 1)
    gather = jnp.zeros((ep, gather_rows), jnp.int32)
    if scale_lanes:
        lane_s = _iota((ep, scale_lanes), 1)
        scale = jnp.zeros((ep, scale_lanes), jnp.int32)
        bits_all = scale_ref[...]  # [ep, t_local]
        lane_t = _iota((ep, t_local), 1)
    source = _iota((ep, 1), 0)
    for j in range(P):
        e_j = column(pairs, lane_p, j)
        hit = e_j == lane_e
        row = jnp.sum(jnp.where(hit, slab_base + seen, 0), axis=1, keepdims=True)
        seen = seen + hit.astype(jnp.int32)
        row = jnp.where((e_j >= first) & (e_j < first + G), row, -1)
        row = jnp.where(row < shard_stride, row, -1)
        token = source * row_stride + j // topk
        gather = gather + jnp.where(lane_g == row, token, 0)
        if scale_lanes:
            bits = column(bits_all, lane_t, j // topk)
            scale = scale + jnp.where(lane_s == row, bits, 0)
    tg_ref[...] = jnp.clip(
        jnp.sum(gather, axis=0, keepdims=True), 0, jnp.int32(gather_hi)
    )
    if scale_lanes:
        ss_ref[...] = lax.bitcast_convert_type(
            jnp.sum(scale, axis=0, keepdims=True), jnp.float32
        )


def decode_plan_supported(*, e_total, ep, t_local, topk, shard_stride):
    """Whether build_decode_plan covers this shape exactly."""
    g_local = e_total // ep
    return (
        e_total % ep == 0
        and g_local % 8 == 0
        and t_local * topk <= 128
        and ep <= 128
        and 2 * (shard_stride + 2 * e_total) < _MAX_TABLE_VALUE
    )


def build_decode_plan(
    pairs_g,
    scale_bits_g,
    rank,
    *,
    e_total,
    ep,
    t_local,
    topk,
    block,
    tile_m,
    shard_stride,
    row_stride=None,
    interpret=False,
):
    """Shard ``rank``'s routing-plan operands from the gathered selections.

    pairs_g [ep, t_local * topk] i32 holds every source's selected expert
    slots in token-major order, e_total marking a pair that routes nowhere;
    scale_bits_g [ep * t_local] i32 the source rows' activation scales, or
    None for unquantized rows. ``row_stride`` is how many rows each source
    occupies in the gathered activations (t_local unless more ride along).
    The values are those of build_routing_tables_sharded and the shard_*
    cutters, bit for bit.
    """
    if not decode_plan_supported(
        e_total=e_total, ep=ep, t_local=t_local, topk=topk, shard_stride=shard_stride
    ):
        raise ValueError(
            f"the decode plan does not cover e_total={e_total} ep={ep} "
            f"t_local={t_local} topk={topk} shard_stride={shard_stride}"
        )
    G, P = e_total // ep, t_local * topk
    row_stride = t_local if row_stride is None else row_stride
    gather_rows = shard_stride + token_gather_window_rows(tile_m)
    quantized = scale_bits_g is not None
    scale_rows = act_scale_slab_rows(shard_stride)
    scale_lanes = scale_rows * HIDDEN_LANE_BLOCK if quantized else 0
    T = ep * t_local
    _, gather_hi = gather_clamp_bounds(
        (T - 1) // t_local * row_stride + (T - 1) % t_local + 1
    )
    kernel = functools.partial(
        _plan_kernel,
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        topk=topk,
        block=block,
        tile_m=tile_m,
        shard_stride=shard_stride,
        row_stride=row_stride,
        gather_rows=gather_rows,
        scale_lanes=scale_lanes,
        gather_hi=gather_hi,
    )
    i32 = jnp.int32
    out_shape = [jax.ShapeDtypeStruct((G, ep), i32) for _ in range(5)]
    out_shape.append(jax.ShapeDtypeStruct((1, gather_rows), i32))
    if quantized:
        out_shape.append(jax.ShapeDtypeStruct((1, scale_lanes), jnp.float32))
    out_shape += [jax.ShapeDtypeStruct((1, G), i32) for _ in range(3)]
    out_shape.append(jax.ShapeDtypeStruct((8, ep), i32))
    out_shape += [jax.ShapeDtypeStruct((1, P), i32) for _ in range(2)]
    inputs = [pairs_g.astype(i32).reshape(ep, P)]
    if quantized:
        inputs.append(scale_bits_g.astype(i32).reshape(ep, t_local))
    vmem = pltpu.MemorySpace.VMEM
    n_in, n_out = len(inputs), len(out_shape)
    scale_index = 6 if quantized else None
    if interpret:
        out_spaces, body, scratch_shapes = [vmem] * n_out, kernel, []
        final_shape = out_shape
    else:
        # The plan is computed into VMEM scratch and DMA'd by the kernel itself
        # to outputs declared in HBM, every copy in flight at once. As VMEM
        # outputs, XLA placed them in VMEM and then copied each one to HBM for
        # the fused EP kernel, serially, ~0.59 us apiece (8 per decode layer
        # at TP8 T16, j-6cb83e31). The scale slab lands directly in the
        # kernel's [rows, 128] layout, one row DMA per lane block, rather than
        # as a [1, rows * 128] row XLA relayouts (0.59 us).
        hbm = pltpu.MemorySpace.HBM
        final_shape = [
            hbm(
                (scale_rows, HIDDEN_LANE_BLOCK) if i == scale_index else o.shape,
                o.dtype,
            )
            for i, o in enumerate(out_shape)
        ]
        out_spaces = [hbm] * n_out
        scratch_shapes = [pltpu.VMEM(o.shape, o.dtype) for o in out_shape] + [
            pltpu.SemaphoreType.DMA
        ]

        def body(me_ref, *refs):
            ins = refs[:n_in]
            outs_hbm = refs[n_in : n_in + n_out]
            scratch = refs[n_in + n_out : n_in + 2 * n_out]
            sem = refs[n_in + 2 * n_out]
            kernel(me_ref, *ins, *scratch)
            copies = []
            for i, (src, dst) in enumerate(zip(scratch, outs_hbm)):
                if i == scale_index:
                    for r in range(scale_rows):
                        copies.append(
                            pltpu.make_async_copy(
                                src.at[:, pl.ds(r * HIDDEN_LANE_BLOCK, HIDDEN_LANE_BLOCK)],
                                dst.at[pl.ds(r, 1), :],
                                sem,
                            )
                        )
                else:
                    copies.append(pltpu.make_async_copy(src, dst, sem))
            for c in copies:
                c.start()
            for c in copies:
                c.wait()

    outs = pl.pallas_call(
        body,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(),
            in_specs=[pl.BlockSpec(memory_space=vmem) for _ in inputs],
            out_specs=[pl.BlockSpec(memory_space=m) for m in out_spaces],
            scratch_shapes=scratch_shapes,
        ),
        out_shape=final_shape,
        interpret=interpret,
        name="fused_ep_moe_v2_decode_plan",
    )(rank.reshape(1).astype(i32), *inputs)
    it = iter(outs)
    rs, al, co, pd, tr, tg = (next(it) for _ in range(6))
    ss = next(it).reshape(scale_rows, HIDDEN_LANE_BLOCK) if quantized else None
    er, sb, vi, cn, pos, mir = it
    return DecodePlan(
        block_tables=(rs, al, co, pd),
        true_rows=tr,
        token_gather=tg.reshape(gather_rows),
        scale_slab=ss,
        expert_rows=er.reshape(G),
        slab_base=sb.reshape(G),
        visit=vi.reshape(G),
        counts=cn,
        pos=pos.reshape(t_local, topk),
        mirror_pos=mir.reshape(t_local, topk),
    )


def _combine_kernel(
    tables_hbm,
    arr_hbm,
    scale_hbm,
    out_ref,
    tables_sm,
    rows_vm,
    scales_vm,
    sums_vm,
    sems,
    *,
    t_local,
    topk,
    tile_m,
    out_dtype,
):
    n = t_local * topk
    fetch = pltpu.make_async_copy(tables_hbm, tables_sm, sems.at[2])
    fetch.start()
    fetch.wait()
    recv_rows = arr_hbm.shape[0]
    mirror_elems = scale_hbm.shape[0] * tile_m
    shift = pow2_shift(tile_m, "tile_m")
    mirror = [jnp.minimum(tables_sm[n + p], mirror_elems - 1) for p in range(n)]
    copies = []
    for p in range(n):
        row = jnp.minimum(tables_sm[p], recv_rows - 1)
        sub = jnp.right_shift(mirror[p], shift)
        copies.append(pltpu.make_async_copy(arr_hbm.at[row], rows_vm.at[p], sems.at[0]))
        copies.append(
            pltpu.make_async_copy(
                scale_hbm.at[pl.ds(sub, 1)], scales_vm.at[p], sems.at[1]
            )
        )
    for c in copies:
        c.start()
    for c in copies:
        c.wait()
    lane = _iota((1, tile_m), 1)
    for t in range(t_local):
        acc = None
        for k in range(topk):
            p = t * topk + k
            row_scale = jnp.sum(
                jnp.where(lane == (mirror[p] & (tile_m - 1)), scales_vm[p], 0.0),
                axis=1,
                keepdims=True,
            )
            weight = lax.bitcast_convert_type(tables_sm[2 * n + p], jnp.float32)
            term = rows_vm[p].astype(jnp.float32) * (row_scale * weight)
            acc = term if acc is None else acc + term
        sums_vm[t] = acc.astype(out_dtype)
    store = pltpu.make_async_copy(sums_vm, out_ref, sems.at[0])
    store.start()
    store.wait()


def build_decode_combine(arrivals, arrival_scales, pos, mirror_pos, weights, out_dtype):
    """out[t] = sum_k scale(mirror[t, k]) * weights[t, k] * arrivals[pos[t, k]].

    The decode-bucket combine as one Mosaic call: topk row DMAs per token in
    place of two SparseCore gathers. Returns [t_local, lane blocks, 128].
    Positions are clamped into both buffers as the XLA gather clamps them.
    The three per-slot tables travel as one HBM array the kernel fetches into
    SMEM itself, rather than as three operands XLA copies there one by one.
    """
    t_local, topk = pos.shape
    _, lane_blocks, lanes = arrivals.shape
    tile_m = arrival_scales.shape[1]
    n = t_local * topk
    hbm = pltpu.MemorySpace.HBM
    tables = jnp.concatenate(
        [
            pos.astype(jnp.int32).reshape(n),
            mirror_pos.astype(jnp.int32).reshape(n),
            lax.bitcast_convert_type(weights.astype(jnp.float32), jnp.int32).reshape(n),
        ]
    )
    kernel = functools.partial(
        _combine_kernel, t_local=t_local, topk=topk, tile_m=tile_m, out_dtype=out_dtype
    )
    return pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=hbm)] * 3,
        out_specs=pl.BlockSpec(memory_space=hbm),
        scratch_shapes=(
            pltpu.SMEM((3 * n,), jnp.int32),
            pltpu.VMEM((n, lane_blocks, lanes), arrivals.dtype),
            pltpu.VMEM((n, 1, tile_m), jnp.float32),
            pltpu.VMEM((t_local, lane_blocks, lanes), out_dtype),
            pltpu.SemaphoreType.DMA((3,)),
        ),
        out_shape=jax.ShapeDtypeStruct((t_local, lane_blocks, lanes), out_dtype),
        name="fused_ep_moe_v2_decode_combine",
    )(
        tables,
        # Pinned to HBM: the kernel DMAs only n rows of each, but an
        # unconstrained operand is prefetched whole into VMEM by XLA first
        # (the [recv_rows, 56, 128] arrivals and the scale mirror, 1.8 + 1.0 us
        # on the critical path per decode layer at TP8 T16).
        pltpu.with_memory_space_constraint(arrivals, hbm),
        pltpu.with_memory_space_constraint(arrival_scales, hbm),
    )


# Barrier semaphore group of the fused combine + restore: the fused EP MoE
# kernel uses 0, the small all-reduce 5, the small all-gather 6 and the
# one-shot gather 7.
COMBINE_RESTORE_COLLECTIVE_ID = 8


def _combine_restore_kernel(
    rank_ref,
    peers_ref,
    tables_hbm,
    arr_hbm,
    scale_hbm,
    out_hbm,
    tables_sm,
    rows_vm,
    scales_vm,
    sums_vm,
    sems,
    *,
    t_local,
    topk,
    tile_m,
    out_dtype,
    n_cores,
    axis,
):
    from vllm_torchtpu.kernels.experimental.small_collectives.common import (
        MESH,
        Group,
        wait_bytes_of,
    )

    n = t_local * topk
    me = rank_ref[0]
    group = Group(n_cores, axis, True, peers_ref)
    # Split barrier: signal every member now, wait only before the pushes,
    # so its round trip hides under the row fetches and the sum.
    barrier = pltpu.get_barrier_semaphore()
    for j in range(n_cores):
        pl.semaphore_signal(
            barrier, 1, device_id=group.device_id(j), device_id_type=MESH
        )
    fetch = pltpu.make_async_copy(tables_hbm, tables_sm, sems.at[2])
    fetch.start()
    fetch.wait()
    recv_rows = arr_hbm.shape[0]
    mirror_elems = scale_hbm.shape[0] * tile_m
    shift = pow2_shift(tile_m, "tile_m")
    mirror = [jnp.minimum(tables_sm[n + p], mirror_elems - 1) for p in range(n)]
    copies = []
    for p in range(n):
        row = jnp.minimum(tables_sm[p], recv_rows - 1)
        sub = jnp.right_shift(mirror[p], shift)
        copies.append(pltpu.make_async_copy(arr_hbm.at[row], rows_vm.at[p], sems.at[0]))
        copies.append(
            pltpu.make_async_copy(
                scale_hbm.at[pl.ds(sub, 1)], scales_vm.at[p], sems.at[1]
            )
        )
    for c in copies:
        c.start()
    for c in copies:
        c.wait()
    lane = _iota((1, tile_m), 1)
    for t in range(t_local):
        acc = None
        for k in range(topk):
            p = t * topk + k
            row_scale = jnp.sum(
                jnp.where(lane == (mirror[p] & (tile_m - 1)), scales_vm[p], 0.0),
                axis=1,
                keepdims=True,
            )
            weight = lax.bitcast_convert_type(tables_sm[2 * n + p], jnp.float32)
            term = rows_vm[p].astype(jnp.float32) * (row_scale * weight)
            acc = term if acc is None else acc + term
        # A row that routed nowhere read arrival rows nobody wrote: zero it,
        # as the layer's jnp.where(row_routes, combined, 0) does.
        routes = tables_sm[3 * n + t] != 0
        sums_vm[t] = jnp.where(routes, acc.astype(out_dtype), jnp.zeros_like(acc, out_dtype))
    pl.semaphore_wait(barrier, n_cores)
    dst = out_hbm.at[pl.ds(me * t_local, t_local)]
    for j in range(n_cores):

        @pl.when(jnp.int32(j) != me)
        def _(j=j):
            pltpu.make_async_remote_copy(
                src_ref=sums_vm,
                dst_ref=dst,
                send_sem=sems.at[3],
                recv_sem=sems.at[4],
                device_id=group.device_id(j),
                device_id_type=MESH,
            ).start()

    local = pltpu.make_async_copy(sums_vm, dst, sems.at[0])
    local.start()
    wait_bytes_of(out_hbm.at[pl.ds(0, (n_cores - 1) * t_local)], sems.at[4])
    wait_bytes_of(out_hbm.at[pl.ds(0, (n_cores - 1) * t_local)], sems.at[3])
    local.wait()


def build_decode_combine_restore(
    arrivals,
    arrival_scales,
    pos,
    mirror_pos,
    weights,
    row_routes,
    out_dtype,
    *,
    num_cores,
    axis,
    rank,
    peers,
):
    """build_decode_combine followed by the TP-group restore, as one call.

    Each member sums its own t_local rows exactly as build_decode_combine
    does (zeroing rows that route nowhere), then pushes them straight from
    VMEM into its slot of every group member's output. Returns [num_cores *
    t_local, lane blocks, 128] in group order on every member -- the bytes
    small_all_gather would have produced from the combine's output, without
    the second kernel, its VMEM staging or the HBM round trip between them.
    rank (int32 [1]) is this core's position in its group, peers (int32
    [num_cores]) each member's mesh index, both as data.
    """
    t_local, topk = pos.shape
    _, lane_blocks, lanes = arrivals.shape
    tile_m = arrival_scales.shape[1]
    n = t_local * topk
    hbm = pltpu.MemorySpace.HBM
    tables = jnp.concatenate(
        [
            pos.astype(jnp.int32).reshape(n),
            mirror_pos.astype(jnp.int32).reshape(n),
            lax.bitcast_convert_type(weights.astype(jnp.float32), jnp.int32).reshape(n),
            row_routes.astype(jnp.int32).reshape(t_local),
        ]
    )
    kernel = functools.partial(
        _combine_restore_kernel,
        t_local=t_local,
        topk=topk,
        tile_m=tile_m,
        out_dtype=out_dtype,
        n_cores=num_cores,
        axis=axis,
    )
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    return pl.pallas_call(
        kernel,
        in_specs=[smem, smem] + [pl.BlockSpec(memory_space=hbm)] * 3,
        out_specs=pl.BlockSpec(memory_space=hbm),
        scratch_shapes=(
            pltpu.SMEM((3 * n + t_local,), jnp.int32),
            pltpu.VMEM((n, lane_blocks, lanes), arrivals.dtype),
            pltpu.VMEM((n, 1, tile_m), jnp.float32),
            pltpu.VMEM((t_local, lane_blocks, lanes), out_dtype),
            pltpu.SemaphoreType.DMA((5,)),
        ),
        out_shape=hbm((num_cores * t_local, lane_blocks, lanes), out_dtype),
        compiler_params=pltpu.CompilerParams(
            collective_id=COMBINE_RESTORE_COLLECTIVE_ID
        ),
        name=f"fused_ep_moe_v2_decode_combine_restore_n{num_cores}",
    )(
        rank.reshape(1).astype(jnp.int32),
        peers.reshape(-1).astype(jnp.int32),
        tables,
        pltpu.with_memory_space_constraint(arrivals, hbm),
        pltpu.with_memory_space_constraint(arrival_scales, hbm),
    )
