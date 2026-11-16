# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Row-gathered moe_tp_decode: each streamed expert multiplies only its routed
rows.

Same layer, same operands, same output bits as kernel.moe_tp_decode (fast
schedule), for decode row counts where the incumbent's dense-over-rows epilogue
binds (above ~64 rows). Differences from kernel._kernel:

  - x is quantized once per call (per row per 512-column block, as gmm_v2) into
    an f32 copy `xf` (the FP8 values, exact in f32) and per-row block scales
    `xsf` [m, 128] (lanes 0 .. h/512 - 1): f32 rows can be read at a runtime
    row index (strided row loads); packed FP8 rows cannot.
  - Slot tables, built once per call after routing: for every expert column e,
    the rows routed to it in ascending order, slots 0 .. R-1 (rows past R are
    the overflow), and the count n_e. rank = (strictly lower-triangular L) @ H
    on the MXU (0/1 operands, exact), slot rows by a select-sum per slot; moved
    VMEM -> SMEM like the hit table.
  - Per ring item: the expert's rows are gathered into an R-row slab (x values,
    block scales, routing-table rows), gate_up and the down projection run on
    the slab, and the result rows are added to the owner rows (TP: per item;
    expert range, n_chunks > 1: a compact f32 accumulator initialised from the
    gathered owner rows at the expert's first chunk and written back at its
    last, so every owner element sees the incumbent's additions in the
    incumbent's order).
  - The shared expert's partial is kept in bf16 (the combine rounds it to bf16
    first: same bits).

An expert with more than R routed rows runs the incumbent's dense finish over
all rows after the slab finish, in a loop of zero or one trip, so the overflow
path stays out of the pipelined ring body.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce, kernel
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_block

FP8 = kernel.FP8
FP8_MAX = kernel.FP8_MAX
# Rows of an expert's gathered slab; an expert routed more rows takes the dense
# finish. 32 covers every expert at decode row counts up to 192 for nearly all
# steps while keeping the slab buffers small.
R_SLAB = 32


def _pad_rows(m):
    return pl.cdiv(m, 128) * 128


def _tab_rows(r):
    return r + 8


def _vmem_plan(m, h, i, num_experts, shared_inter, info, spec):
    """(need, limit) or None, as kernel._vmem_plan, for this kernel's
    buffers."""
    r = R_SLAB
    t = functools.partial(all_reduce.tiled_bytes, info=info)
    e_pad = pl.cdiv(num_experts, info.num_lanes) * info.num_lanes
    f32, bf16 = jnp.float32, jnp.bfloat16
    lanes = info.num_lanes
    fixed = (
        2 * t((m, h), bf16)  # x in, out
        + t((m, e_pad), f32)  # scores
        + t((1, e_pad), f32)  # bias
        + t((h, 2 * shared_inter), FP8)
        + t((1, 2 * shared_inter), f32)
        + t((shared_inter, h), FP8)
        + t((1, h), f32)
        + t((m, h), FP8)  # sxq (per row, shared expert)
        + t((m, h), f32)  # xf
        + t((m, lanes), f32)  # xsf
        + t((m, h), f32)  # owner
        + t((m, h), bf16)  # shared (bf16)
        + t((m, e_pad), f32)  # wtab
        + t((4, e_pad), jnp.int32)  # hits_v
        + t((_tab_rows(r), e_pad), jnp.int32)  # slot table (vmem side)
        + t((r, h), f32)  # slab x
        + t((r, lanes), f32)  # slab scales
        + t((r, e_pad), f32)  # slab routing rows
        + 2 * t((r, h), f32)  # y, ycomp
        + t((r, lanes), f32)  # row scales of y
    )
    slot = t((h, 2 * i), FP8) + t((i, h), FP8) + t((1, 2 * i), f32) + t((1, h), f32)
    need = fixed + kernel._NUM_BUFFERS * slot + all_reduce.vmem_bytes(spec, info)
    limit = int(need * _vmem_headroom())
    if limit > int(info.vmem_capacity_bytes * kernel._vmem_fraction()):
        return None
    return need, min(limit, info.vmem_capacity_bytes)


def _vmem_headroom():
    """TPU_MOE_TP_DECODE_RG_VMEM_HEADROOM: the factor on the estimated VMEM need
    that admission and the compile limit use."""
    from vllm_torchtpu import envs

    return float(envs.TPU_MOE_TP_DECODE_RG_VMEM_HEADROOM)


def can_use_moe_tp_decode_rg(
    x,
    scores,
    bias,
    w13,
    w2,
    s13,
    s2,
    shared_w13,
    shared_s13,
    shared_w2,
    shared_s2,
    *,
    topk: int,
    n_ranks: int,
) -> bool:
    """kernel.can_use_moe_tp_decode's checks (this kernel needs more VMEM, so
    an incumbent refusal is ours too), then this kernel's VMEM plan."""
    weights = (w13, w2, s13, s2, shared_w13, shared_s13, shared_w2, shared_s2)
    if not kernel.can_use_moe_tp_decode(
        x, scores, bias, *weights, topk=topk, n_ranks=n_ranks
    ):
        return False
    m, h = x.shape
    i = w2.shape[1]
    info = pltpu.get_tpu_info()
    if m % 8 or h // kernel._Q_BLOCK > info.num_lanes:
        return False
    if not all_reduce.supported(n_ranks, m, h, info):
        return False
    spec = all_reduce.make_spec(n_ranks, m, h)
    return (
        _vmem_plan(
            m, h, kernel._i_chunk(i), scores.shape[1], shared_w2.shape[0], info, spec
        )
        is not None
    )


def kernel_name(experts, m, h, i, shared_inter, n_ranks, num_experts) -> str:
    name = (
        f"moe_tp_decode-rg{R_SLAB}-e_{experts}-m_{m}-h_{h}-i_{i}-fs_{shared_inter}"
        f"-nb_{kernel._NUM_BUFFERS}-n_{n_ranks}"
    )
    if kernel._split_meta():
        name += "-sm"
    if num_experts != experts:
        name += f"-ne_{num_experts}"
    return name


def _split_entry(n_valid_ref, rank_ref, e0_ref, *refs, **kwargs):
    return _kernel(kernel._Meta((n_valid_ref, rank_ref, e0_ref)), *refs, **kwargs)


def _quantize_rows(x, xf_ref, xsf_ref):
    """kernel._quantize_blocks, stored as f32 values (exact) and f32 block
    scales in lanes of xsf_ref."""
    m, h = x.shape
    for b, k in enumerate(range(0, h, kernel._Q_BLOCK)):
        block = x[:, k : k + kernel._Q_BLOCK]
        scale = jnp.max(jnp.abs(block), axis=1, keepdims=True) / FP8_MAX
        inv = jnp.where(scale == 0, 0, 1 / scale)
        xf_ref[:, k : k + kernel._Q_BLOCK] = (
            (block * inv).astype(FP8).astype(jnp.float32)
        )
        xsf_ref[:, b : b + 1] = scale.astype(jnp.float32)


def _shared_expert_bf16(xq_ref, xs, sw13_ref, ss13_ref, sw2_ref, ss2_ref, out_ref):
    """kernel._shared_expert, stored rounded to bf16 (the combine's first
    rounding)."""
    shared_inter = sw2_ref.shape[0]
    gu = jnp.dot(xq_ref[...], sw13_ref[...], preferred_element_type=jnp.float32)
    gu = (gu * xs * ss13_ref[...]).astype(jnp.bfloat16).astype(jnp.float32)
    gate, up = gu[:, :shared_inter], gu[:, shared_inter:]
    act = jax.nn.silu(gate).astype(jnp.bfloat16).astype(jnp.float32) * up
    act_q, act_scale = quantize_block(
        act.astype(jnp.bfloat16), axis=1, target_dtype=FP8
    )
    y = jnp.dot(act_q, sw2_ref[...], preferred_element_type=jnp.float32)
    out_ref[...] = (y * act_scale * ss2_ref[...]).astype(jnp.bfloat16)


def _kernel(
    meta_ref,
    x_ref,
    scores_ref,
    bias_ref,
    w1_hbm,
    w2_hbm,
    s1_hbm,
    s2_hbm,
    sw13_hbm,
    ss13_hbm,
    sw2_hbm,
    ss2_hbm,
    out_ref,
    w1_ref,
    w2_ref,
    s1_ref,
    s2_ref,
    sw13_ref,
    ss13_ref,
    sw2_ref,
    ss2_ref,
    sxq_ref,
    xf_ref,
    xsf_ref,
    owner_ref,
    shared_ref,
    wtab_ref,
    hits_v_ref,
    hits_s_ref,
    tab_v_ref,
    tab_s_ref,
    xg_ref,
    xsg_ref,
    wg_ref,
    y_ref,
    yc_ref,
    msr_ref,
    ring_sem,
    shared_sem,
    hit_sem,
    tab_sem,
    *ar_scr,
    local_experts,
    n_chunks,
    topk,
    routed_scale,
    ar_spec,
):
    all_reduce.prologue(ar_spec)
    m, h = x_ref.shape
    i = w2_hbm.shape[1]
    ic = w2_ref.shape[1]
    nc = n_chunks
    nb = kernel._NUM_BUFFERS
    rs = R_SLAB
    n_valid = meta_ref[0]
    rank = meta_ref[1]
    e0 = meta_ref[2]
    e_pad = wtab_ref.shape[1]
    f32, bf16 = jnp.float32, jnp.bfloat16

    shared_copies = (
        pltpu.make_async_copy(sw13_hbm, sw13_ref, shared_sem.at[0]),
        pltpu.make_async_copy(ss13_hbm, ss13_ref, shared_sem.at[1]),
        pltpu.make_async_copy(sw2_hbm, sw2_ref, shared_sem.at[2]),
        pltpu.make_async_copy(ss2_hbm, ss2_ref, shared_sem.at[3]),
    )
    for copy in shared_copies:
        copy.start()

    def copy_hits():
        hit_copy = pltpu.make_async_copy(hits_v_ref, hits_s_ref, hit_sem.at[0])
        hit_copy.start()
        hit_copy.wait()

    def store_active():
        ids, n_active = kernel._compact_vector(wtab_ref[...], e0, local_experts)
        hits_v_ref[2:3, 0 : ids.shape[1]] = ids.astype(jnp.int32)
        hits_v_ref[3:4, :] = jnp.broadcast_to(
            n_active, (1, hits_v_ref.shape[1])
        ).astype(jnp.int32)

    # Routing: the incumbent's fast schedule, unchanged.
    wtab, flag = kernel._route_fast(scores_ref[...], bias_ref[...], n_valid, topk=topk)
    wtab_ref[...] = wtab
    store_active()
    hits_v_ref[1:2, :] = jnp.broadcast_to(flag, (1, hits_v_ref.shape[1])).astype(
        jnp.int32
    )
    copy_hits()

    @pl.when(hits_s_ref[1, 0] != 0)
    def _exact_routing():
        wtab_ref[...] = kernel._route(
            scores_ref[...], bias_ref[...], n_valid, topk=topk
        )
        store_active()
        copy_hits()

    def active(j):
        return hits_s_ref[2, j]

    def chunk_transfers(step):
        e = active(lax.div(step, nc))
        c0 = pl.multiple_of(lax.rem(step, nc) * ic, ic)
        slot = step % nb
        sem = ring_sem.at[slot]
        lo, hi = pl.ds(0, ic), pl.ds(ic, ic)
        gate, up = pl.ds(c0, ic), pl.ds(i + c0, ic)
        return [
            pltpu.make_async_copy(
                w1_hbm.at[e, :, gate], w1_ref.at[slot, :, lo], sem.at[0]
            ),
            pltpu.make_async_copy(
                w1_hbm.at[e, :, up], w1_ref.at[slot, :, hi], sem.at[1]
            ),
            pltpu.make_async_copy(w2_hbm.at[e, gate], w2_ref.at[slot], sem.at[2]),
            pltpu.make_async_copy(
                s1_hbm.at[e, 0, :, gate], s1_ref.at[slot, :, lo], sem.at[3]
            ),
            pltpu.make_async_copy(
                s1_hbm.at[e, 0, :, up], s1_ref.at[slot, :, hi], sem.at[4]
            ),
            pltpu.make_async_copy(s2_hbm.at[e, 0], s2_ref.at[slot], sem.at[5]),
        ]

    def transfers(step):
        if nc > 1:
            return chunk_transfers(step)
        e, slot = active(step), step % nb
        return [
            pltpu.make_async_copy(w1_hbm.at[e], w1_ref.at[slot], ring_sem.at[slot, 0]),
            pltpu.make_async_copy(w2_hbm.at[e], w2_ref.at[slot], ring_sem.at[slot, 1]),
            pltpu.make_async_copy(
                s1_hbm.at[e, 0], s1_ref.at[slot], ring_sem.at[slot, 2]
            ),
            pltpu.make_async_copy(
                s2_hbm.at[e, 0], s2_ref.at[slot], ring_sem.at[slot, 3]
            ),
        ]

    count = hits_s_ref[3, 0]
    items = count * nc

    def prime_fast(item, carry):
        for copy in transfers(item):
            copy.start()
        return carry

    lax.fori_loop(0, jnp.minimum(items, nb), prime_fast, 0)

    # Slot tables (overlap the first ring transfers). Row j < rs of the table:
    # the row in slot j of every expert column (m where the expert has <= j
    # rows); row rs: the counts.
    hit = (wtab_ref[...] != 0).astype(f32)  # [m, e_pad]
    mp = _pad_rows(m)
    hit_p = (
        jnp.concatenate([hit, jnp.zeros((mp - m, e_pad), f32)], axis=0)
        if mp != m
        else hit
    )
    rr = lax.broadcasted_iota(jnp.int32, (m, mp), 0)
    cc = lax.broadcasted_iota(jnp.int32, (m, mp), 1)
    lower = jnp.where(cc < rr, 1.0, 0.0).astype(f32)
    rank_tab = jnp.dot(
        lower, hit_p, preferred_element_type=f32
    )  # [m, e_pad], exact small integers
    n_e = jnp.sum(hit, axis=0, keepdims=True)  # [1, e_pad]
    rows_f = lax.broadcasted_iota(jnp.int32, (m, e_pad), 0).astype(f32)
    for j in range(rs):
        sel = jnp.logical_and(hit != 0, rank_tab == float(j))
        rj = jnp.sum(jnp.where(sel, rows_f, 0.0), axis=0, keepdims=True)
        rj = jnp.where(n_e > j, rj, float(m))
        tab_v_ref[j : j + 1, :] = rj.astype(jnp.int32)
    tab_v_ref[rs : rs + 1, :] = n_e.astype(jnp.int32)
    tab_copy = pltpu.make_async_copy(tab_v_ref, tab_s_ref, tab_sem.at[0])
    tab_copy.start()

    for copy in shared_copies:
        copy.wait()

    owner_ref[...] = jnp.zeros((m, h), f32)
    xg_ref[...] = jnp.zeros(xg_ref.shape, f32)
    xsg_ref[...] = jnp.zeros(xsg_ref.shape, f32)
    wg_ref[...] = jnp.zeros(wg_ref.shape, f32)
    sxq, sxs = quantize_block(x_ref[...], axis=1, target_dtype=FP8)
    sxq_ref[...] = sxq
    _shared_expert_bf16(sxq_ref, sxs, sw13_ref, ss13_ref, sw2_ref, ss2_ref, shared_ref)
    _quantize_rows(x_ref[...], xf_ref, xsf_ref)
    tab_copy.wait()

    expert_cols = lax.broadcasted_iota(jnp.int32, (rs, e_pad), 1)
    dense_cols = lax.broadcasted_iota(jnp.int32, (m, e_pad), 1)
    lanes = 2 * ic

    def col_of(e):
        return e0 + e

    def n_rows(e):
        return tab_s_ref[rs, col_of(e)]

    def slab_rows(e):
        return jnp.minimum(n_rows(e), rs)

    def overflow(e):
        return n_rows(e) > rs

    def gather(e, do):
        """Expert e's routed rows (ascending) into the slab: x values, block
        scales, routing-table rows."""
        cnt = jnp.where(do, slab_rows(e), 0)
        g = col_of(e)

        def body(j, c):
            r = tab_s_ref[j, g]
            xg_ref[pl.ds(j, 1), :] = xf_ref[pl.ds(r, 1), :]
            xsg_ref[pl.ds(j, 1), :] = xsf_ref[pl.ds(r, 1), :]
            wg_ref[pl.ds(j, 1), :] = wtab_ref[pl.ds(r, 1), :]
            return c

        lax.fori_loop(0, cnt, body, 0)

    def gate_up_rows(xv_ref, sv, slot):
        """gmm_v2's per-block chain on the rows of xv_ref [rows, h] (FP8-valued
        f32), block scales sv [rows, lanes]."""
        rows = xv_ref.shape[0]
        w_scale = s1_ref[slot].astype(bf16)
        acc = jnp.zeros((rows, lanes), bf16)
        for b, k in enumerate(range(0, h, kernel._Q_BLOCK)):
            part = jnp.matmul(
                xv_ref[:, k : k + kernel._Q_BLOCK].astype(FP8),
                w1_ref[slot, k : k + kernel._Q_BLOCK, :],
                preferred_element_type=f32,
            ).astype(bf16)
            part *= jnp.broadcast_to(sv[:, b : b + 1], (rows, lanes)).astype(bf16)
            part *= w_scale
            acc += part
        return acc

    def gate_up_slab(slot):
        return gate_up_rows(xg_ref, xsg_ref[...], slot)

    def coeff_slab(e):
        return jnp.sum(
            jnp.where(expert_cols == e0 + e, wg_ref[...], 0.0), axis=1, keepdims=True
        )

    def mid_q(acc1, coeff):
        act = jax.nn.silu(acc1[:, :ic]) * acc1[:, ic:]
        mid = jnp.where(coeff != 0, act.astype(f32) * coeff, 0.0)
        ms = jnp.max(jnp.abs(mid), axis=1, keepdims=True) / FP8_MAX
        mq = (mid * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(FP8)
        return mq, ms

    def finish_dense(step, slot):
        """The incumbent's finish over all m rows (overflow experts)."""
        e = active(lax.div(step, nc))
        acc1 = gate_up_rows(xf_ref, xsf_ref[...], slot)
        coeff = jnp.sum(
            jnp.where(dense_cols == e0 + e, wtab_ref[...], 0.0), axis=1, keepdims=True
        )
        mq, ms = mid_q(acc1, coeff)
        for n in range(0, h, kernel._N_CHUNK):
            acc2 = jnp.matmul(
                mq, w2_ref[slot, :, n : n + kernel._N_CHUNK], preferred_element_type=f32
            )
            owner_ref[:, n : n + kernel._N_CHUNK] += (
                acc2 * s2_ref[slot, :, n : n + kernel._N_CHUNK] * ms
            )

    def finish_slab(step, slot, acc1, coeff, live):
        e = active(lax.div(step, nc))
        g = col_of(e)
        cnt = jnp.where(live, slab_rows(e), 0)
        mq, ms = mid_q(acc1, coeff)
        if nc == 1:
            # y = acc2 * s2 per row; the row scale is applied in the owner
            # update, so each owner element is updated by the incumbent's
            # expression owner + acc2 * s2 * ms (same contraction freedom, same
            # bits).
            for n in range(0, h, kernel._N_CHUNK):
                acc2 = jnp.matmul(
                    mq,
                    w2_ref[slot, :, n : n + kernel._N_CHUNK],
                    preferred_element_type=f32,
                )
                y_ref[:, n : n + kernel._N_CHUNK] = (
                    acc2 * s2_ref[slot, :, n : n + kernel._N_CHUNK]
                )
            msr_ref[...] = jnp.broadcast_to(ms, msr_ref.shape)

            def scatter(j, c):
                r = tab_s_ref[j, g]
                owner_ref[pl.ds(r, 1), :] = (
                    owner_ref[pl.ds(r, 1), :]
                    + y_ref[pl.ds(j, 1), :] * msr_ref[pl.ds(j, 1), 0:1]
                )
                return c

            lax.fori_loop(0, cnt, scatter, 0)
            return
        first = lax.rem(step, nc) == 0
        last = lax.rem(step, nc) == nc - 1
        init_cnt = jnp.where(first, cnt, 0)

        def init(j, c):
            yc_ref[pl.ds(j, 1), :] = owner_ref[pl.ds(tab_s_ref[j, g], 1), :]
            return c

        lax.fori_loop(0, init_cnt, init, 0)
        for n in range(0, h, kernel._N_CHUNK):
            acc2 = jnp.matmul(
                mq, w2_ref[slot, :, n : n + kernel._N_CHUNK], preferred_element_type=f32
            )
            yc_ref[:, n : n + kernel._N_CHUNK] += (
                acc2 * s2_ref[slot, :, n : n + kernel._N_CHUNK] * ms
            )

        def put(j, c):
            r = tab_s_ref[j, g]
            owner_ref[pl.ds(r, 1), :] = yc_ref[pl.ds(j, 1), :]
            return c

        lax.fori_loop(0, jnp.where(last, cnt, 0), put, 0)

    def finish(step, slot, acc1, coeff):
        ovf = overflow(active(lax.div(step, nc)))

        def dense_once(_, c):
            finish_dense(step, slot)
            return c

        # The overflow finish runs after the slab finish's write-back loop, an
        # existing block boundary, so it does not split the pipelined
        # gate_up(next) / slab finish block. Item order, hence the output bits,
        # unchanged.
        finish_slab(step, slot, acc1, coeff, live=jnp.logical_not(ovf))
        lax.fori_loop(0, ovf.astype(jnp.int32), dense_once, 0)

    def is_new_expert(step):
        return lax.rem(step, nc) == 0 if nc > 1 else jnp.bool_(True)

    def ring_step(step, carry, refill):
        acc1, coeff = carry
        nxt = step + 1
        for copy in transfers(nxt):
            copy.wait()
        # This item's coefficients are taken from the slab before the next
        # expert's rows replace it.
        e_nxt = active(lax.div(nxt, nc))
        gather(e_nxt, is_new_expert(nxt))
        acc1_next = gate_up_slab(nxt % nb)
        coeff_next = coeff_slab(e_nxt)
        finish(step, step % nb, acc1, coeff)
        if refill:
            for copy in transfers(step + nb):
                copy.start()
        return acc1_next, coeff_next

    @pl.when(items > 0)
    def run_experts():
        for copy in transfers(jnp.int32(0)):
            copy.wait()
        e_first = active(0)
        gather(e_first, jnp.bool_(True))
        carry = (gate_up_slab(0), coeff_slab(e_first))
        n_refill = jnp.maximum(items - nb, 0)
        carry = lax.fori_loop(0, n_refill, lambda s, c: ring_step(s, c, True), carry)
        carry = lax.fori_loop(
            n_refill, items - 1, lambda s, c: ring_step(s, c, False), carry
        )
        finish(items - 1, (items - 1) % nb, *carry)

    all_reduce.epilogue(
        ar_spec,
        rank,
        n_valid,
        kernel._combine(owner_ref[...], shared_ref[...].astype(f32), routed_scale),
        out_ref,
        ar_scr,
    )


@functools.partial(
    jax.jit,
    static_argnames=("routed_scale", "topk", "n_ranks", "hbm_operands"),
)
def moe_tp_decode_rg(
    x,
    scores,
    bias,
    n_valid,
    rank,
    w13,
    w2,
    s13,
    s2,
    shared_w13,
    shared_s13,
    shared_w2,
    shared_s2,
    experts_start=0,
    *,
    routed_scale: float,
    topk: int,
    n_ranks: int,
    hbm_operands: bool = False,
):
    """kernel.moe_tp_decode's layer output at its fast schedule's bits.

    Same arguments as kernel.moe_tp_decode without `fast` (this kernel only has
    the fast schedule). Each streamed expert multiplies only the rows routed to
    it, gathered into an R_SLAB-row slab; see the module docstring. Check
    can_use_moe_tp_decode_rg first: a plan that does not fit VMEM raises.
    """
    if not kernel.exact_in_bf16(routed_scale):
        raise ValueError(f"routed_scale {routed_scale!r} is not exact in bf16")
    m, h = x.shape
    experts, i, _ = w2.shape
    num_experts = scores.shape[1]
    shared_inter = shared_w2.shape[0]
    info = pltpu.get_tpu_info()
    ar_spec = all_reduce.make_spec(n_ranks, m, h)
    ic = kernel._i_chunk(i)
    plan = _vmem_plan(m, h, ic, num_experts, shared_inter, info, ar_spec)
    if plan is None:
        raise ValueError(
            "moe_tp_decode_rg does not fit VMEM; check can_use_moe_tp_decode_rg"
        )
    _, limit = plan
    nb = kernel._NUM_BUFFERS
    e_pad = pl.cdiv(num_experts, info.num_lanes) * info.num_lanes
    lanes = info.num_lanes
    scores = scores.astype(jnp.float32)
    bias = bias.astype(jnp.float32)
    if e_pad != num_experts:
        scores = jnp.pad(scores, ((0, 0), (0, e_pad - num_experts)))
        bias = jnp.pad(bias, (0, e_pad - num_experts), constant_values=-jnp.inf)
    split = kernel._split_meta()
    if split:
        metas = tuple(
            jnp.asarray(v, jnp.int32).reshape(1) for v in (n_valid, rank, experts_start)
        )
    else:
        metas = (
            jnp.stack(
                [
                    jnp.asarray(n_valid, jnp.int32).reshape(()),
                    jnp.asarray(rank, jnp.int32).reshape(()),
                    jnp.asarray(experts_start, jnp.int32).reshape(()),
                ]
            ),
        )
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    weights = (
        w13,
        w2,
        s13,
        s2,
        shared_w13,
        shared_s13.reshape(1, 2 * shared_inter),
        shared_w2,
        shared_s2.reshape(1, h),
    )
    if hbm_operands:
        weights = tuple(
            pltpu.with_memory_space_constraint(w, pltpu.HBM) for w in weights
        )
    tab_rows = _tab_rows(R_SLAB)
    return pl.pallas_call(
        functools.partial(
            _split_entry if split else _kernel,
            local_experts=experts,
            n_chunks=i // ic,
            topk=topk,
            routed_scale=float(routed_scale),
            ar_spec=ar_spec,
        ),
        out_shape=jax.ShapeDtypeStruct((m, h), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(metas),
            in_specs=[vmem, vmem, vmem, *[hbm] * 8],
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((nb, h, 2 * ic), kernel.FP8),
                pltpu.VMEM((nb, ic, h), kernel.FP8),
                pltpu.VMEM((nb, 1, 2 * ic), jnp.float32),
                pltpu.VMEM((nb, 1, h), jnp.float32),
                pltpu.VMEM((h, 2 * shared_inter), kernel.FP8),
                pltpu.VMEM((1, 2 * shared_inter), jnp.float32),
                pltpu.VMEM((shared_inter, h), kernel.FP8),
                pltpu.VMEM((1, h), jnp.float32),
                pltpu.VMEM((m, h), kernel.FP8),  # sxq
                pltpu.VMEM((m, h), jnp.float32),  # xf
                pltpu.VMEM((m, lanes), jnp.float32),  # xsf
                pltpu.VMEM((m, h), jnp.float32),  # owner
                pltpu.VMEM((m, h), jnp.bfloat16),  # shared
                pltpu.VMEM((m, e_pad), jnp.float32),  # wtab
                pltpu.VMEM((4, e_pad), jnp.int32),
                pltpu.SMEM((4, e_pad), jnp.int32),
                pltpu.VMEM((tab_rows, e_pad), jnp.int32),
                pltpu.SMEM((tab_rows, e_pad), jnp.int32),
                pltpu.VMEM((R_SLAB, h), jnp.float32),  # xg
                pltpu.VMEM((R_SLAB, lanes), jnp.float32),  # xsg
                pltpu.VMEM((R_SLAB, e_pad), jnp.float32),  # wg
                pltpu.VMEM((R_SLAB, h), jnp.float32),  # y
                pltpu.VMEM((R_SLAB, h), jnp.float32),  # ycomp
                pltpu.VMEM((R_SLAB, lanes), jnp.float32),  # row scales of y (TP)
                pltpu.SemaphoreType.DMA((nb, 4 if i == ic else 6)),
                pltpu.SemaphoreType.DMA((4,)),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.SemaphoreType.DMA((1,)),
                *all_reduce.scratch_shapes(ar_spec),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=limit, collective_id=all_reduce.COLLECTIVE_ID
        ),
        name=kernel_name(experts, m, h, i, shared_inter, n_ranks, num_experts),
    )(*metas, x, scores, bias.reshape(1, e_pad), *weights)
