# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE decode kernel: routed experts, shared expert and all-reduce in one call.

Under MoE tensor parallelism every rank holds all experts with the
intermediate dimension sharded, so each expert's local slice is small (for
Qwen3.5-397B at TP8: i = 128, 1.5 MiB of FP8 per expert) and rows per expert
at decode are ~1-3. The GMM path pays a fixed cost per active expert in each
of its two kernels plus a permute/sort/combine around them.

This kernel visits each active expert once. Its w13/w2 slices stream from HBM
through a ring of VMEM slots while the previous expert computes on ALL tokens
in their original order; the route weight zeroes rows the expert does not
own. The first matmul of expert j+1 is issued in step j, ahead
of expert j's elementwise chain and second matmul, so the MXU stays busy while
the VPU quantizes. Nothing is permuted and no intermediate goes to HBM.
Experts too large for three whole-expert slots (GLM-5.2 at EP16: 37.7 MB of
FP8 each) stream instead in blocks of the intermediate dimension, each block a
ring unit (blocked mode, plan_weight_ring).

The same call also runs the layer's shared expert slice and the TP
all-reduce, so one kernel replaces the whole MoE block after the router
(served by layers/adapter/fused_moe_decode.py). With expert parallelism and
TP attention a rank holds whole experts instead of slices and the call is the
same, restricted to the rank's expert range.

Numerics: hidden states are quantized to FP8 once per call with one scale per
row per 512 columns, as gmm_v2 quantizes them with the kernel off; the
activated intermediate likewise per row per 512 columns of the expert (per row
per block in blocked mode). Both matmuls are FP8 x FP8; each 512-column block's
partial is scaled and summed in FP32, into an FP32 VMEM accumulator.
Router top-k stays with the caller.

Derived from moe_dense_expert.py, which targets i % 512 == 0 (EP layouts).
"""

import contextlib
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Output columns per second-matmul step. Each step computes an f32 block of
# [tokens, _N_CHUNK] and adds it into the accumulator, so this bounds the live
# f32 result. It is a compile-time constant: keep it a multiple of 128 (the
# lane width) that divides the hidden size. Smaller lowers register and VMEM
# pressure at large token counts and issues more, shorter matmuls.
_N_CHUNK = 512
# Upper bound on the weight ring depth. The depth one call uses is picked by
# _weight_ring_depth: the most slots, at most this and at least 3, whose expert
# weights fit in half of VMEM. One slot holds the expert whose first matmul is
# in flight, so depth - 2 weight transfers run ahead of compute. For
# Qwen3.5-397B at 8 ranks this bound sets the depth with TP experts (6 slots of
# 1.5 MiB); with EP experts (12 MiB per slot) VMEM caps it at 3. A deeper ring
# hides more DMA latency when experts are small, but the VMEM it takes lowers
# the token cap vmem_bytes allows.
_NUM_BUFFERS = 6
# Columns per FP8 activation scale: each row of the hidden states (and of the
# activated intermediate) gets one scale per block of this many columns, the
# block gmm_v2 uses on the GMM path, so one outlier channel coarsens only its
# own block. Each block's partial product is scaled before it is summed.
_Q_BLOCK = 512
# q38 DEC-FB-2 lever 5b, routed-row compaction (compact_rows): table row of the
# per-expert row counts in the scalar tables; the token rows of expert slot
# p = e * R + r sit at [p // 128, p % 128] before it.
_COMPACT_LANES = 128


# The expert loop, plus the layer's shared expert (its TP slice, gated by
# sigmoid(x . w_g) or ungated) and the TP all-reduce, in one call per rank under
# shard_map. The all-reduce is latency bound at decode sizes, so it runs once
# at the end, as a direct all-to-all with a local sum (reduce-scatter
# semantics, see all_to_all_sum) followed by an all-gather; the barrier it
# needs is signalled at entry and waited only before the all-to-all, so the
# expert loop hides it. bf16 on the wire, FP32 sum of the slots.


def _kernel(
    active_ref,
    rank_ref,
    x_ref,
    coeff_ref,
    shared_gate_ref,
    w1_hbm,
    w2_hbm,
    w1_scale_hbm,
    w2_scale_hbm,
    shared_w1_hbm,
    shared_w1_scale_hbm,
    shared_w2_hbm,
    shared_w2_scale_hbm,
    out_ref,
    w1_ring,
    w2_ring,
    w1_scale_ring,
    w2_scale_ring,
    shared_w1_vmem,
    shared_w1_scale_vmem,
    shared_w2_vmem,
    shared_w2_scale_vmem,
    xq_ref,
    xs_ref,
    acc_ref,
    acc_bf16,
    recv,
    red,
    ids_ref,
    sem_ref,
    shared_sem,
    a2a_send,
    a2a_recv,
    ag_send,
    ag_recv,
    *compact_scratch,
    m,
    h,
    i,
    experts,
    nb,
    n_ranks,
    weight_dtype,
    rows=None,
    ib=None,
    folded=False,
    compact_r=None,
    scopes=0,
    compact_pipe=False,
    exact_width=False,
):
    """One rank's MoE block. Shapes use m tokens (padded to 16), h hidden, i
    and i_s the routed and shared intermediate sizes of this rank's slice, E
    local experts, nb ring slots, n ranks and c = h // n.

    Scalar prefetch (SMEM):
      active_ref [E] int32: 1 where at least one token routes to local expert e.
      rank_ref [1] int32: this rank's mesh index.
    Inputs in VMEM:
      x_ref [m, h] bf16: hidden states.
      coeff_ref [m, E padded to 128] f32: route weight of token t for local
        expert e, 0 where t does not route to e.
      shared_gate_ref [m, 128] f32: sigmoid(x . w_g) per token, repeated over
        lanes; 1 for an ungated shared expert.
    Inputs in HBM (weights in `weight_dtype`, scales f32 per output channel):
      w1_hbm [E, h, 2 i] and w1_scale_hbm [E, 1, 1, 2 i]: gate and up projections.
      w2_hbm [E, i, h] and w2_scale_hbm [E, 1, 1, h]: down projection.
      shared_w1_hbm [h, 2 i_s], shared_w1_scale_hbm [1, 2 i_s],
      shared_w2_hbm [i_s, h], shared_w2_scale_hbm [1, h]: the shared expert.
    Output:
      out_ref [m, h] bf16: shared + routed output summed over the n ranks.
        With `rows` set (DP attention), [r_pad, h]: only this rank's `rows`
        token rows of that sum, r_pad = rows padded to the sublane tile.
    Scratch:
      w1_ring [nb, h, 2 i], w2_ring [nb, i, h], w1_scale_ring [nb, 1, 2 i],
        w2_scale_ring [nb, 1, h]: weight ring, one expert per slot. In
        blocked mode one unit per slot, with ib in place of i.
      shared_w1_vmem, shared_w1_scale_vmem, shared_w2_vmem,
        shared_w2_scale_vmem: the shared expert, loaded once.
      xq_ref [m, h] weight_dtype: hidden states quantized per row per
        _Q_BLOCK columns.
      xs_ref [h / _Q_BLOCK, m, 128] f32: their scales, one [m, 128] plane per
        column block, repeated over lanes.
      acc_ref [m, h] f32: this rank's partial sum.
      acc_bf16 [m, h] bf16: the same, as the all-reduce source.
      recv [n, m, c] bf16: all-to-all receive slots, one per sending rank.
      red [m, c] bf16: this rank's reduced column chunk.
      With `rows` set, acc_bf16 and recv are [n, r_pad, h] (index d holds
      peer d's tokens, the send slice for peer d) and red is unused.
      ids_ref [E] int32 (SMEM): active expert ids, compacted.
      sem_ref DMA [nb, 4]: ring slot transfers (w1, w2, w1 scale, w2 scale);
        [nb, 6] in blocked mode (gate and up blocks of w1 and w1 scale apart).

    ib (static): None streams whole experts, one ring slot per expert; an int
    streams each expert as i // ib units of ib intermediate columns (gate
    block | matching up block of w13, the matching rows of w2). Every block
    adds its down-projection partial into the same f32 accumulator, which is
    exact because SiLU(gate) * up is elementwise in the intermediate and the
    down projection sums over it.
      shared_sem DMA [4]: shared expert transfers.
      a2a_send DMA [n], a2a_recv DMA: all-to-all.
      ag_send DMA [n], ag_recv DMA: all-gather.

    compact_r (static, q38 DEC-FB-2 lever 5b): None computes every active expert
    on all m rows (route weight zeroes the rows it does not own). An int R
    computes each expert on its own routed rows only, R at most: the prologue
    gathers every local expert's rows into an expert-sorted [E * R, h] buffer
    with one 0/1 matmul (exact: one nonzero term per element), the expert loop
    runs both matmuls on that expert's R rows, and the per-row results are
    added into acc_ref row by row. Per-row arithmetic is the dense loop's
    (same K blocks, per-row quantisation), and rows an expert does not own are
    not touched (the dense loop adds +0.0 to them; see the closing pass). An
    expert with more than R rows runs the dense body. compact_scratch then
    holds gsel [E*R, m] bf16, xsort [E*R, h] bf16, csort [E*R, 128] f32,
    tabv [T, 128] i32 (VMEM), tabs [T, 128] i32 (SMEM), contrib [R, h] f32
    and a DMA semaphore.

    compact_pipe (static, q38 L3 D60, form 5c "pipelined loop"; needs
    compact_r): the same per-row arithmetic as compact_r (so bitwise the dense
    loop), scheduled for the weight stream. (a) The routed weight ring is
    primed BEFORE the compaction prologue, so its first nb transfers stream
    while the tables are built (compact_r alone primes after them). (b) The
    tables are slot-major: tabv / tabs [Rp, 128] i32 hold the token of slot r
    of local expert e at [r, e] (r < R) and expert e's row count at [R, e];
    ctab [Rp, 128] f32 holds that slot's route weight; each is one pass over
    the [m, 128] route plane per slot r instead of one unrolled pass per
    expert. The gather matrix gsel [E*R, m] and the expert-sorted rows xsort
    [E*R, h] are as compact_r's. (c) The expert loop is software-pipelined as
    the dense loop is: step j waits expert j+1's weights and issues its
    gathered-row first matmul in the same block as expert j's elementwise
    chain and second matmul, carrying it in acc1buf [2, R, 2 i] f32; the row
    adds follow. An expert with more than R rows runs the dense body (its own
    first matmul over all m rows) and still issues expert j+1's. compact_scratch
    then holds gsel, xsort, ctab, tabv, tabs, contrib, acc1buf and a DMA
    semaphore.

    exact_width (static, q38 PORT-BL MOE, ported from Benliang Wang's b0c1453ba;
    TP all-reduce output only, needs h // 128 >= n): the all-to-all and the
    all-gather split h's lane tiles over the ranks as evenly as whole tiles
    allow, the same columns in both halves, so no rank sends padding, and
    out_ref is [m, h] instead of [m, h padded to n lanes of 128]. Each column is
    still the f32 sum of the n bf16 slots in rank order, rounded to bf16 once,
    as in the padded split, so the values are bitwise the same.
    """
    nblk = 1 if ib is None else i // ib
    me = rank_ref[0, 0] if folded else rank_ref[0]
    qmax = float(jnp.finfo(weight_dtype).max)
    # Column chunk each rank reduces: h padded to n_ranks lanes of 128, so every
    # chunk is lane aligned (Qwen3.8-Flash: h 2560 over 8 ranks pads to 3072).
    hp = _padded_hidden(h, n_ranks)
    chunk = hp // n_ranks
    barrier = pltpu.get_barrier_semaphore()
    for d in range(n_ranks):
        pl.semaphore_signal(
            barrier,
            inc=1,
            device_id=(jnp.int32(d),),
            device_id_type=pl.DeviceIdType.MESH,
        )

    shared_copies = [
        pltpu.make_async_copy(shared_w1_hbm, shared_w1_vmem, shared_sem.at[0]),
        pltpu.make_async_copy(shared_w2_hbm, shared_w2_vmem, shared_sem.at[1]),
    ]
    # The folded call hands a BF16 shared expert's unused scales over as
    # stored (1-D), so they are not copied.
    if not (folded and shared_w1_vmem.dtype != weight_dtype):
        shared_copies += [
            pltpu.make_async_copy(
                shared_w1_scale_hbm, shared_w1_scale_vmem, shared_sem.at[2]
            ),
            pltpu.make_async_copy(
                shared_w2_scale_hbm, shared_w2_scale_vmem, shared_sem.at[3]
            ),
        ]
    for copy in shared_copies:
        copy.start()

    def sc(name):
        # q38 IQD #10 measurement: in-kernel phase markers (Mosaic trace scopes), visible on the "XLA TraceMe" line only
        # with LIBTPU_INIT_ARGS=--xla_enable_custom_call_region_trace=true. scopes=0 (the default) is the same program.
        return jax.named_scope(name) if scopes else contextlib.nullcontext()

    prologue_scope = sc("moe_prologue")
    prologue_scope.__enter__()
    acc_ref[...] = jnp.zeros((m, h), jnp.float32)
    # K blocks lead xs_ref so each block's scales are one contiguous [m, 128]
    # plane (as in moe_dense_expert.py); K between tokens and lanes relayouts.
    for k in range(0, h, _Q_BLOCK):
        x = x_ref[:, k : k + _Q_BLOCK].astype(jnp.float32)
        xsc = jnp.max(jnp.abs(x), axis=1, keepdims=True) / qmax
        xq_ref[:, k : k + _Q_BLOCK] = (x * jnp.where(xsc == 0, 0.0, 1.0 / xsc)).astype(
            weight_dtype
        )
        xs_ref[k // _Q_BLOCK] = jnp.broadcast_to(xsc, (m, 128))
    cols = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)

    def compact(e, count):
        @pl.when(active_ref[e] != 0)
        def record():
            ids_ref[count] = e

        return count + (active_ref[e] != 0).astype(jnp.int32)

    count = lax.fori_loop(0, experts, compact, jnp.int32(0))
    prologue_scope.__exit__(None, None, None)
    # Ring units: one per active expert, or nblk per active expert (blocked).
    units = count if ib is None else count * nblk

    if compact_r is not None and not compact_pipe:
        gsel_ref, xsort_ref, csort_ref, tabv_ref, tabs_ref, contrib_ref, tab_sem = (
            compact_scratch
        )
        R = compact_r
        cnt_row = (experts * R) // _COMPACT_LANES
        epad = coeff_ref.shape[1]
        coeff_all = coeff_ref[...]  # [m, epad] f32
        owned = coeff_all != 0
        # occ[t, e]: how many earlier rows expert e owns (exclusive running
        # count down the rows), as a strictly-lower-triangular 0/1 matmul:
        # 0/1 products, f32 sums of at most m ones, exact.
        ti_ = lax.broadcasted_iota(jnp.int32, (m, m), 0)
        tj_ = lax.broadcasted_iota(jnp.int32, (m, m), 1)
        lower = (tj_ < ti_).astype(jnp.bfloat16)
        occ = jnp.dot(
            lower, owned.astype(jnp.bfloat16), preferred_element_type=jnp.float32
        ).astype(jnp.int32)
        cnt = jnp.sum(owned.astype(jnp.int32), axis=0, keepdims=True)  # [1, epad]
        occ_t = jnp.transpose(occ)  # [epad, m]
        own_t = jnp.transpose(owned.astype(jnp.int32))  # [epad, m]
        coeff_t = jnp.transpose(coeff_all)  # [epad, m]
        slot_sub = lax.broadcasted_iota(jnp.int32, (R, m), 0)
        tok_sub = lax.broadcasted_iota(jnp.int32, (m, R), 0)
        slot_lane = lax.broadcasted_iota(jnp.int32, (m, R), 1)
        tabv_ref[...] = jnp.zeros(tabv_ref.shape, jnp.int32)
        for e in range(experts):
            # rows of expert e: slot r holds the token t with occ[t, e] == r
            sel = (own_t[e : e + 1, :] != 0) & (occ_t[e : e + 1, :] == slot_sub)
            gsel_ref[e * R : (e + 1) * R, :] = sel.astype(jnp.bfloat16)
            cw = jnp.sum(
                jnp.where(sel, coeff_t[e : e + 1, :], 0.0), axis=1, keepdims=True
            )
            csort_ref[e * R : (e + 1) * R, :] = jnp.broadcast_to(cw, (R, 128))
            col = (owned[:, e : e + 1]) & (occ[:, e : e + 1] == slot_lane)  # [m, R]
            tok = jnp.sum(jnp.where(col, tok_sub, 0), axis=0, keepdims=True)  # [1, R]
            p0 = e * R
            tabv_ref[
                p0 // _COMPACT_LANES : p0 // _COMPACT_LANES + 1,
                p0 % _COMPACT_LANES : p0 % _COMPACT_LANES + R,
            ] = tok
        tabv_ref[cnt_row : cnt_row + 1, :] = cnt[:, :_COMPACT_LANES]
        tab_copy = pltpu.make_async_copy(tabv_ref, tabs_ref, tab_sem)
        tab_copy.start()
        # every local expert's rows, expert-sorted, in one 0/1 gather matmul
        # per 512 columns (one nonzero term per element: the row, exactly)
        for n in range(0, h, _N_CHUNK):
            xsort_ref[:, n : n + _N_CHUNK] = jnp.dot(
                gsel_ref[...],
                x_ref[:, n : n + _N_CHUNK],
                preferred_element_type=jnp.float32,
            ).astype(xsort_ref.dtype)
        tab_copy.wait()

    def transfers(step):
        if ib is not None:
            return block_transfers(step)
        e, slot = ids_ref[step], step % nb
        return [
            pltpu.make_async_copy(w1_hbm.at[e], w1_ring.at[slot], sem_ref.at[slot, 0]),
            pltpu.make_async_copy(w2_hbm.at[e], w2_ring.at[slot], sem_ref.at[slot, 1]),
            pltpu.make_async_copy(
                w1_scale_hbm.at[e, 0], w1_scale_ring.at[slot], sem_ref.at[slot, 2]
            ),
            pltpu.make_async_copy(
                w2_scale_hbm.at[e, 0], w2_scale_ring.at[slot], sem_ref.at[slot, 3]
            ),
        ]

    def block_transfers(step):
        """Unit `step` = block step % nblk of active expert step // nblk."""
        assert ib is not None
        e, slot = ids_ref[step // nblk], step % nb
        gate = pl.multiple_of((step % nblk) * ib, ib)
        up = pl.multiple_of(i + (step % nblk) * ib, ib)
        copies = []
        for dst, src in ((0, gate), (ib, up)):
            copies.append(
                pltpu.make_async_copy(
                    w1_hbm.at[e, :, pl.ds(src, ib)],
                    w1_ring.at[slot, :, pl.ds(dst, ib)],
                    sem_ref.at[slot, len(copies)],
                )
            )
            copies.append(
                pltpu.make_async_copy(
                    w1_scale_hbm.at[e, 0, :, pl.ds(src, ib)],
                    w1_scale_ring.at[slot, :, pl.ds(dst, ib)],
                    sem_ref.at[slot, len(copies)],
                )
            )
        copies.append(
            pltpu.make_async_copy(
                w2_hbm.at[e, pl.ds(gate, ib)], w2_ring.at[slot], sem_ref.at[slot, 4]
            )
        )
        copies.append(
            pltpu.make_async_copy(
                w2_scale_hbm.at[e, 0], w2_scale_ring.at[slot], sem_ref.at[slot, 5]
            )
        )
        return copies

    def first_matmul(w1):
        """xq @ w1 ([h, 2 width] ref) in f32, each _Q_BLOCK-row slab of w1
        against its column block of the hidden states, scaled by that block's
        row scales before the sum."""
        acc = None
        for k in range(0, h, _Q_BLOCK):
            part = jnp.matmul(
                xq_ref[:, k : k + _Q_BLOCK],
                w1[k : k + _Q_BLOCK, :],
                preferred_element_type=jnp.float32,
            )
            xs = xs_ref[k // _Q_BLOCK]
            part = part * jnp.concatenate([xs] * (part.shape[1] // 128), axis=1)
            acc = part if acc is None else acc + part
        return acc

    def accumulate(acc1, coeff, w1_scale, w2, w2_scale):
        """Finish one expert from its first-matmul result acc1 [m, 2 width]
        (x scales applied): dequantize, SiLU(gate) * up weighted by coeff
        [m, 1], quantize per row per _Q_BLOCK columns, and add the second
        matmul with w2 [width, h] into acc_ref."""
        width = acc1.shape[1] // 2  # the routed or the shared intermediate size
        acc1 = acc1 * w1_scale
        mid = jnp.where(
            coeff != 0, jax.nn.silu(acc1[:, :width]) * acc1[:, width:] * coeff, 0.0
        )
        # One scale per row per _Q_BLOCK columns; the last block is narrower
        # when width is not a multiple of it (Qwen3.8-Flash EP: i = 640).
        blocks = []
        for k in range(0, width, _Q_BLOCK):
            blk = mid[:, k : k + _Q_BLOCK]
            ms = jnp.max(jnp.abs(blk), axis=1, keepdims=True) / qmax
            blocks.append(
                (
                    (blk * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(weight_dtype),
                    ms,
                    k,
                )
            )
        for n in range(0, h, _N_CHUNK):
            acc2 = None
            for mq, ms, k in blocks:
                part = jnp.matmul(
                    mq,
                    w2[k : k + mq.shape[1], n : n + _N_CHUNK],
                    preferred_element_type=jnp.float32,
                )
                part = part * ms
                acc2 = part if acc2 is None else acc2 + part
            acc_ref[:, n : n + _N_CHUNK] += acc2 * w2_scale[:, n : n + _N_CHUNK]

    def accumulate_unquantized_shared(coeff):
        """A shared expert left unquantized (BF16 in the checkpoint, as
        Qwen3.8-Flash's): BF16 x BF16 matmuls with f32 accumulation, rounded to
        BF16 where vLLM's linears and SiluAndMul round, then gated by coeff
        [m, 1]. Its scale operands are unused."""
        i_s = shared_w2_vmem.shape[0]
        gate_up = jnp.matmul(
            x_ref[...], shared_w1_vmem[...], preferred_element_type=jnp.float32
        )
        gate_up = gate_up.astype(x_ref.dtype).astype(jnp.float32)
        mid = (jax.nn.silu(gate_up[:, :i_s]) * gate_up[:, i_s:]).astype(x_ref.dtype)
        for n in range(0, h, _N_CHUNK):
            down = jnp.matmul(
                mid,
                shared_w2_vmem[:, n : n + _N_CHUNK],
                preferred_element_type=jnp.float32,
            )
            acc_ref[:, n : n + _N_CHUNK] += (
                down.astype(x_ref.dtype).astype(jnp.float32) * coeff
            )

    # Prime the weight ring: start the weight transfers of the first
    # min(units, nb) ring units (active experts, or their blocks) into slots
    # 0 .. nb - 1 now, so they stream from HBM while the shared expert below
    # computes. The expert loop then waits on slot step % nb and refills it
    # with unit step + nb.
    def prime_routed():
        for initial in range(nb):

            @pl.when(units > initial)
            def prime(*, initial=initial):
                for copy in transfers(jnp.int32(initial)):
                    copy.start()

    # q38 L3 D60 (5c): with compact_pipe the prime is issued before the
    # compaction tables below; otherwise where it always was (the same program).
    early_prime = bool(compact_pipe)
    if early_prime:
        pl.when(count > 0)(prime_routed)

    if compact_r is not None and compact_pipe:
        # q38 L3 D60 (5c): the compaction tables, built while the primed
        # weight transfers stream (compact_r alone builds them before the prime).
        gsel_ref, xsort_ref, ctab_ref, tabv_ref, tabs_ref, contrib_ref, acc1buf, tab_sem = (
            compact_scratch
        )
        R = compact_r
        coeff_all = coeff_ref[...]  # [m, epad] f32
        owned = coeff_all != 0
        # occ[t, e]: how many earlier rows expert e owns, by a strictly-lower-
        # triangular 0/1 matmul (0/1 products, f32 sums of at most m ones: exact);
        # code = occ on owned rows, -1 elsewhere (the slot of row t in expert e).
        ti_ = lax.broadcasted_iota(jnp.int32, (m, m), 0)
        tj_ = lax.broadcasted_iota(jnp.int32, (m, m), 1)
        lower = (tj_ < ti_).astype(jnp.bfloat16)
        occ = jnp.dot(
            lower, owned.astype(jnp.bfloat16), preferred_element_type=jnp.float32
        ).astype(jnp.int32)
        code = jnp.where(owned, occ, -1)
        cnt = jnp.sum(owned.astype(jnp.int32), axis=0, keepdims=True)  # [1, epad]
        tok_plane = lax.broadcasted_iota(jnp.int32, code.shape, 0)
        tabv_ref[...] = jnp.zeros(tabv_ref.shape, jnp.int32)
        ctab_ref[...] = jnp.zeros(ctab_ref.shape, jnp.float32)
        for r in range(R):
            # slot r of every expert at once: its token, and that token's route
            # weight (one nonzero term per lane, so both sums are exact)
            at_r = code == r
            tabv_ref[r : r + 1, :] = jnp.sum(
                jnp.where(at_r, tok_plane, 0), axis=0, keepdims=True
            )
            ctab_ref[r : r + 1, :] = jnp.sum(
                jnp.where(at_r, coeff_all, 0.0), axis=0, keepdims=True
            )
        tabv_ref[R : R + 1, :] = cnt[:, :_COMPACT_LANES]
        tab_copy = pltpu.make_async_copy(tabv_ref, tabs_ref, tab_sem)
        tab_copy.start()
        code_t = jnp.transpose(code)  # [epad, m]
        slot_sub = lax.broadcasted_iota(jnp.int32, (R, m), 0)
        for e in range(experts):
            gsel_ref[e * R : (e + 1) * R, :] = (
                code_t[e : e + 1, :] == slot_sub
            ).astype(jnp.bfloat16)
        # every local expert's rows, expert-sorted, in one 0/1 gather matmul
        # per 512 columns (one nonzero term per element: the row, exactly)
        for n in range(0, h, _N_CHUNK):
            xsort_ref[:, n : n + _N_CHUNK] = jnp.dot(
                gsel_ref[...],
                x_ref[:, n : n + _N_CHUNK],
                preferred_element_type=jnp.float32,
            ).astype(xsort_ref.dtype)
        tab_copy.wait()

    if not early_prime:
        pl.when(count > 0)(prime_routed)

    # Shared expert first; its weights were issued at entry with the routed primes.
    with sc("moe_shared"):
        for copy in shared_copies:
            copy.wait()
        shared_coeff = jnp.max(shared_gate_ref[...], axis=1, keepdims=True)
        if shared_w1_vmem.dtype == weight_dtype:
            accumulate(
                first_matmul(shared_w1_vmem),
                shared_coeff,
                shared_w1_scale_vmem[...],
                shared_w2_vmem,
                shared_w2_scale_vmem[...],
            )
        else:
            accumulate_unquantized_shared(shared_coeff)

    def first_matmul_rows(xq_rows, xs_rows, w1):
        """first_matmul on R gathered rows: the same per-block product, scale
        and sum, row by row (xs_rows[k] [R, 1] per column block)."""
        acc = None
        for k in range(0, h, _Q_BLOCK):
            part = jnp.matmul(
                xq_rows[:, k : k + _Q_BLOCK],
                w1[k : k + _Q_BLOCK, :],
                preferred_element_type=jnp.float32,
            )
            xs = jnp.broadcast_to(xs_rows[k // _Q_BLOCK], (xq_rows.shape[0], 128))
            part = part * jnp.concatenate([xs] * (part.shape[1] // 128), axis=1)
            acc = part if acc is None else acc + part
        return acc

    def compact_expert(e, n_e, slot):
        """One expert on its own rows (compact mode): the dense body's
        arithmetic on R gathered rows, then each owned row's contribution
        added into acc_ref (rows it does not own untouched)."""
        base = pl.multiple_of(e * compact_r, compact_r)
        rows_x = xsort_ref[pl.ds(base, compact_r), :].astype(jnp.float32)
        xq_blocks, xs_blocks = [], []
        for k in range(0, h, _Q_BLOCK):
            xk = rows_x[:, k : k + _Q_BLOCK]
            xsc = jnp.max(jnp.abs(xk), axis=1, keepdims=True) / qmax
            xq_blocks.append(
                (xk * jnp.where(xsc == 0, 0.0, 1.0 / xsc)).astype(weight_dtype)
            )
            xs_blocks.append(xsc)
        xq_rows = jnp.concatenate(xq_blocks, axis=1)
        acc1 = first_matmul_rows(xq_rows, xs_blocks, w1_ring.at[slot])
        coeff = csort_ref[pl.ds(base, compact_r), :][:, 0:1]
        width = acc1.shape[1] // 2
        acc1 = acc1 * w1_scale_ring[slot]
        mid = jnp.where(
            coeff != 0, jax.nn.silu(acc1[:, :width]) * acc1[:, width:] * coeff, 0.0
        )
        blocks = []
        for k in range(0, width, _Q_BLOCK):
            blk = mid[:, k : k + _Q_BLOCK]
            ms = jnp.max(jnp.abs(blk), axis=1, keepdims=True) / qmax
            blocks.append(
                (
                    (blk * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(weight_dtype),
                    ms,
                    k,
                )
            )
        w2 = w2_ring.at[slot]
        w2_scale = w2_scale_ring[slot]
        for n in range(0, h, _N_CHUNK):
            acc2 = None
            for mq, ms, k in blocks:
                part = jnp.matmul(
                    mq,
                    w2[k : k + mq.shape[1], n : n + _N_CHUNK],
                    preferred_element_type=jnp.float32,
                )
                part = part * ms
                acc2 = part if acc2 is None else acc2 + part
            contrib_ref[:, n : n + _N_CHUNK] = acc2
        sub8 = lax.broadcasted_iota(jnp.int32, (8, h), 0)
        for r in range(compact_r):

            @pl.when(r < n_e)
            def add_row(r=r):
                p = base + r
                t = tabs_ref[p // _COMPACT_LANES, p % _COMPACT_LANES]
                tb = pl.multiple_of((t // 8) * 8, 8)
                tile = acc_ref[pl.ds(tb, 8), :]
                # the dense body's own expression, acc + acc2 * w2_scale, for the
                # owned row (same rounding, a fused multiply-add included or not);
                # the tile's other rows are written back unchanged
                upd = tile + contrib_ref[r : r + 1, :] * w2_scale
                acc_ref[pl.ds(tb, 8), :] = jnp.where(sub8 == t - tb, upd, tile)

    def dense_expert(e, slot):
        """The dense body for one expert (compact mode's overflow path)."""
        acc1 = first_matmul(w1_ring.at[slot])
        coeff = jnp.sum(jnp.where(cols == e, coeff_ref[...], 0.0), axis=1, keepdims=True)
        accumulate(
            acc1, coeff, w1_scale_ring[slot], w2_ring.at[slot], w2_scale_ring[slot]
        )

    def run_routed_compact():
        def expert_body(step, carry):
            slot = step % nb
            for copy in transfers(step):
                copy.wait()
            e = ids_ref[step]
            n_e = tabs_ref[(experts * compact_r) // _COMPACT_LANES, e]

            @pl.when(n_e <= compact_r)
            def _():
                compact_expert(e, n_e, slot)

            @pl.when(n_e > compact_r)
            def _():
                dense_expert(e, slot)

            @pl.when(step + nb < units)
            def refill():
                for copy in transfers(step + nb):
                    copy.start()

            return carry

        lax.fori_loop(0, units, expert_body, jnp.int32(0))
        # The dense loop adds +0.0 to every row an expert does not own, which
        # turns a -0.0 accumulator into +0.0. After the shared expert no
        # accumulator is -0.0 (+0 + -0 = +0), so this is a no-op kept to pin the
        # dense semantics by construction.
        acc_ref[...] = acc_ref[...] + 0.0

    def pipe_first(step):
        """q38 L3 D60 (5c): ring unit `step`'s gathered-row first matmul (x
        scales applied, w1 scale not yet), compact_expert's own expressions.
        `step` may be `units` (the last step's look-ahead): it then reads the
        last expert's rows against a slot no transfer is writing, and the
        result is never used."""
        e = ids_ref[jnp.minimum(step, units - 1)]
        base = pl.multiple_of(e * compact_r, compact_r)
        rows_x = xsort_ref[pl.ds(base, compact_r), :].astype(jnp.float32)
        xq_blocks, xs_blocks = [], []
        for k in range(0, h, _Q_BLOCK):
            xk = rows_x[:, k : k + _Q_BLOCK]
            xsc = jnp.max(jnp.abs(xk), axis=1, keepdims=True) / qmax
            xq_blocks.append(
                (xk * jnp.where(xsc == 0, 0.0, 1.0 / xsc)).astype(weight_dtype)
            )
            xs_blocks.append(xsc)
        xq_rows = jnp.concatenate(xq_blocks, axis=1)
        return first_matmul_rows(xq_rows, xs_blocks, w1_ring.at[step % nb])

    def pipe_tail(e, slot, acc1):
        """q38 L3 D60 (5c): compact_expert after its first matmul, up to the
        per-row contributions in contrib_ref; returns w2_scale for the row adds.
        The route weight of slot r is ctab[r, e] (compact_r's csort value)."""
        lane = lax.broadcasted_iota(jnp.int32, (compact_r, _COMPACT_LANES), 1)
        coeff = jnp.sum(
            jnp.where(lane == e, ctab_ref[0:compact_r, :], 0.0), axis=1, keepdims=True
        )
        width = acc1.shape[1] // 2
        acc1 = acc1 * w1_scale_ring[slot]
        mid = jnp.where(
            coeff != 0, jax.nn.silu(acc1[:, :width]) * acc1[:, width:] * coeff, 0.0
        )
        blocks = []
        for k in range(0, width, _Q_BLOCK):
            blk = mid[:, k : k + _Q_BLOCK]
            ms = jnp.max(jnp.abs(blk), axis=1, keepdims=True) / qmax
            blocks.append(
                (
                    (blk * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(weight_dtype),
                    ms,
                    k,
                )
            )
        w2 = w2_ring.at[slot]
        w2_scale = w2_scale_ring[slot]
        for n in range(0, h, _N_CHUNK):
            acc2 = None
            for mq, ms, k in blocks:
                part = jnp.matmul(
                    mq,
                    w2[k : k + mq.shape[1], n : n + _N_CHUNK],
                    preferred_element_type=jnp.float32,
                )
                part = part * ms
                acc2 = part if acc2 is None else acc2 + part
            contrib_ref[:, n : n + _N_CHUNK] = acc2
        return w2_scale

    def pipe_row_adds(e, n_e, w2_scale):
        """compact_expert's row adds, the token of slot r read at tabs[r, e]."""
        sub8 = lax.broadcasted_iota(jnp.int32, (8, h), 0)
        for r in range(compact_r):

            @pl.when(r < n_e)
            def add_row(r=r):
                t = tabs_ref[r, e]
                tb = pl.multiple_of((t // 8) * 8, 8)
                tile = acc_ref[pl.ds(tb, 8), :]
                upd = tile + contrib_ref[r : r + 1, :] * w2_scale
                acc_ref[pl.ds(tb, 8), :] = jnp.where(sub8 == t - tb, upd, tile)

    def run_routed_compact_pipe():
        for copy in transfers(jnp.int32(0)):
            copy.wait()
        acc1buf[0] = pipe_first(jnp.int32(0))

        def expert_body(step, carry):
            slot = step % nb
            nxt = step + 1
            e = ids_ref[step]
            n_e = tabs_ref[compact_r, e]

            @pl.when(nxt < units)
            def wait_next():
                for copy in transfers(nxt):
                    copy.wait()

            @pl.when(n_e <= compact_r)
            def _():
                w2_scale = pipe_tail(e, slot, acc1buf[step % 2])
                acc1buf[nxt % 2] = pipe_first(nxt)
                pipe_row_adds(e, n_e, w2_scale)

            @pl.when(n_e > compact_r)
            def _():
                dense_expert(e, slot)
                acc1buf[nxt % 2] = pipe_first(nxt)

            @pl.when(step + nb < units)
            def refill():
                for copy in transfers(step + nb):
                    copy.start()

            return carry

        lax.fori_loop(0, units, expert_body, jnp.int32(0))
        # as run_routed_compact: pins the dense loop's -0.0 -> +0.0 by construction
        acc_ref[...] = acc_ref[...] + 0.0

    if compact_r is not None and not compact_pipe:
        with sc("moe_experts"):
            pl.when(count > 0)(run_routed_compact)
    if compact_r is not None and compact_pipe:
        with sc("moe_experts"):
            pl.when(count > 0)(run_routed_compact_pipe)

    def run_routed():
        for copy in transfers(jnp.int32(0)):
            copy.wait()

        def expert_body(step, acc1):
            nxt = step + 1

            @pl.when(nxt < units)
            def wait_next():
                for copy in transfers(nxt):
                    copy.wait()

            acc1_next = first_matmul(w1_ring.at[nxt % nb])
            slot = step % nb
            e = ids_ref[step] if ib is None else ids_ref[step // nblk]
            coeff = jnp.sum(
                jnp.where(cols == e, coeff_ref[...], 0.0), axis=1, keepdims=True
            )
            accumulate(
                acc1, coeff, w1_scale_ring[slot], w2_ring.at[slot], w2_scale_ring[slot]
            )

            @pl.when(step + nb < units)
            def refill():
                for copy in transfers(step + nb):
                    copy.start()

            return acc1_next

        lax.fori_loop(
            0,
            units,
            expert_body,
            first_matmul(w1_ring.at[0]),
        )

    if compact_r is None:
        with sc("moe_experts"):
            pl.when(count > 0)(run_routed)

    def all_to_all_sum(send_slice, dst):
        """Reduce-scatter semantics by a direct all-to-all and a local sum,
        which both output modes end with. send_slice(d) is this rank's partial
        of the slice rank d keeps; one remote copy per peer sends it straight
        to rank d's recv[me]. Each rank then sums the n slots it received in FP32
        into dst. There is no ring and no per-hop reduction: a single hop of n
        concurrent copies, where a ring needs n - 1 dependent steps for the
        same bytes, which suits the latency-bound sizes of decode."""
        # Every peer has entered this call, so its receive buffer is live.
        with sc("ar_barrier_wait"):
            pl.semaphore_wait(barrier, n_ranks)
        a2a_scope = sc("ar_a2a")
        a2a_scope.__enter__()
        copies = [
            pltpu.make_async_remote_copy(
                send_slice(d),
                recv.at[me],
                a2a_send.at[d],
                a2a_recv,
                device_id=(jnp.int32(d),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            for d in range(n_ranks)
        ]
        for copy in copies:
            copy.start()
        pltpu.make_async_copy(recv, recv, a2a_recv).wait()
        acc = recv[0].astype(jnp.float32)
        for s in range(1, n_ranks):
            acc = acc + recv[s].astype(jnp.float32)
        dst[...] = acc.astype(dst.dtype)
        for copy in copies:
            copy.wait_send()
        a2a_scope.__exit__(None, None, None)

    if rows is not None:
        # DP attention: all-to-all + local sum over token rows, no all-gather.
        # Send slice d is this rank's partial of tokens [d * rows, (d + 1) *
        # rows), the rows rank d keeps. Each send slice is a whole leading
        # index of acc_bf16 / recv, so it is tile aligned for any rows.
        for d in range(n_ranks):
            acc_bf16[d, :rows, :] = acc_ref[d * rows : (d + 1) * rows, :].astype(
                acc_bf16.dtype
            )
        all_to_all_sum(lambda d: acc_bf16.at[d], out_ref)
        return

    # TP attention, all-reduce: all-to-all + local sum over column chunks (rank
    # d keeps columns [d * chunk, (d + 1) * chunk) of every token), then an
    # all-gather of the reduced chunks.
    if exact_width and h // 128 >= n_ranks:
        # q38 PORT-BL MOE (exact width; ported from Benliang Wang's b0c1453ba).
        # Balanced ownership of the h // 128 lane tiles: rank r keeps
        # tiles(r) = base + (r < extra) tiles from tile r * base + min(r, extra)
        # (Qwen3.8-Flash: 3, 3, 3, 3, 2, 2, 2, 2 tiles of 128), the same columns
        # in the all-to-all and the all-gather, so every pair exchanges the
        # owner's width and no rank sends padding. The receiver sums its slots
        # in rank order whichever rank owns a column, as the padded split does.
        base, extra = divmod(h // 128, n_ranks)
        widths = [128 * (base + (r < extra)) for r in range(n_ranks)]
        starts = [128 * (r * base + min(r, extra)) for r in range(n_ranks)]
        acc_bf16[:, :h] = acc_ref[...].astype(acc_bf16.dtype)
        # Every peer has entered the call, so its receive buffer is live.
        with sc("ar_barrier_wait"):
            pl.semaphore_wait(barrier, n_ranks)
        with sc("ar_a2a"):
            scatter = [
                pltpu.make_async_remote_copy(
                    acc_bf16.at[:, pl.ds(starts[d], widths[d])],
                    recv.at[me, :, pl.ds(0, widths[d])],
                    a2a_send.at[d],
                    a2a_recv,
                    device_id=(jnp.int32(d),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                for d in range(n_ranks)
            ]
            for copy in scatter:
                copy.start()
        for owner in range(n_ranks):
            width = widths[owner]

            @pl.when(me == owner)
            def reduce_and_gather(*, owner=owner, width=width):
                with sc("ar_a2a"):
                    # The receive wait counts the n slots of this rank's width.
                    received = recv.at[:, :, pl.ds(0, width)]
                    pltpu.make_async_copy(received, received, a2a_recv).wait()
                    acc = recv[0, :, :width].astype(jnp.float32)
                    for s in range(1, n_ranks):
                        acc = acc + recv[s, :, :width].astype(jnp.float32)
                    red[:, :width] = acc.astype(red.dtype)
                with sc("ar_gather"):
                    gather = [
                        pltpu.make_async_remote_copy(
                            red.at[:, pl.ds(0, width)],
                            out_ref.at[:, pl.ds(starts[owner], width)],
                            ag_send.at[d],
                            ag_recv,
                            device_id=(jnp.int32(d),),
                            device_id_type=pl.DeviceIdType.MESH,
                        )
                        for d in range(n_ranks)
                    ]
                    for copy in gather:
                        copy.start()
                    # out_ref is [m, h]; the owners' widths sum to h.
                    pltpu.make_async_copy(out_ref, out_ref, ag_recv).wait()
                    for copy in gather:
                        copy.wait_send()

        for copy in scatter:
            copy.wait_send()
        return
    if hp != h:
        acc_bf16[:, h:] = jnp.zeros((m, hp - h), acc_bf16.dtype)
        acc_bf16[:, :h] = acc_ref[...].astype(acc_bf16.dtype)
    else:
        acc_bf16[...] = acc_ref[...].astype(acc_bf16.dtype)
    all_to_all_sum(lambda d: acc_bf16.at[:, pl.ds(d * chunk, chunk)], red)
    col = pl.multiple_of(me * chunk, chunk)
    gather_scope = sc("ar_gather")
    gather_scope.__enter__()
    gather = [
        pltpu.make_async_remote_copy(
            red,
            out_ref.at[:, pl.ds(col, chunk)],
            ag_send.at[d],
            ag_recv,
            device_id=(jnp.int32(d),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        for d in range(n_ranks)
    ]
    for copy in gather:
        copy.start()
    pltpu.make_async_copy(out_ref, out_ref, ag_recv).wait()
    for copy in gather:
        copy.wait_send()
    gather_scope.__exit__(None, None, None)


class _ActiveBits:
    """The folded call's stand-in for active_ref: expert e's flag, read from
    16-bit groups of flags that the prologue moved to scalars (a sum of
    distinct powers of two below 2**16 is exact in f32)."""

    def __init__(self, groups):
        self.groups = groups

    def __getitem__(self, e):
        g = self.groups[0]
        for j in range(1, len(self.groups)):
            g = jnp.where(e >= 16 * j, self.groups[j], g)
        return lax.shift_right_logical(g, e % 16) & 1


def _folded_kernel(
    rank_ref,
    x_ref,
    tw_ref,
    ti_ref,
    *refs,
    tokens,
    m,
    experts,
    expert_parallel,
    routed_scale,
    gate,
    **kwargs,
):
    """_kernel behind the routing prologue the unfolded call runs in XLA, so
    the call takes the router's outputs and the hidden states as they are.

    Scalar prefetch: rank_ref [1, 1 or 2] int32: this rank's mesh index and,
    with expert parallelism, its first expert's global id.
    Inputs in VMEM: x_ref [tokens, h] bf16; tw_ref [tokens, k] route weights
    (any float dtype); ti_ref [tokens, k] int32 global expert ids; then, by
    `gate`: "xla" gate_ref [tokens, 1] f32 = sigmoid(x . w_g) computed by the
    caller, "kernel" gate_ref = w_g as stored ([1, h] or [h, 1]), None no
    operand (ungated). The weights in HBM, output and scratch as _kernel's,
    plus coeff [m, E padded to 128] f32, shared gate [m, 128] f32 and, when
    tokens < m, the hidden states zero-padded to m rows, bf16 [m, h].

    Each value is the one the unfolded call computes, by the same f32 ops in
    the same order (coeff accumulates the routes in route order from +0), so
    the output is bitwise the same; only rows past `tokens`, which no caller
    reads, differ.
    """
    gate_ref, refs = (refs[0], refs[1:]) if gate else (None, refs)
    hbm, out_ref, scratch = refs[:8], refs[8], refs[9:]
    pad = m - tokens
    if pad:
        *scratch, coeff_ref, shared_gate_ref, xpad_ref = scratch
    else:
        *scratch, coeff_ref, shared_gate_ref = scratch
    f32 = jnp.float32
    h = x_ref.shape[1]
    if pad:
        for k in range(0, h, _Q_BLOCK):
            x = x_ref[:, k : k + _Q_BLOCK].astype(f32)
            xpad_ref[:, k : k + _Q_BLOCK] = jnp.concatenate(
                [x, jnp.zeros((pad, _Q_BLOCK), f32)], axis=0
            ).astype(xpad_ref.dtype)
        x_src = xpad_ref
    else:
        x_src = x_ref

    epad = coeff_ref.shape[1]
    lanes = lax.broadcasted_iota(jnp.int32, (tokens, epad), 1)
    ids = ti_ref[...]
    if expert_parallel:
        ids = ids - rank_ref[0, 1]
    w = tw_ref[...].astype(x_ref.dtype).astype(f32)
    if routed_scale != 1.0:
        w = w * f32(routed_scale)
    coeff = jnp.zeros((tokens, epad), f32)
    for r in range(ids.shape[1]):
        coeff = coeff + jnp.where(
            (ids[:, r : r + 1] == lanes) & (lanes < experts), w[:, r : r + 1], 0.0
        )
    if pad:
        coeff = jnp.concatenate([coeff, jnp.zeros((pad, epad), f32)], axis=0)
    coeff_ref[...] = coeff

    act = jnp.max(jnp.where(coeff != 0, 1.0, 0.0), axis=0, keepdims=True)
    lane = lax.broadcasted_iota(jnp.int32, (1, epad), 1)
    groups = []
    for j in range(-(-experts // 16)):
        inside = (lane >= 16 * j) & (lane < 16 * j + 16)
        bit = jnp.left_shift(jnp.int32(1), jnp.where(inside, lane - 16 * j, 0))
        groups.append(
            jnp.sum(jnp.where(inside, act * bit.astype(f32), 0.0)).astype(jnp.int32)
        )

    if gate is None:
        g = jnp.ones((m, 1), f32)
    elif gate == "xla":
        g = gate_ref[...]
        if pad:
            g = jnp.concatenate([g, jnp.ones((pad, 1), f32)], axis=0)
    else:
        # w_g repeated over 128 rows, so the MXU returns the gate already
        # broadcast over lanes (an N=1 dot does not lower).
        wg = gate_ref[...].astype(x_ref.dtype)
        wg = wg if wg.shape[0] == 1 else wg.T
        g = jax.nn.sigmoid(
            lax.dot_general(
                x_src[...],
                jnp.broadcast_to(wg, (128, h)),
                (((1,), (1,)), ((), ())),
                preferred_element_type=f32,
            )
        )
    shared_gate_ref[...] = jnp.broadcast_to(g, (m, 128))

    _kernel(
        _ActiveBits(groups),
        rank_ref,
        x_src,
        coeff_ref,
        shared_gate_ref,
        *hbm,
        out_ref,
        *scratch,
        m=m,
        experts=experts,
        folded=True,
        **kwargs,
    )


def _padded_hidden(h, n_ranks):
    """Width of the TP all-reduce buffers: h rounded up to n_ranks lanes."""
    return -(-h // (n_ranks * 128)) * (n_ranks * 128)


def _vmem_capacity():
    return pltpu.get_tpu_info().vmem_capacity_bytes


def _weight_ring_depth(h, i, weight_dtype):
    """Deepest weight ring (at most _NUM_BUFFERS slots, at least 3) whose slots
    fit in half of VMEM; one slot holds an expert's w1 and w2."""
    slot = 3 * h * i * jnp.dtype(weight_dtype).itemsize
    fit = _vmem_capacity() // 2 // slot
    return max(3, min(_NUM_BUFFERS, fit))


# Blocked mode: candidate block widths in order of preference, and the VMEM
# held back from the weight ring for Mosaic's internal scratch. 256 first: on
# GLM-5.2 EP16 (2x2x2) it tied or beat 512 and 1024 at every token count
# measured (1-256), its deeper ring keeping more of the expert in flight.
_I_BLOCKS = (256, 128)
_VMEM_RESERVE = 8 << 20


def _unit_bytes(h, width, wt):
    """One ring slot: w13 [h, 2 width] + w2 [width, h] plus their f32 scales."""
    return 3 * h * width * wt + (2 * width + h) * 4


def _token_side_bytes(m, h, i_s, experts, n_ranks, wt):
    """Everything the call declares but the routed weight ring: the shared
    expert (counted at 2 bytes per weight, its widest dtype) and the
    token-proportional buffers."""
    f32, bf16 = 4, 2
    shared = _unit_bytes(h, i_s, max(wt, bf16))
    hp = _padded_hidden(h, n_ranks)
    rows = m * (
        h * bf16  # hidden
        + (experts + (-experts) % 128) * f32  # coeff
        + 128 * f32  # shared gate
        + hp * bf16  # output
        + h * wt  # quantized hidden
        + (h // _Q_BLOCK) * 128 * f32  # per-block row scales, over lanes
        + h * f32  # accumulator
        + hp * bf16  # accumulator in bf16, the all-reduce source
        + hp * bf16  # all-to-all receive slots
        + (hp // n_ranks) * bf16  # reduced chunk
    )
    return shared + rows


def plan_weight_ring(m, h, i, i_s, experts, n_ranks, weight_dtype, i_block=None):
    """(ib, nb) for `m` padded tokens: the block width (None for whole
    experts) and the ring depth, chosen at trace time.

    Whole experts, depth _weight_ring_depth, whenever three whole-expert slots
    alone fit VMEM: the decision depends on the expert's size only, so small
    experts (Qwen3.5, GLM-5.2 at TP) never switch mode with the token count.
    Otherwise (or when i_block forces it) each expert streams in blocks of ib
    intermediate columns, ib the first of _I_BLOCKS narrower than and dividing
    i, the ring as deep as fits beside the token-side buffers (3 to
    _NUM_BUFFERS units). No such width: whole experts.
    """
    wt = jnp.dtype(weight_dtype).itemsize
    capacity = _vmem_capacity()
    if i_block is None:
        widths = [w for w in _I_BLOCKS if w < i and i % w == 0]
        if 3 * _unit_bytes(h, i, wt) <= capacity or not widths:
            return None, _weight_ring_depth(h, i, weight_dtype)
    elif i % i_block or i_block % 128:
        raise ValueError(f"i_block {i_block} must divide i={i} in lanes of 128")
    else:
        widths = [i_block]

    def fit(w):
        live = 2 * m * 2 * w * 4  # this unit's and the next's first-matmul result
        free = (
            capacity
            - _VMEM_RESERVE
            - live
            - _token_side_bytes(m, h, i_s, experts, n_ranks, wt)
        )
        return free // _unit_bytes(h, w, wt)

    ib = next((w for w in widths if fit(w) >= 3), widths[-1])
    return ib, max(3, min(_NUM_BUFFERS, fit(ib)))


def vmem_bytes(tokens, h, i, i_s, experts, n_ranks, weight_dtype):
    """VMEM the call declares for `tokens` rows: its VMEM operands, output and
    scratch (compiler temporaries excluded). The weight ring takes about half
    of VMEM, so the token-proportional buffers bound the rows one call takes."""
    m = tokens + (-tokens) % 16
    wt = jnp.dtype(weight_dtype).itemsize
    ib, nb = plan_weight_ring(m, h, i, i_s, experts, n_ranks, weight_dtype)
    return nb * _unit_bytes(h, i if ib is None else ib, wt) + _token_side_bytes(
        m, h, i_s, experts, n_ranks, wt
    )


def fused_moe_decode(
    rank,
    hidden,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    topk_weights,
    topk_ids,
    *,
    n_ranks,
    expert_start=None,
    scatter_rows=False,
    routed_scale=1.0,
    i_block=None,
    interpret=False,
    fold=False,
    gate_in_kernel=False,
    compact_rows=None,
    scopes=0,
    compact_pipe=False,
    exact_width=False,
):
    """Per-rank body; call inside shard_map over the TP mesh.

    Returns the TP-all-reduced (shared + routed) MoE output, BF16 [tokens,
    hidden], identical on every rank (with scatter_rows, only this rank's
    rows; see below).

    Args:
      rank: this rank's mesh index, an int32 array of one element
        (partition-id cannot be exported through torch_tpu).
      hidden: [tokens, H] BF16 hidden states, the same on every rank.
      w1: [E, H, 2 i] this rank's gate and up projections of its E experts,
        concatenated on the last axis; i is the rank's intermediate slice.
      w2: [E, i, H] the down projections.
      w1_scale: [E, 1, 1, 2 i] FP32 per-output-channel scales of w1.
      w2_scale: [E, 1, 1, H] FP32 per-output-channel scales of w2.
      shared_w1: [H, 2 i_s] the shared expert's gate and up slice (i_s <= i),
        in the routed weights' dtype, or BF16 for a shared expert the
        checkpoint leaves unquantized (Qwen3.8-Flash); a BF16 shared expert
        runs BF16 matmuls and ignores its two scale operands.
      shared_w1_scale: [1, 2 i_s] FP32 scales of shared_w1.
      shared_w2: [i_s, H] the shared expert's down slice.
      shared_w2_scale: [1, H] FP32 scales of shared_w2.
      shared_gate_weight: [H, 1] the replicated shared_expert_gate; the shared
        expert's output is scaled by sigmoid(hidden @ shared_gate_weight)
        (Qwen3.5). None for an ungated shared expert (GLM-5.2, DeepSeek-V3).
      topk_weights: [tokens, k] route weights from the caller's router.
      topk_ids: [tokens, k] int32 expert ids of those routes.
      n_ranks: ranks in the mesh, the all-reduce width.
      expert_start: with expert parallelism, the global id of this rank's
        first expert; the rank holds experts [expert_start, expert_start + E)
        and topk_ids are global. Routes to other ranks' experts contribute
        nothing here and are summed in by the all-reduce. None for TP experts.
      scatter_rows: for DP attention. The caller all-gathered every rank's
        tokens and routing rank-major, so `hidden` holds the global batch.
        The all-reduce becomes an all-to-all + local sum over token rows
        (reduce-scatter semantics): rank r returns only its own rows
        [r * tokens / n_ranks, (r + 1) * tokens / n_ranks), [tokens / n_ranks,
        H], and no all-gather runs. Needs tokens
        % n_ranks == 0. The wire is BF16 when tokens / n_ranks is a multiple
        of 16, else FP32.
      routed_scale: multiplies the routed experts' sum only, never the shared
        expert, as vLLM's MoERunner applies routed_scaling_factor to the
        routed output before adding the shared one (GLM-5.2: 2.5). Folded
        into the route weights after their rounding to the activation dtype,
        where the GMM path rounds them.
      i_block: force blocked mode with this block width (tests); by default
        plan_weight_ring picks it for experts too large to stream whole.
      interpret: run the Pallas interpreter (CPU tests).
      fold: run the routing prologue (row padding, route weights to coeff,
        active experts, shared gate broadcast) inside the kernel instead of as
        separate XLA ops, which cost ~0.6 us each per call at decode. Bitwise
        the same output. `rank` is then [1, 1] int32, or [1, 2] holding
        (rank, expert_start) with expert parallelism, where expert_start must
        be the string "rank"; shared_gate_weight may be [1, H] or [H, 1] and
        a BF16 shared expert's scales may be 1-D.
      gate_in_kernel: with fold, also compute sigmoid(hidden . w_g) inside
        the kernel (its bitwise match to XLA's dot is checked by test, not
        guaranteed by construction).
      scopes: q38 DEC-MOEKV (IQD #10 measurement). 0 (default) emits no markers, the same program. 1 wraps
        the prologue, shared expert, routed experts and the all-reduce's barrier wait / all-to-all / all-gather
        in named scopes: Mosaic trace regions, visible with LIBTPU_INIT_ARGS=
        --xla_enable_custom_call_region_trace=true. For traces only; not a numerics reference.
      compact_rows: q38 DEC-FB-2 lever 5b. None (default) computes every
        active expert on all rows. An int R (a multiple of 16) computes each
        expert on its own routed rows, up to R (experts with more run the
        dense body): the dense body is compute-bound at large token counts
        (each expert owns ~5 of 256 rows), this is weight-streaming bound.
        Bitwise the same output by construction (see _kernel). Whole-expert
        mode, TP all-reduce output (no scatter_rows), at most 128 local
        experts.
      compact_pipe: q38 L3 D60 (form 5c). With compact_rows, the same
        arithmetic scheduled for the weight stream: ring primed before the
        compaction prologue, slot-major tables, software-pipelined expert
        loop (see _kernel). Bitwise compact_rows' (and so the dense) output.
        The kernel's name ends in -cr_<R>p.
      exact_width: q38 PORT-BL MOE (FUSED_MOE_DECODE_EXACT_WIDTH; ported from
        Benliang Wang's b0c1453ba). TP all-reduce output only: the exchange
        splits H's lane tiles over the ranks as evenly as whole tiles allow (no
        rank sends padding, the same columns in the all-to-all and the
        all-gather) and the kernel writes [m, H] instead of [m, H padded to
        n_ranks lanes of 128], so no slice follows it. Bitwise the same values.
        Needs H // 128 >= n_ranks; otherwise (and with scatter_rows) the padded
        split runs. The kernel's name then carries -xw.

    The weights' dtype (FP8 in production) is also the dtype the kernel
    quantizes the hidden states and the activated intermediate to (one scale
    per row per 512 columns), so both matmuls run in it with FP32
    accumulation. The hidden size must be a multiple of 512.
    """
    tokens, h = hidden.shape
    if h % _Q_BLOCK:
        raise ValueError(f"hidden size {h} must be a multiple of {_Q_BLOCK}")
    if scatter_rows and tokens % n_ranks:
        raise ValueError(f"scatter_rows needs tokens % n_ranks == 0; got {tokens}")
    rows = tokens // n_ranks if scatter_rows else None
    experts, i, _ = w2.shape
    pad = (-tokens) % 16
    m = tokens + pad
    if fold:
        expert_parallel = expert_start is not None
        if rows is not None:
            raise ValueError("fold does not support scatter_rows")
        if expert_parallel and expert_start != "rank":
            raise ValueError('fold with expert parallelism needs expert_start="rank"')
        if rank.ndim != 2 or rank.shape[1] < 1 + expert_parallel:
            raise ValueError(f"fold needs rank [1, 1 or 2]: {rank.shape}")
        gate, gate_operands = None, []
        if shared_gate_weight is not None and gate_in_kernel:
            gate, gate_operands = "kernel", [shared_gate_weight]
        elif shared_gate_weight is not None:
            # The unfolded call's own dot, on the [H, 1] weight: XLA's NT form
            # of it rounds differently [measured: j-d1daee42].
            gate = "xla"
            gate_operands = [
                jax.nn.sigmoid(
                    jnp.dot(
                        hidden,
                        shared_gate_weight.reshape(-1, 1).astype(hidden.dtype),
                        preferred_element_type=jnp.float32,
                    )
                )
            ]
    else:
        if pad:
            hidden = jnp.pad(hidden, ((0, pad), (0, 0)))
            topk_weights = jnp.pad(topk_weights, ((0, pad), (0, 0)))
            topk_ids = jnp.pad(topk_ids, ((0, pad), (0, 0)))
        if expert_start is not None:
            topk_ids = topk_ids - expert_start
        active, coeff, shared_gate = _routing_prologue(
            hidden, topk_weights, topk_ids, shared_gate_weight, m, experts, routed_scale
        )
    i_s = shared_w2.shape[0]
    weight_dtype = w1.dtype
    ib, nb = plan_weight_ring(m, h, i, i_s, experts, n_ranks, weight_dtype, i_block)
    if compact_pipe and compact_rows is None:
        raise ValueError("compact_pipe needs compact_rows")
    if compact_rows is not None:
        compact_rows = int(compact_rows)
        if compact_rows <= 0 or compact_rows % 16:
            raise ValueError(f"compact_rows must be a positive multiple of 16: {compact_rows}")
        if ib is not None or rows is not None:
            raise ValueError("compact_rows needs whole-expert mode and no scatter_rows")
        if experts > _COMPACT_LANES or (experts * compact_rows) % _COMPACT_LANES:
            raise ValueError(
                f"compact_rows needs <= {_COMPACT_LANES} local experts and experts * R "
                f"a multiple of {_COMPACT_LANES}: {experts} x {compact_rows}"
            )
    w = i if ib is None else ib  # ring slot width
    hp = _padded_hidden(h, n_ranks)
    chunk = hp // n_ranks
    out_cols = h
    # q38 PORT-BL MOE: exact width only on the TP all-reduce output with at
    # least one 128-lane tile per rank; otherwise the padded split.
    exact_width = bool(exact_width) and rows is None and h // 128 >= n_ranks
    if rows is None:
        out_rows, wire, out_cols = m, hidden.dtype, h if exact_width else hp
        acc_bf16_shape, recv_shape, red_shape = (
            (m, hp),
            (n_ranks, m, chunk),
            (m, chunk),
        )
    else:
        # BF16 packs 16 rows per tile; smaller send slices go over the wire in
        # FP32.
        wire = hidden.dtype if rows % 16 == 0 else jnp.float32
        out_rows = rows if rows % 16 == 0 else -(-rows // 8) * 8
        acc_bf16_shape = recv_shape = (n_ranks, out_rows, h)
        red_shape = (8, 128)  # unused
    scratch_shapes = [
        pltpu.VMEM((nb, h, 2 * w), weight_dtype),
        pltpu.VMEM((nb, w, h), weight_dtype),
        pltpu.VMEM((nb, 1, 2 * w), jnp.float32),
        pltpu.VMEM((nb, 1, h), jnp.float32),
        pltpu.VMEM((h, 2 * i_s), shared_w1.dtype),
        pltpu.VMEM((1, 2 * i_s), jnp.float32),
        pltpu.VMEM((i_s, h), shared_w2.dtype),
        pltpu.VMEM((1, h), jnp.float32),
        pltpu.VMEM((m, h), weight_dtype),
        pltpu.VMEM((h // _Q_BLOCK, m, 128), jnp.float32),
        pltpu.VMEM((m, h), jnp.float32),
        pltpu.VMEM(acc_bf16_shape, wire),
        pltpu.VMEM(recv_shape, wire),
        pltpu.VMEM(red_shape, wire),
        pltpu.SMEM((experts,), jnp.int32),
        pltpu.SemaphoreType.DMA((nb, 4 if ib is None else 6)),
        pltpu.SemaphoreType.DMA((4,)),
        pltpu.SemaphoreType.DMA((n_ranks,)),
        pltpu.SemaphoreType.DMA(()),
        pltpu.SemaphoreType.DMA((n_ranks,)),
        pltpu.SemaphoreType.DMA(()),
    ]
    if compact_rows is not None and compact_pipe:
        slots = experts * compact_rows
        tab_rows = -(-(compact_rows + 1) // 8) * 8  # R token rows + the count row
        scratch_shapes += [
            pltpu.VMEM((slots, m), jnp.bfloat16),  # gsel
            pltpu.VMEM((slots, h), jnp.bfloat16),  # xsort
            pltpu.VMEM((tab_rows, _COMPACT_LANES), jnp.float32),  # ctab
            pltpu.VMEM((tab_rows, _COMPACT_LANES), jnp.int32),  # tabv
            pltpu.SMEM((tab_rows, _COMPACT_LANES), jnp.int32),  # tabs
            pltpu.VMEM((compact_rows, h), jnp.float32),  # contrib
            pltpu.VMEM((2, compact_rows, 2 * i), jnp.float32),  # acc1buf
            pltpu.SemaphoreType.DMA(()),
        ]
    elif compact_rows is not None:
        slots = experts * compact_rows
        tab_rows = -(-(slots // _COMPACT_LANES + 1) // 8) * 8
        scratch_shapes += [
            pltpu.VMEM((slots, m), jnp.bfloat16),  # gsel
            pltpu.VMEM((slots, h), jnp.bfloat16),  # xsort
            pltpu.VMEM((slots, 128), jnp.float32),  # csort
            pltpu.VMEM((tab_rows, _COMPACT_LANES), jnp.int32),  # tabv
            pltpu.SMEM((tab_rows, _COMPACT_LANES), jnp.int32),  # tabs
            pltpu.VMEM((compact_rows, h), jnp.float32),  # contrib
            pltpu.SemaphoreType.DMA(()),
        ]
    weights = (
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
    )
    kernel_kwargs = dict(
        m=m,
        h=h,
        i=i,
        experts=experts,
        nb=nb,
        n_ranks=n_ranks,
        weight_dtype=weight_dtype,
        rows=rows,
        ib=ib,
        compact_r=compact_rows,
        scopes=int(scopes),
    )
    if compact_pipe:
        kernel_kwargs["compact_pipe"] = True
    if exact_width:  # q38 PORT-BL MOE: only when on, so the off kwargs are unchanged
        kernel_kwargs["exact_width"] = True
    if fold:
        scratch_shapes += [
            pltpu.VMEM((m, experts + (-experts) % 128), jnp.float32),
            pltpu.VMEM((m, 128), jnp.float32),
        ]
        if pad:
            scratch_shapes.append(pltpu.VMEM((m, h), hidden.dtype))
        kernel = functools.partial(
            _folded_kernel,
            tokens=tokens,
            expert_parallel=expert_parallel,
            routed_scale=routed_scale,
            gate=gate,
            **kernel_kwargs,
        )
        num_scalar_prefetch = 1
        vmem_operands = [hidden, topk_weights, topk_ids, *gate_operands]
        operands = (rank.astype(jnp.int32), *vmem_operands, *weights)
    else:
        kernel = functools.partial(_kernel, **kernel_kwargs)
        num_scalar_prefetch = 2
        vmem_operands = [hidden, coeff, shared_gate]
        operands = (
            active,
            rank.reshape(1).astype(jnp.int32),
            *vmem_operands,
            *weights,
        )
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((out_rows, out_cols), wire),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * len(vmem_operands)
            + [pl.BlockSpec(memory_space=pltpu.HBM)] * 8,
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            vmem_limit_bytes=_vmem_capacity(),
        ),
        name=f"fused_moe_decode{'_rs' if rows else ''}{'_fold' if fold else ''}"
        f"-e_{experts}-m_{m}-h_{h}"
        f"-i_{i}-is_{i_s}{'' if shared_w1.dtype == weight_dtype else '_bf16'}"
        f"-n_{n_ranks}-buffers_{nb}{f'-ib_{ib}' if ib else ''}"
        f"{f'-cr_{compact_rows}' if compact_rows else ''}{'p' if compact_pipe else ''}"
        f"{'-xw' if exact_width else ''}",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(*operands)
    if rows is not None:
        return out[:rows].astype(hidden.dtype)
    if hp != h and not exact_width:
        out = out[:, :h]
    return out[:tokens] if pad else out


def _routing_prologue(
    hidden, topk_weights, topk_ids, shared_gate_weight, m, experts, routed_scale
):
    """The unfolded call's XLA prologue on the padded rows: active [E] int32,
    coeff [m, E padded to 128] f32 and shared_gate [m, 128] f32 (see _kernel)."""
    expert_ids = jnp.arange(experts, dtype=jnp.int32)[None, :]
    weights = topk_weights.astype(hidden.dtype).astype(jnp.float32)
    if routed_scale != 1.0:
        weights = weights * jnp.float32(routed_scale)
    coeff = jnp.zeros((m, experts), jnp.float32)
    for route in range(topk_ids.shape[1]):
        selected = topk_ids[:, route : route + 1] == expert_ids
        coeff = coeff + selected.astype(jnp.float32) * weights[:, route : route + 1]
    active = jnp.any(coeff != 0, axis=0).astype(jnp.int32)
    coeff = jnp.pad(coeff, ((0, 0), (0, (-experts) % 128)))
    if shared_gate_weight is None:
        shared_gate = jnp.ones((m, 128), jnp.float32)
    else:
        shared_gate = jax.nn.sigmoid(
            jnp.dot(
                hidden,
                shared_gate_weight.astype(hidden.dtype),
                preferred_element_type=jnp.float32,
            )
        )
        shared_gate = jnp.broadcast_to(shared_gate, (m, 128))
    return active, coeff, shared_gate
