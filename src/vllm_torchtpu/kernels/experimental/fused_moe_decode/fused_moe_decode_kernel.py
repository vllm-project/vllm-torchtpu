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
    *routed_refs,
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
    routed=None,
    ar2d=False,
    ar2d_delay_ns=None,
    ar2d_partial_ref=None,
    exact_width=False,
    valid_rows=False,
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

    ar2d (static): the TP all-reduce runs _ar2d_exchange instead. acc_bf16 is
    then [m, h], recv the s1 receive [m, h / 2], red AF [m, h / 2], a2a_send and
    a2a_recv DMA [len(AR2D_STREAMS)] (send and receive semaphore per stream),
    ag_send and ag_recv unused, and routed_refs starts with _ar2d_scratch's
    eight buffers. ar2d_delay_ns, ar2d_partial_ref: test-only, see
    _ar2d_exchange.

    valid_rows (static, folded only): rank_ref[0, 2] is the count n of real
    rows; rows from n on read as zero hidden states, whatever x_ref holds
    there. The caller zeroes their coefficients and gives them a finite gate.

    routed (static): None runs every active expert over all m rows (dense).
    (block, tile_rows) runs each active expert only over the aligned blocks
    of `block` rows that hold at least one of its routed rows: the blocks
    are copied into a tile of tile_rows rows, the two matmuls run on the
    tile, and each block's rows are added back into acc at their positions.
    Rows of a copied block that the expert does not own carry coefficient 0
    and add 0, as every row does in the dense mode, so each row receives the
    same terms in the same expert order and the output is the dense mode's
    bit for bit. An expert with more blocks than a tile runs further tiles
    on the same ring slot. routed_refs then holds:
      xqf_ref [m, h] f32: the quantized hidden states as f32 (exact), the
        gather source;
      xqt_f32 [tile_rows, h] f32, xq_t [tile_rows, h] weight_dtype,
        xs_t [h / _Q_BLOCK, tile_rows, 128] f32, c_t [tile_rows, 128] f32:
        one tile's rows, quantized rows, row scales and coefficients;
      gm_ref [ngroups padded to 8, 128] f32: per local expert (lane), its
        block flags in groups of 16 bits, block b at bit b % 16 of group
        b // 16 (a sum of distinct powers of two below 2**16, exact in f32);
      list_ref SMEM [m / block] int32: the current expert's block ids;
      grp_ref SMEM [ngroups] int32: the current expert's flag groups.
    """
    nblk = 1 if ib is None else i // ib
    me = rank_ref[0, 0] if folded else rank_ref[0]
    keep = None
    if valid_rows:
        keep = lax.broadcasted_iota(jnp.int32, (m, 1), 0) < rank_ref[0, 2]
    if ar2d:
        ar2d_refs, routed_refs = routed_refs[:8], routed_refs[8:]
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

    acc_ref[...] = jnp.zeros((m, h), jnp.float32)
    # K blocks lead xs_ref so each block's scales are one contiguous [m, 128]
    # plane (as in moe_dense_expert.py); K between tokens and lanes relayouts.
    if routed is not None:
        block, tile_rows, lists, store_all, pipeline = routed
        xqf_ref, xqt_f32, xq_t, xs_t, c_t, gm_ref, list_ref, grp_ref = routed_refs[:8]
        if lists != "scan":
            tab_v, tab_s, tab_sem = routed_refs[8:11]
            tab_hbm = routed_refs[11] if lists == "hbm" else None
    for k in range(0, h, _Q_BLOCK):
        x = x_ref[:, k : k + _Q_BLOCK].astype(jnp.float32)
        if keep is not None:
            x = jnp.where(keep, x, 0.0)
        xsc = jnp.max(jnp.abs(x), axis=1, keepdims=True) / qmax
        xq = (x * jnp.where(xsc == 0, 0.0, 1.0 / xsc)).astype(weight_dtype)
        xq_ref[:, k : k + _Q_BLOCK] = xq
        if routed is not None:
            xqf_ref[:, k : k + _Q_BLOCK] = xq.astype(jnp.float32)
        xs_ref[k // _Q_BLOCK] = jnp.broadcast_to(xsc, (m, 128))
    cols = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)
    if routed is not None:
        nblk = m // block
        ngroups = -(-nblk // 16)
        count_row = tab_v.shape[0] - 8 if lists != "scan" else 0
        if lists == "scan":
            for g, flags in enumerate(routed_block_flags(coeff_ref[...], block)):
                gm_ref[g : g + 1, :] = flags
        else:
            # Every expert's block list at once, then one copy into SMEM:
            # tab[s, e] is the s-th block holding a routed row of local expert
            # e (ascending), tab[count_row, e] how many there are.
            routed_block_table(coeff_ref[...], block, tab_v, count_row)
            if lists == "vmem":
                copy = pltpu.make_async_copy(tab_v, tab_s, tab_sem)
                copy.start()
                copy.wait()
            else:
                copy = pltpu.make_async_copy(tab_v, tab_hbm, tab_sem)
                copy.start()
                copy.wait()
                copy = pltpu.make_async_copy(tab_hbm, tab_s, tab_sem)
                copy.start()
                copy.wait()

    def compact(e, count):
        @pl.when(active_ref[e] != 0)
        def record():
            ids_ref[count] = e

        return count + (active_ref[e] != 0).astype(jnp.int32)

    count = lax.fori_loop(0, experts, compact, jnp.int32(0))
    # Ring units: one per active expert, or nblk per active expert (blocked).
    units = count if ib is None else count * nblk

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

    def first_matmul(w1, xq_src=None, xs_src=None):
        """xq @ w1 ([h, 2 width] ref) in f32, each _Q_BLOCK-row slab of w1
        against its column block of the hidden states, scaled by that block's
        row scales before the sum. xq_src / xs_src: a routed tile's rows in
        place of all m rows."""
        xq_src = xq_ref if xq_src is None else xq_src
        xs_src = xs_ref if xs_src is None else xs_src
        acc = None
        for k in range(0, h, _Q_BLOCK):
            part = jnp.matmul(
                xq_src[:, k : k + _Q_BLOCK],
                w1[k : k + _Q_BLOCK, :],
                preferred_element_type=jnp.float32,
            )
            xs = xs_src[k // _Q_BLOCK]
            part = part * jnp.concatenate([xs] * (part.shape[1] // 128), axis=1)
            acc = part if acc is None else acc + part
        return acc

    def accumulate(acc1, coeff, w1_scale, w2, w2_scale, store=None):
        """Finish one expert from its first-matmul result acc1 [rows, 2 width]
        (x scales applied): dequantize, SiLU(gate) * up weighted by coeff
        [rows, 1], quantize per row per _Q_BLOCK columns, and add the second
        matmul with w2 [width, h] into acc_ref (all m rows), or hand each
        _N_CHUNK-column block to store(n, value) (a routed tile)."""
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
            if store is None:
                acc_ref[:, n : n + _N_CHUNK] += acc2 * w2_scale[:, n : n + _N_CHUNK]
            else:
                store(n, acc2 * w2_scale[:, n : n + _N_CHUNK])

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
            term = down.astype(x_ref.dtype).astype(jnp.float32) * coeff
            if keep is not None:
                # A zero row's term is a signed zero, and acc holds +0 there
                # (the shared expert adds first), so +0 is the same sum.
                term = jnp.where(keep, term, 0.0)
            acc_ref[:, n : n + _N_CHUNK] += term

    # Prime the weight ring: start the weight transfers of the first
    # min(units, nb) ring units (active experts, or their blocks) into slots
    # 0 .. nb - 1 now, so they stream from HBM while the shared expert below
    # computes. The expert loop then waits on slot step % nb and refills it
    # with unit step + nb.
    @pl.when(count > 0)
    def prime_routed():
        for initial in range(nb):

            @pl.when(units > initial)
            def prime(*, initial=initial):
                for copy in transfers(jnp.int32(initial)):
                    copy.start()

    # Shared expert first; its weights were issued at entry with the routed primes.
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

    def expert_blocks(step):
        """(local expert id, its block count, block_at(k) -> its k-th block)."""
        e = ids_ref[step]
        if lists != "scan":
            return e, tab_s[count_row, e], lambda k: tab_s[k, e]
        lane = lax.broadcasted_iota(jnp.int32, (1, gm_ref.shape[1]), 1)
        for g in range(ngroups):
            grp_ref[g] = jnp.sum(jnp.where(lane == e, gm_ref[g : g + 1, :], 0.0)).astype(
                jnp.int32
            )

        def collect(b, n_blocks):
            hit = lax.shift_right_logical(grp_ref[b // 16], b % 16) & 1

            @pl.when(hit != 0)
            def record():
                list_ref[n_blocks] = b

            return n_blocks + hit

        return e, lax.fori_loop(0, nblk, collect, jnp.int32(0)), lambda k: list_ref[k]

    def tile_srcs(n_blocks, block_at, t):
        """First rows of tile t's blocks. Slots past the expert's last block
        repeat it; their coefficient is zeroed."""
        tb = tile_rows // block
        return [
            pl.multiple_of(block_at(jnp.minimum(t * tb + j, n_blocks - 1)) * block, block)
            for j in range(tb)
        ]

    def gather(srcs, buf):
        """Copies the blocks' quantized rows, row scales and coefficients into
        tile buffer buf and casts the rows back to weight_dtype (exact)."""
        for j, src in enumerate(srcs):
            rows_j = slice(j * block, (j + 1) * block)
            xqt_f32[buf, rows_j, :] = xqf_ref[pl.ds(src, block), :]
            xs_t[buf, :, rows_j, :] = xs_ref[:, pl.ds(src, block), :]
            c_t[buf, rows_j, :] = coeff_ref[pl.ds(src, block), :]
        xq_t[buf] = xqt_f32[buf].astype(weight_dtype)

    def finish(acc1, slot, e, n_blocks, srcs, t, buf):
        """Accumulates tile t of expert e from its first-matmul result."""
        tb = tile_rows // block
        tile_slot = lax.broadcasted_iota(jnp.int32, (tile_rows, 1), 0) // block
        tile_cols = lax.broadcasted_iota(jnp.int32, (tile_rows, c_t.shape[-1]), 1)
        coeff = jnp.sum(jnp.where(tile_cols == e, c_t[buf], 0.0), axis=1, keepdims=True)
        coeff = jnp.where(t * tb + tile_slot < n_blocks, coeff, 0.0)

        def store(n, value):
            for j, dst in enumerate(srcs):
                if store_all:
                    # A slot past the last block repeats it with coefficient
                    # 0: its rows are exact zeros, and adding them changes no
                    # accumulator element.
                    acc_ref[pl.ds(dst, block), n : n + _N_CHUNK] += value[
                        j * block : (j + 1) * block, :
                    ]
                    continue

                @pl.when(t * tb + j < n_blocks)
                def add(j=j, dst=dst):
                    acc_ref[pl.ds(dst, block), n : n + _N_CHUNK] += value[
                        j * block : (j + 1) * block, :
                    ]

        accumulate(
            acc1, coeff, w1_scale_ring[slot], w2_ring.at[slot], w2_scale_ring[slot], store=store
        )

    def tile_matmul(slot, buf):
        return first_matmul(w1_ring.at[slot], xq_t.at[buf], xs_t.at[buf])

    def expert_tiles(step, e, n_blocks, block_at, first, buf):
        """Tiles first.. of expert `step`, serially, in tile buffer buf."""
        slot = step % nb
        tb = tile_rows // block

        def tile(t, carry):
            srcs = tile_srcs(n_blocks, block_at, t)
            gather(srcs, buf)
            finish(tile_matmul(slot, buf), slot, e, n_blocks, srcs, t, buf)
            return carry

        lax.fori_loop(first, (n_blocks + tb - 1) // tb, tile, 0)

    @pl.when(count > 0)
    def run_routed():
        for copy in transfers(jnp.int32(0)):
            copy.wait()

        if routed is not None and not pipeline:

            def routed_body(step, carry):
                @pl.when(step > 0)
                def wait_current():
                    for copy in transfers(step):
                        copy.wait()

                e, n_blocks, block_at = expert_blocks(step)
                expert_tiles(step, e, n_blocks, block_at, 0, 0)

                @pl.when(step + nb < units)
                def refill():
                    for copy in transfers(step + nb):
                        copy.start()

                return carry

            lax.fori_loop(0, units, routed_body, 0)
            return

        if routed is not None:
            # As the dense loop: the next expert's first tile is gathered and
            # its first matmul issued before this expert's accumulate, so the
            # MXU and the vector unit overlap across experts. Tile buffers
            # alternate between experts; this expert's further tiles reuse its
            # own buffer after its first tile.
            e0, n0, at0 = expert_blocks(jnp.int32(0))
            gather(tile_srcs(n0, at0, 0), 0)

            def piped_body(step, acc1):
                slot, buf = step % nb, step % 2
                nxt = step + 1

                @pl.when(nxt < units)
                def wait_next():
                    for copy in transfers(nxt):
                        copy.wait()

                e_n, n_n, at_n = expert_blocks(jnp.minimum(nxt, units - 1))
                gather(tile_srcs(n_n, at_n, 0), nxt % 2)
                acc1_next = tile_matmul(nxt % nb, nxt % 2)
                e, n_blocks, block_at = expert_blocks(step)
                finish(acc1, slot, e, n_blocks, tile_srcs(n_blocks, block_at, 0), 0, buf)
                expert_tiles(step, e, n_blocks, block_at, 1, buf)

                @pl.when(step + nb < units)
                def refill():
                    for copy in transfers(step + nb):
                        copy.start()

                return acc1_next

            lax.fori_loop(0, units, piped_body, tile_matmul(0, 0))
            return

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

    def all_to_all_sum(send_slice, dst):
        """Reduce-scatter semantics by a direct all-to-all and a local sum,
        which both output modes end with. send_slice(d) is this rank's partial
        of the slice rank d keeps; one remote copy per peer sends it straight
        to rank d's recv[me]. Each rank then sums the n slots it received in FP32
        into dst. There is no ring and no per-hop reduction: a single hop of n
        concurrent copies, where a ring needs n - 1 dependent steps for the
        same bytes, which suits the latency-bound sizes of decode."""
        # Every peer has entered this call, so its receive buffer is live.
        pl.semaphore_wait(barrier, n_ranks)
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

    if ar2d:
        _ar2d_exchange(
            me,
            barrier,
            n_ranks,
            acc_ref,
            acc_bf16,
            recv,
            red,
            out_ref,
            a2a_send,
            a2a_recv,
            ar2d_refs,
            m,
            h,
            ar2d_delay_ns,
            ar2d_partial_ref,
        )
        return

    # TP attention, all-reduce: all-to-all + local sum over column chunks (rank
    # d keeps columns [d * chunk, (d + 1) * chunk) of every token), then an
    # all-gather of the reduced chunks.
    if exact_width and h // 128 >= n_ranks:
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
        pl.semaphore_wait(barrier, n_ranks)
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
                # The receive wait counts the n slots of this rank's width.
                received = recv.at[:, :, pl.ds(0, width)]
                pltpu.make_async_copy(received, received, a2a_recv).wait()
                acc = recv[0, :, :width].astype(jnp.float32)
                for s in range(1, n_ranks):
                    acc = acc + recv[s, :, :width].astype(jnp.float32)
                red[:, :width] = acc.astype(red.dtype)
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


# The AR2D exchange (_ar2d_exchange): its streams, each with its own send and
# receive semaphore, and the one shape it serves.
AR2D_STREAMS = ("s1", "r1x", "r1y", "r2y", "r2x", "r3y", "r3x", "r4x", "r4y", "s3")
_AR2D_RANKS = 8
_AR2D_HIDDEN = 2560
# Test-only skew: pl.delay sleeps at most about 0.5 ms a call on tpu7x
# (j-dac0e9c1), so longer delays are slept in steps.
_DELAY_STEP_NS = 50_000


def ar2d_applies(m, h, n_ranks, min_tokens, scatter_rows=False, whole_experts=True):
    """Whether an m-row call (m padded to 16) runs the AR2D exchange in place of
    the all-to-all + all-gather: FUSED_MOE_DECODE_KERNEL_AR2D_MIN_TOKENS set and
    at most m, 8 ranks, hidden 2560, the TP all-reduce output, whole experts,
    and m / 4 a multiple of 16 so every copied region is whole bf16 tiles."""
    return (
        0 < min_tokens <= m
        and n_ranks == _AR2D_RANKS
        and h == _AR2D_HIDDEN
        and not scatter_rows
        and whole_experts
        and m % 64 == 0
    )


def _ar2d_scratch(m, h, wire):
    """The AR2D exchange's buffers beyond acc_bf16 [m, h], the s1 receive
    [m, h / 2] and AF [m, h / 2]: r1x, r1y, B0, B1 [m / 2, h / 4], r2y, r2x
    [m / 4, h / 4], B0f, B1f [m / 2, h / 4]. Every receive region is its own
    storage."""
    part = h // 4
    half_rows, quarter_rows = (m // 2, part), (m // 4, part)
    shapes = [half_rows] * 4 + [quarter_rows] * 2 + [half_rows] * 2
    return [pltpu.VMEM(s, wire) for s in shapes]


def _ar2d_exchange(
    me, barrier, n_ranks, acc_ref, a, s1_recv, af, out_ref, ssem, rsem, bufs, m, h,
    delay_ns, partial_ref,
):
    """The TP all-reduce of acc_ref [m, h] f32 into out_ref[:, :h] over the
    chip mate (me ^ 1) and the two neighbours (me ^ 2, me ^ 4), me the mesh
    index. Core c = me & 1 owns column half c; part 0 of the half (h / 4
    columns) reduces across me ^ 4 then me ^ 2, part 1 across me ^ 2 then
    me ^ 4, each over row halves then quarters, and gathers back in reverse:

      s1  bf16 of the mate's half to me ^ 1; A = own half (f32) + received,
          f32, in place in acc_ref; bf16(A) into a's own half, the r1 source
      r1  A0 rows half (1 - xb) to me ^ 4, A1 rows half (1 - yb) to me ^ 2;
          B0 = A0 rows half xb (f32) + received, f32, in place in acc_ref,
          bf16(B0) into B0 (the r2 source); B1 likewise
      r2  B0 quarter (1 - yb) to me ^ 2, B1 quarter (1 - xb) to me ^ 4;
          C0 = bf16(B0 quarter yb (f32) + received) into B0f, C1 into B1f
      r3  C0 to me ^ 2's B0f, C1 to me ^ 4's B1f, at this rank's quarter
      r4  B0f to me ^ 4's AF part 0, B1f to me ^ 2's AF part 1, at this rank's
          half; own B0f, B1f into own AF
      s3  AF to me ^ 1's output at this rank's half; own half stored locally

    At every sum the rank's own operand is its f32 value and only the operand
    that crossed a wire was rounded to bf16; the sum is rounded once, at C.
    Each sum is made by one rank and copied to the others, so all ranks hold
    the same bits (model: ar2d_reference.py). Each copy has its
    own semaphores and every receive region its own storage, so a copy from a
    rank that is ahead lands where nothing else writes and each wait is met by
    the one copy it names. Test-only: delay_ns, a per-rank delay (ns) before
    the exchange, after the entry signals, slept in steps of at most
    _DELAY_STEP_NS; partial_ref [m, h] f32, where the rank's accumulator (the
    exchange's input) is stored."""
    r1x, r1y, b0, b1, r2y, r2x, b0f, b1f = bufs
    f32, bf16 = jnp.float32, a.dtype
    half, part = h // 2, h // 4
    rh, rq = m // 2, m // 4
    c, yb, xb = me % 2, (me // 2) % 2, me // 4
    idx = {name: k for k, name in enumerate(AR2D_STREAMS)}

    def copy(src, dst, name, peer):
        return pltpu.make_async_remote_copy(
            src,
            dst,
            ssem.at[idx[name]],
            rsem.at[idx[name]],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def rows(bit, n):
        return pl.ds(pl.multiple_of(bit * n, 16), n)

    def add(x, y):
        return (x.astype(f32) + y.astype(f32)).astype(bf16)

    def per_half(fn):
        # Vector slices need static columns: one branch per column owner.
        for cc in (0, 1):
            pl.when(c == cc)(functools.partial(fn, cc * half))

    if delay_ns is not None:
        steps, rest = jnp.int32(0), jnp.int32(0)
        for d, t in enumerate(delay_ns):
            steps = jnp.where(me == d, jnp.int32(t // _DELAY_STEP_NS), steps)
            rest = jnp.where(me == d, jnp.int32(t % _DELAY_STEP_NS), rest)

        def sleep(i, carry):
            pl.delay(jnp.int32(_DELAY_STEP_NS))
            return carry

        lax.fori_loop(0, steps, sleep, 0)
        pl.delay(rest)
    # Every peer has entered this call, so its receive buffers are live.
    pl.semaphore_wait(barrier, n_ranks)
    a[...] = acc_ref[...].astype(bf16)
    if partial_ref is not None:
        partial_ref[...] = acc_ref[...]
    own = pl.multiple_of(c * half, 128)
    mate = pl.multiple_of((1 - c) * half, 128)

    d_s1 = copy(a.at[:, pl.ds(mate, half)], s1_recv, "s1", me ^ 1)
    d_s1.start()
    d_s1.wait_recv()

    def form_a(c0):
        # acc_ref is free once its bf16 copy exists: no copy reads or targets it.
        v = acc_ref[:, c0 : c0 + half] + s1_recv[...].astype(f32)
        acc_ref[:, c0 : c0 + half] = v
        a[:, c0 : c0 + half] = v.astype(bf16)

    per_half(form_a)

    d_r1 = [
        copy(a.at[rows(1 - xb, rh), pl.ds(own, part)], r1x, "r1x", me ^ 4),
        copy(a.at[rows(1 - yb, rh), pl.ds(own + part, part)], r1y, "r1y", me ^ 2),
    ]
    for d in d_r1:
        d.start()
    for d in d_r1:
        d.wait_recv()

    def form_b(c0):
        v0 = acc_ref[rows(xb, rh), c0 : c0 + part] + r1x[...].astype(f32)
        acc_ref[rows(xb, rh), c0 : c0 + part] = v0
        b0[...] = v0.astype(bf16)
        v1 = acc_ref[rows(yb, rh), c0 + part : c0 + half] + r1y[...].astype(f32)
        acc_ref[rows(yb, rh), c0 + part : c0 + half] = v1
        b1[...] = v1.astype(bf16)

    per_half(form_b)

    d_r2 = [
        copy(b0.at[rows(1 - yb, rq)], r2y, "r2y", me ^ 2),
        copy(b1.at[rows(1 - xb, rq)], r2x, "r2x", me ^ 4),
    ]
    for d in d_r2:
        d.start()
    for d in d_r2:
        d.wait_recv()
    def form_c(c0):
        # The quarter kept is at rows xb * rh + yb * rq of acc_ref (part 0) and
        # yb * rh + xb * rq (part 1).
        q0 = pl.ds(pl.multiple_of(xb * rh + yb * rq, 16), rq)
        q1 = pl.ds(pl.multiple_of(yb * rh + xb * rq, 16), rq)
        b0f[rows(yb, rq), :] = add(acc_ref[q0, c0 : c0 + part], r2y[...])
        b1f[rows(xb, rq), :] = add(acc_ref[q1, c0 + part : c0 + half], r2x[...])

    per_half(form_c)

    d_r3 = [
        copy(b0f.at[rows(yb, rq)], b0f.at[rows(yb, rq)], "r3y", me ^ 2),
        copy(b1f.at[rows(xb, rq)], b1f.at[rows(xb, rq)], "r3x", me ^ 4),
    ]
    for d in d_r3:
        d.start()
    for d in d_r3:
        d.wait_recv()

    d_r4 = [
        copy(b0f, af.at[rows(xb, rh), pl.ds(0, part)], "r4x", me ^ 4),
        copy(b1f, af.at[rows(yb, rh), pl.ds(part, part)], "r4y", me ^ 2),
    ]
    for d in d_r4:
        d.start()
    af[rows(xb, rh), :part] = b0f[...]
    af[rows(yb, rh), part:] = b1f[...]
    for d in d_r4:
        d.wait_recv()

    d_s3 = copy(af, out_ref.at[:, pl.ds(own, half)], "s3", me ^ 1)
    d_s3.start()

    def store_own(c0):
        out_ref[:, c0 : c0 + half] = af[...]

    per_half(store_own)
    d_s3.wait_recv()
    for d in (d_s1, *d_r1, *d_r2, *d_r3, *d_r4, d_s3):
        d.wait_send()


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
    ar2d_partials=False,
    valid_rows=False,
    **kwargs,
):
    """_kernel behind the routing prologue the unfolded call runs in XLA, so
    the call takes the router's outputs and the hidden states as they are.

    Scalar prefetch: rank_ref [1, 1 or 2] int32: this rank's mesh index and,
    with expert parallelism, its first expert's global id. With valid_rows,
    [1, 3]: then rank_ref[0, 2] is the count n of real rows, and every row from
    n on enters as the unfolded call's padded rows do: zero hidden states and
    zero route weights, whatever x_ref, tw_ref and ti_ref hold there.
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
    # The routed mode's "hbm" block lists take a second, HBM output as the
    # staging buffer of their copy into SMEM.
    routed = kwargs.get("routed")
    staged = routed is not None and routed[2] == "hbm"
    hbm, out_ref = refs[:8], refs[8]
    if ar2d_partials:
        # Test-only second output: the rank's bf16 partial (see _ar2d_exchange).
        kwargs["ar2d_partial_ref"], refs = refs[9], refs[:9] + refs[10:]
    tab_hbm, scratch = (refs[9], refs[10:]) if staged else (None, refs[9:])
    pad = m - tokens
    if pad:
        *scratch, coeff_ref, shared_gate_ref, xpad_ref = scratch
    else:
        *scratch, coeff_ref, shared_gate_ref = scratch
    if staged:
        scratch = [*scratch, tab_hbm]
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
    # With valid_rows, which of the m rows are real.
    keep = None
    if valid_rows:
        n_valid = rank_ref[0, 2]
        keep = lax.broadcasted_iota(jnp.int32, (m, 1), 0) < n_valid
        w = jnp.where(lax.broadcasted_iota(jnp.int32, (tokens, 1), 0) < n_valid, w, 0.0)
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
    if keep is not None and gate is not None:
        # sigmoid(0), the gate of a zero row. Any finite positive gate gives
        # the same output: the row's shared expert output is a signed zero.
        g = jnp.where(keep, g, 0.5)
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
        valid_rows=valid_rows,
        **kwargs,
    )


def routed_block_flags(coeff, block):
    """Which aligned `block`-row blocks of coeff [m, lanes] hold a nonzero
    coefficient, per lane (local expert), as groups of 16 flag bits: one
    [1, lanes] f32 row per group g, block 16 g + j at bit j. A row whose
    route weights are all 0 (a padded row) sets no flag, so a block of padded
    rows only is never scheduled. Sums of distinct powers of two below 2**16
    are exact in f32."""
    m, lanes = coeff.shape
    nblk = m // block
    out = []
    for g in range(-(-nblk // 16)):
        n = min(16, nblk - 16 * g)
        rows = coeff[16 * g * block : (16 * g + n) * block, :]
        nz = jnp.where(rows != 0, 1.0, 0.0)
        if block > 1:
            nz = jnp.max(nz.reshape(n, block, lanes), axis=1)
        bit = jnp.left_shift(
            jnp.int32(1), lax.broadcasted_iota(jnp.int32, (n, 1), 0)
        ).astype(jnp.float32)
        out.append(jnp.sum(nz * bit, axis=0, keepdims=True))
    return out


def routed_block_table(coeff, block, tab_ref, count_row):
    """Writes, per lane e (local expert) of coeff [m, lanes], the aligned
    `block`-row blocks holding a nonzero coefficient: tab_ref[s, e] the s-th
    such block in ascending order (s below the count), tab_ref[count_row, e]
    their count. Rows with route weights 0 (padding) add no block."""
    m, lanes = coeff.shape
    nblk = m // block
    nz = jnp.where(coeff != 0, 1.0, 0.0)
    bm = jnp.max(nz.reshape(nblk, block, lanes), axis=1) if block > 1 else nz
    hit = bm > 0
    b_ids = lax.broadcasted_iota(jnp.int32, (nblk, lanes), 0)
    counts = jnp.sum(hit.astype(jnp.int32), axis=0, keepdims=True)
    tab_ref[count_row : count_row + 1, :] = counts
    big = jnp.int32(nblk)

    def slot(s, prev):
        nxt = jnp.min(jnp.where(hit & (b_ids > prev), b_ids, big), axis=0, keepdims=True)
        tab_ref[pl.ds(s, 1), :] = nxt
        return nxt

    # Up to the largest count only: no lane reads a slot at or past its count.
    lax.fori_loop(0, jnp.max(counts), slot, jnp.full((1, lanes), -1, jnp.int32))


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


def _token_side_bytes(m, h, i_s, experts, n_ranks, wt, ar2d=False):
    """Everything the call declares but the routed weight ring: the shared
    expert (counted at 2 bytes per weight, its widest dtype) and the
    token-proportional buffers. ar2d: the exchange's buffers are the AR2D
    exchange's (see _ar2d_scratch) instead of the all-to-all's."""
    f32, bf16 = 4, 2
    shared = _unit_bytes(h, i_s, max(wt, bf16))
    hp = _padded_hidden(h, n_ranks)
    if ar2d:
        part = h // 4
        exchange = (
            h * bf16  # accumulator in bf16, A formed in place
            + (h // 2) * bf16  # s1 receive
            + (h // 2) * bf16  # AF, the s3 source
            + 2 * part * bf16 // 2  # r1 receives
            + 2 * part * bf16 // 2  # B0, B1
            + 2 * part * bf16 // 4  # r2 receives
            + 2 * part * bf16 // 2  # B0f, B1f
        )
    else:
        exchange = (
            hp * bf16  # accumulator in bf16, the all-reduce source
            + hp * bf16  # all-to-all receive slots
            + (hp // n_ranks) * bf16  # reduced chunk
        )
    rows = m * (
        h * bf16  # hidden
        + (experts + (-experts) % 128) * f32  # coeff
        + 128 * f32  # shared gate
        + hp * bf16  # output
        + h * wt  # quantized hidden
        + (h // _Q_BLOCK) * 128 * f32  # per-block row scales, over lanes
        + h * f32  # accumulator
        + exchange
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
        # The all-to-all's count even under AR2D, which only runs with whole
        # experts: the plan never depends on the exchange.
        free = (
            capacity
            - _VMEM_RESERVE
            - live
            - _token_side_bytes(m, h, i_s, experts, n_ranks, wt)
        )
        return free // _unit_bytes(h, w, wt)

    ib = next((w for w in widths if fit(w) >= 3), widths[-1])
    return ib, max(3, min(_NUM_BUFFERS, fit(ib)))


def _routed_bytes(m, h, tile_rows, wt, tile_buffers=1):
    """The routed mode's extra scratch (see _kernel): the f32 gather source,
    the tile buffers, and the block flags and lists."""
    tile = tile_rows * h * (4 + wt) + (h // _Q_BLOCK) * tile_rows * 128 * 4 + tile_rows * 128 * 4
    return m * h * 4 + tile_buffers * tile + 2 * (m // 8 + 16) * 128 * 4


# Rows per block of the routed mode: the unit it copies into a tile and adds
# back into the accumulator, one f32 sublane tile.
ROUTED_BLOCK = 8


def routed_tile_rows(m):
    """Tile height of the routed mode for an m-row call. 64 rows (8 blocks)
    measured fastest at every bucket from 208 to 512 rows on AgentX routing
    (32 and 128 slower: more passes of the weights, or more rows computed)."""
    return 64


def vmem_bytes(
    tokens, h, i, i_s, experts, n_ranks, weight_dtype, routed=None, ar2d_min_tokens=0
):
    """VMEM the call declares for `tokens` rows: its VMEM operands, output and
    scratch (compiler temporaries excluded). The weight ring takes about half
    of VMEM, so the token-proportional buffers bound the rows one call takes.
    routed: the (block, tile_rows) the call runs with, None for dense.
    ar2d_min_tokens: FUSED_MOE_DECODE_KERNEL_AR2D_MIN_TOKENS (see ar2d_applies)."""
    m = tokens + (-tokens) % 16
    wt = jnp.dtype(weight_dtype).itemsize
    ib, nb = plan_weight_ring(m, h, i, i_s, experts, n_ranks, weight_dtype)
    ar2d = ar2d_applies(m, h, n_ranks, ar2d_min_tokens, whole_experts=ib is None)
    extra = (
        0
        if routed is None
        else _routed_bytes(m, h, routed[1], wt, 2 if len(routed) > 4 and routed[4] else 1)
    )
    return (
        nb * _unit_bytes(h, i if ib is None else ib, wt)
        + _token_side_bytes(m, h, i_s, experts, n_ranks, wt, ar2d)
        + extra
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
    routed=None,
    ar2d_min_tokens=0,
    ar2d_delay_ns=None,
    ar2d_partials=False,
    exact_width=False,
    valid_rows=False,
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
      routed: (block, tile_rows) to run each active expert only over the
        aligned `block`-row blocks that hold its routed rows (see _kernel);
        bitwise the dense output. Needs fold, whole experts and the TP
        all-reduce output; block divides the padded row count and tile_rows,
        and tile_rows is a multiple of 32. None: dense.
      ar2d_min_tokens: FUSED_MOE_DECODE_KERNEL_AR2D_MIN_TOKENS. Calls that
        ar2d_applies accepts run the TP all-reduce as _ar2d_exchange instead of
        the all-to-all + all-gather; the output differs from it in rounding
        (ar2d_reference.py), is identical on every rank, and the plan is the
        same. 0: never.
      ar2d_delay_ns: test-only, 8 per-rank delays in ns before the AR2D
        exchange (skew injection); needs an AR2D call.
      ar2d_partials: test-only, needs fold and an AR2D call: also return this
        rank's f32 accumulator [m, H] (all padded rows), the exchange's input,
        as a second output.
      exact_width: TP all-reduce only: the exchange splits H's lane tiles over
        the ranks as evenly as whole tiles allow (no rank sends padding, the
        same columns in the all-to-all and the all-gather) and the output is
        [m, H] instead of [m, H padded to n_ranks lanes of 128] sliced
        afterwards. Bitwise the same values. Needs H // 128 >= n_ranks;
        smaller H keeps the padded split.
      valid_rows: FUSED_MOE_DECODE_KERNEL_PADDING_MASK, needs fold. `rank` is
        [1, 3] and rank[0, 2] the count n of real rows: rows from n on are
        computed as zero hidden states with zero route weights, so the caller
        passes them unmasked (they may hold any bits, NaN included) and the
        output is bitwise the output for the masked inputs.

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
    if routed is not None:
        if not fold or rows is not None or i_block is not None:
            raise ValueError("routed needs fold, the all-reduce output and whole experts")
    if valid_rows and not fold:
        raise ValueError("valid_rows needs fold")
    if fold:
        expert_parallel = expert_start is not None
        if rows is not None:
            raise ValueError("fold does not support scatter_rows")
        if expert_parallel and expert_start != "rank":
            raise ValueError('fold with expert parallelism needs expert_start="rank"')
        if rank.ndim != 2 or rank.shape[1] < 1 + expert_parallel:
            raise ValueError(f"fold needs rank [1, 1 or 2]: {rank.shape}")
        if valid_rows and rank.shape[1] != 3:
            raise ValueError(f"valid_rows needs rank [1, 3]: {rank.shape}")
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
    if routed is not None:
        routed = tuple(routed) + ("table", True, False)[len(routed) - 2 :]
        block, tile_rows, lists, store_all, pipeline = routed
        if ib is not None or m % block or tile_rows % block or tile_rows % 32:
            raise ValueError(f"routed {routed} does not fit m={m} or blocked mode {ib}")
        if lists == "table":
            lists = "vmem"
            routed = (block, tile_rows, lists, store_all, pipeline)
        if lists not in ("scan", "vmem", "hbm") or (pipeline and lists == "scan"):
            raise ValueError(f"routed lists {lists!r} (pipeline {pipeline})")
    w = i if ib is None else ib  # ring slot width
    hp = _padded_hidden(h, n_ranks)
    chunk = hp // n_ranks
    out_cols = h
    ar2d = ar2d_applies(m, h, n_ranks, ar2d_min_tokens, rows is not None, ib is None)
    if ar2d_delay_ns is not None and (not ar2d or len(ar2d_delay_ns) != n_ranks):
        raise ValueError(f"ar2d_delay_ns needs an AR2D call and {n_ranks} delays")
    if ar2d_partials and not (ar2d and fold):
        raise ValueError("ar2d_partials needs fold and an AR2D call")
    n_sem = len(AR2D_STREAMS) if ar2d else n_ranks
    if rows is None:
        exact_width = exact_width and h // 128 >= n_ranks
    if ar2d:
        out_rows, wire, out_cols = m, hidden.dtype, h if exact_width else hp
        acc_bf16_shape, recv_shape, red_shape = (m, h), (m, h // 2), (m, h // 2)
    elif rows is None:
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
        pltpu.SemaphoreType.DMA((n_sem,)),
        pltpu.SemaphoreType.DMA((n_sem,) if ar2d else ()),
        pltpu.SemaphoreType.DMA((n_ranks,)),
        pltpu.SemaphoreType.DMA(()),
    ]
    if ar2d:
        scratch_shapes += _ar2d_scratch(m, h, wire)
    if routed is not None:
        ngroups = -(-(m // block) // 16)
        nbt = 2 if pipeline else 1  # tile buffers
        scratch_shapes += [
            pltpu.VMEM((m, h), jnp.float32),
            pltpu.VMEM((nbt, tile_rows, h), jnp.float32),
            pltpu.VMEM((nbt, tile_rows, h), weight_dtype),
            pltpu.VMEM((nbt, h // _Q_BLOCK, tile_rows, 128), jnp.float32),
            pltpu.VMEM((nbt, tile_rows, experts + (-experts) % 128), jnp.float32),
            pltpu.VMEM((-(-ngroups // 8) * 8, experts + (-experts) % 128), jnp.float32),
            pltpu.SMEM((m // block,), jnp.int32),
            pltpu.SMEM((ngroups,), jnp.int32),
        ]
        if lists != "scan":
            tab_rows = -(-(m // block) // 8) * 8 + 8
            lanes_e = experts + (-experts) % 128
            scratch_shapes += [
                pltpu.VMEM((tab_rows, lanes_e), jnp.int32),
                pltpu.SMEM((tab_rows, lanes_e), jnp.int32),
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
    )
    if routed is not None:
        kernel_kwargs["routed"] = tuple(routed)
    if ar2d:
        kernel_kwargs["ar2d"] = True
        kernel_kwargs["ar2d_delay_ns"] = (
            None if ar2d_delay_ns is None else tuple(int(t) for t in ar2d_delay_ns)
        )
    if exact_width and rows is None:
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
            ar2d_partials=ar2d_partials,
            valid_rows=valid_rows,
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
    routed_tag = (
        ""
        if routed is None
        else f"-rb_{routed[0]}-rt_{routed[1]}-rl_{routed[2]}"
        + ("-sa" if routed[3] else "")
        + ("-pp" if routed[4] else "")
    )
    out_shape = jax.ShapeDtypeStruct((out_rows, out_cols), wire)
    out_specs = pl.BlockSpec(memory_space=pltpu.VMEM)
    staged = routed is not None and routed[2] == "hbm"
    outs, specs = [out_shape], [out_specs]
    if ar2d_partials:
        outs.append(jax.ShapeDtypeStruct((m, h), jnp.float32))
        specs.append(pl.BlockSpec(memory_space=pltpu.VMEM))
    if staged:
        outs.append(jax.ShapeDtypeStruct((tab_rows, lanes_e), jnp.int32))
        specs.append(pl.BlockSpec(memory_space=pltpu.HBM))
    if len(outs) > 1:
        out_shape, out_specs = tuple(outs), tuple(specs)
    out = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * len(vmem_operands)
            + [pl.BlockSpec(memory_space=pltpu.HBM)] * 8,
            out_specs=out_specs,
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
        f"{routed_tag}{'-ar2d' if ar2d else ''}{'-vr' if valid_rows else ''}",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(*operands)
    partials = None
    if ar2d_partials:
        partials = out[1]
    if staged or ar2d_partials:
        out = out[0]
    if partials is not None:
        return (out[:, :h][:tokens], partials)
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
