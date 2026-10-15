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
    *,
    m,
    h,
    i,
    experts,
    nb,
    n_ranks,
    weight_dtype,
    rows=None,
    ib=None,
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
    """
    nblk = 1 if ib is None else i // ib
    me = rank_ref[0]
    qmax = float(jnp.finfo(weight_dtype).max)
    chunk = h // n_ranks
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
        qa = min(_Q_BLOCK, width)
        blocks = []
        for k in range(0, width, qa):
            blk = mid[:, k : k + qa]
            ms = jnp.max(jnp.abs(blk), axis=1, keepdims=True) / qmax
            blocks.append(
                ((blk * jnp.where(ms == 0, 0.0, 1.0 / ms)).astype(weight_dtype), ms)
            )
        for n in range(0, h, _N_CHUNK):
            acc2 = None
            for b, (mq, ms) in enumerate(blocks):
                part = jnp.matmul(
                    mq,
                    w2[b * qa : (b + 1) * qa, n : n + _N_CHUNK],
                    preferred_element_type=jnp.float32,
                )
                part = part * ms
                acc2 = part if acc2 is None else acc2 + part
            acc_ref[:, n : n + _N_CHUNK] += acc2 * w2_scale[:, n : n + _N_CHUNK]

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
    accumulate(
        first_matmul(shared_w1_vmem),
        shared_coeff,
        shared_w1_scale_vmem[...],
        shared_w2_vmem,
        shared_w2_scale_vmem[...],
    )

    @pl.when(count > 0)
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

    # TP attention, all-reduce: all-to-all + local sum over column chunks (rank
    # d keeps columns [d * chunk, (d + 1) * chunk) of every token), then an
    # all-gather of the reduced chunks.
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


def _weight_ring_depth(h, i, weight_dtype):
    """Deepest weight ring (at most _NUM_BUFFERS slots, at least 3) whose slots
    fit in half of VMEM; one slot holds an expert's w1 and w2."""
    slot = 3 * h * i * jnp.dtype(weight_dtype).itemsize
    fit = pltpu.get_tpu_info().vmem_capacity_bytes // 2 // slot
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
    expert and the token-proportional buffers."""
    f32, bf16 = 4, 2
    shared = _unit_bytes(h, i_s, wt)
    rows = m * (
        h * bf16  # hidden
        + (experts + (-experts) % 128) * f32  # coeff
        + 128 * f32  # shared gate
        + h * bf16  # output
        + h * wt  # quantized hidden
        + (h // _Q_BLOCK) * 128 * f32  # per-block row scales, over lanes
        + h * f32  # accumulator
        + h * bf16  # accumulator in bf16, the all-reduce source
        + n_ranks * (h // n_ranks) * bf16  # all-to-all receive slots
        + (h // n_ranks) * bf16  # reduced chunk
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
    capacity = pltpu.get_tpu_info().vmem_capacity_bytes
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
      shared_w1: [H, 2 i_s] the shared expert's gate and up slice (i_s <= i).
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
    if pad:
        hidden = jnp.pad(hidden, ((0, pad), (0, 0)))
        topk_weights = jnp.pad(topk_weights, ((0, pad), (0, 0)))
        topk_ids = jnp.pad(topk_ids, ((0, pad), (0, 0)))
    m = tokens + pad
    if expert_start is not None:
        topk_ids = topk_ids - expert_start
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
    i_s = shared_w2.shape[0]
    weight_dtype = w1.dtype
    ib, nb = plan_weight_ring(m, h, i, i_s, experts, n_ranks, weight_dtype, i_block)
    w = i if ib is None else ib  # ring slot width
    chunk = h // n_ranks
    if rows is None:
        out_rows, wire = m, hidden.dtype
        acc_bf16_shape, recv_shape, red_shape = (m, h), (n_ranks, m, chunk), (m, chunk)
    else:
        # BF16 packs 16 rows per tile; smaller send slices go over the wire in
        # FP32.
        wire = hidden.dtype if rows % 16 == 0 else jnp.float32
        out_rows = rows if rows % 16 == 0 else -(-rows // 8) * 8
        acc_bf16_shape = recv_shape = (n_ranks, out_rows, h)
        red_shape = (8, 128)  # unused
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            m=m,
            h=h,
            i=i,
            experts=experts,
            nb=nb,
            n_ranks=n_ranks,
            weight_dtype=weight_dtype,
            rows=rows,
            ib=ib,
        ),
        out_shape=jax.ShapeDtypeStruct((out_rows, h), wire),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 3
            + [pl.BlockSpec(memory_space=pltpu.HBM)] * 8,
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((nb, h, 2 * w), weight_dtype),
                pltpu.VMEM((nb, w, h), weight_dtype),
                pltpu.VMEM((nb, 1, 2 * w), jnp.float32),
                pltpu.VMEM((nb, 1, h), jnp.float32),
                pltpu.VMEM((h, 2 * i_s), weight_dtype),
                pltpu.VMEM((1, 2 * i_s), jnp.float32),
                pltpu.VMEM((i_s, h), weight_dtype),
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
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name=f"fused_moe_decode{'_rs' if rows else ''}-e_{experts}-m_{m}-h_{h}"
        f"-i_{i}-is_{i_s}-n_{n_ranks}-buffers_{nb}{f'-ib_{ib}' if ib else ''}",
    )(
        active,
        rank.reshape(1).astype(jnp.int32),
        hidden,
        coeff,
        shared_gate,
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
    )
    if rows is not None:
        return out[:rows].astype(hidden.dtype)
    return out[:tokens] if pad else out
