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
"""The moe_lowtok decode layer with its DP dispatch and EP combine inside the kernel.

On a DP x TP expert-parallel decode cell the routed MoE layer is an all-gather of
the engine rows and router logits over the DP axis, the moe_lowtok call over the
rank's experts, a DP reduce-scatter of the partial sums and a TP all-reduce, each
its own program. This entry runs the collectives as remote DMAs between the cores
of the (dp, tp) mesh inside one Pallas call: the rows and logits are pushed to
the DP peers in the prologue while the expert weight ring primes, and the partial
sums are reduced onto their owning engine in the epilogue. The expert loop is
moe_lowtok's, with a weight ring of `depth` experts.

Forms, per side:

  dispatch "flat"   every core pushes its rows to the DP peers of its TP index
                    (dp - 1 inter-chip DMAs of the full rows per core);
  dispatch "hier"   each core pushes one column half of its rows to its DP peers
                    and the TP pair exchanges the halves over the intra-chip
                    link (half the inter-chip bytes; tp == 2);
  combine "flat"    the foreign row blocks are pushed to their owners and summed
                    there, then the TP pair exchanges its block and sums;
  combine "hier"    the TP pair reduce-scatters column halves intra-chip, each
                    core pushes its half of the foreign blocks to the owners,
                    and the pair exchanges the finished halves (half the
                    inter-chip bytes; tp == 2).

Every stage sums in float32 and rounds once; the two cores of a TP pair end with
the same bytes. The caller runs the entry inside a shard_map over a mesh whose
axes are ("dp", "tp") in that order, with TP pairs on one chip.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.moe_lowtok.kernel import (
    _FP8,
    _LANE,
    _dot,
    _layout,
    _quantize_rows,
    _select_topk,
)

# Rows per gathered block of the gather formulation: one fp8 sublane tile. An
# expert routed more rows runs several passes over the same resident weights.
GATHER_ROWS = 32
# The scatter-back carries each f32 contribution as three bf16 pieces stacked
# along the contraction dim, so one matmul reconstructs it exactly.
_PIECES = 3

# Barrier semaphore group; the small collectives use 5 and 7, the fused EP
# kernels 0.
COLLECTIVE_ID = 9
MESH = pl.DeviceIdType.MESH
DP_AXIS = "dp"
TP_AXIS = "tp"
# Row blocks are pushed by DMA and sliced by rows; bf16 tiles are 16 sublanes.
_ROW_TILE = 16
_VMEM_FRACTION = 0.92
_VMEM_MARGIN = 1.3
FORMS = ("flat", "hier")
SIDES = ("xla", "inkernel")


def _push(src, dst, send_sem, recv_sem, dp_idx, tp_idx):
    pltpu.make_async_remote_copy(
        src_ref=src,
        dst_ref=dst,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=(dp_idx, tp_idx),
        device_id_type=MESH,
    ).start()


def _wait_bytes(ref_slice, sem):
    """Block until `sem` has counted the bytes of `ref_slice`."""
    pltpu.make_async_copy(ref_slice, ref_slice, sem).wait()


def _barrier(dp, tp):
    """Every core signals every core once and waits for all of them.

    A core enters a call only after every peer has finished the previous one,
    so the scratch a peer pushes into is never still being read.
    """
    sem = pltpu.get_barrier_semaphore()
    for j in range(dp):
        for u in range(tp):
            pl.semaphore_signal(
                sem, 1, device_id=(jnp.int32(j), jnp.int32(u)), device_id_type=MESH
            )
    pl.semaphore_wait(sem, dp * tp)


def _rows_of(d, rows_l):
    if isinstance(d, int):
        return pl.ds(d * rows_l, rows_l)
    return pl.ds(pl.multiple_of(d * rows_l, _ROW_TILE), rows_l)


def _stream_ring(
    gid_s,
    hits_s,
    wtab_ref,
    x_all,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    acc,
    ids,
    w13_buf,
    w2_buf,
    s13_buf,
    s2_buf,
    w13_sem,
    w2_sem,
    h_acc,
    *,
    n_experts,
    inter,
    w13_parts,
    depth,
    before_compute,
    rows_fp8=None,
    gather=None,
    xs_all=None,
):
    """moe_lowtok's expert loop over a ring of `depth` experts' weight units.

    The ring holds `depth` w13 unit groups and `depth` w2 units; a unit is
    prefetched `depth * units - 1` units ahead of its use. `before_compute()`
    runs once the ring is primed and before the first matmul. `rows_fp8`
    returns the rows already quantized per row (fp8 values and the restoring
    scale) when the caller holds them that way; otherwise `x_all` holds bf16
    rows and the loop quantizes them.

    With `gather` (the scratch of `_gather_scratch`), each expert is applied
    only to the rows routed to it: a one-hot matmul gathers them into blocks of
    GATHER_ROWS rows (several passes for an expert with more rows), the
    matmuls run on the block, and a second one-hot matmul scatters the f32
    contributions back into the accumulator, carried as three bf16 pieces so
    the sum is the dense formulation's value. The row positions come from one
    prefix-sum matmul over the routing mask per call. `xs_all` holds the
    per-row scales when the rows arrive quantized.
    """
    rows = wtab_ref.shape[0]
    half = inter // w13_parts
    units = w13_parts + 1
    lane_128 = lax.broadcasted_iota(jnp.int32, (rows, _LANE), 1)

    def collect(e, n):
        h = hits_s[0, gid_s[0, e]]

        @pl.when(h == 1)
        def _():
            ids[n] = e

        return n + h

    n_hit = lax.fori_loop(0, n_experts, collect, jnp.int32(0))
    n_units = n_hit * units

    def unit_copies(e_idx, part, ring):
        expert = ids[e_idx]
        slot = ring * w13_parts + jnp.minimum(part, w13_parts - 1)
        if w13_parts == 2:
            col = pl.multiple_of(jnp.minimum(part, 1) * inter, inter)
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
        else:
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
            pltpu.make_async_copy(w2_hbm.at[expert], w2_buf.at[ring], w2_sem.at[ring]),
            pltpu.make_async_copy(s2_hbm.at[expert], s2_buf.at[ring], w2_sem.at[ring]),
        )
        return w13_copies, w2_copies

    def step(e_idx, part, ring):
        """The next unit."""
        p = part + 1
        wrap = p >= units
        ring = jnp.where(wrap, ring + 1, ring)
        return (
            jnp.where(wrap, e_idx + 1, e_idx),
            jnp.where(wrap, 0, p),
            jnp.where(ring >= depth, ring - depth, ring),
        )

    def ahead(e_idx, part, ring):
        """The unit `depth * units - 1` units ahead, without a division."""
        wrap = jnp.where(part >= 1, 1, 0)
        e2 = e_idx + (depth - 1) + wrap
        part2 = jnp.where(part >= 1, part - 1, units - 1)
        r2 = ring + (depth - 1) + wrap
        return e2, part2, jnp.where(r2 >= depth, r2 - depth, r2)

    def start(e_idx, part, ring):
        w13_copies, w2_copies = unit_copies(e_idx, part, ring)

        @pl.when(part < w13_parts)
        def _():
            for c in w13_copies:
                c.start()

        @pl.when(part == w13_parts)
        def _():
            for c in w2_copies:
                c.start()

    def wait(e_idx, part, ring):
        w13_copies, w2_copies = unit_copies(e_idx, part, ring)

        @pl.when(part < w13_parts)
        def _():
            for c in w13_copies:
                c.wait()

        @pl.when(part == w13_parts)
        def _():
            for c in w2_copies:
                c.wait()

    for u in range(depth * units - 1):

        @pl.when(u < n_units)
        def _prime(u=u):
            start(
                jnp.int32(u // units),
                jnp.int32(u % units),
                jnp.int32((u // units) % depth),
            )

    before_compute()
    acc[...] = jnp.zeros_like(acc)
    if gather is not None:
        if w13_parts != 2:
            raise NotImplementedError("the gather formulation needs two w13 units")
        _gather_prologue(gid_s, wtab_ref, x_all, xs_all, gather, rows)
    elif rows_fp8 is None:
        x_q, x_s = _quantize_rows(x_all[...].astype(jnp.float32))
    else:
        x_q, x_s = rows_fp8()

    def routing_weight(expert):
        g = gid_s[0, expert]
        blk = pl.multiple_of((g >> 7) << 7, _LANE)
        w_slice = wtab_ref[:, pl.ds(blk, _LANE)]
        return jnp.sum(
            jnp.where(lane_128 + blk == g, w_slice, 0.0), axis=1, keepdims=True
        )

    def body(u, carry):
        e_idx, part, ring = carry
        expert = ids[e_idx]

        @pl.when(u + depth * units - 1 < n_units)
        def _prefetch():
            start(*ahead(e_idx, part, ring))

        wait(e_idx, part, ring)
        for r in range(depth):
            if gather is not None:
                _gather_expert(
                    gid_s,
                    gather,
                    expert,
                    part,
                    ring == r,
                    w13_buf.at[2 * r],
                    w13_buf.at[2 * r + 1],
                    s13_buf.at[2 * r],
                    s13_buf.at[2 * r + 1],
                    w2_buf.at[r],
                    s2_buf.at[r],
                    acc,
                    rows=rows,
                    fp8_rows=xs_all is not None,
                )
            elif w13_parts == 2:

                @pl.when((part == 0) & (ring == r))
                def _gate_half(r=r):
                    h = _dot(x_q, w13_buf[2 * r])
                    gate = h * x_s * s13_buf[2 * r]
                    h_acc[...] = gate * jax.nn.sigmoid(gate)

                @pl.when((part == 1) & (ring == r))
                def _up_half(r=r):
                    h = _dot(x_q, w13_buf[2 * r + 1])
                    up = h * x_s * s13_buf[2 * r + 1]
                    h_acc[...] = h_acc[...] * up

                @pl.when((part == 2) & (ring == r))
                def _down(r=r):
                    a_q, a_s = _quantize_rows(h_acc[...])
                    row_scale = a_s * routing_weight(expert)
                    o = _dot(a_q, w2_buf[r])
                    acc[...] += (o * s2_buf[r]) * row_scale
            else:
                for p in range(w13_parts):

                    @pl.when((part == p) & (ring == r))
                    def _gate_up_half(p=p, r=r):
                        slot = r * w13_parts + p
                        h = _dot(x_q, w13_buf[slot])
                        scales = s13_buf[slot]
                        col = p * half
                        h_acc[:, col : col + half] = (
                            h[:, :half] * x_s * scales[:, :half]
                        )
                        h_acc[:, inter + col : inter + col + half] = (
                            h[:, half:] * x_s * scales[:, half:]
                        )

                @pl.when((part == w13_parts) & (ring == r))
                def _down(r=r):
                    gate = h_acc[:, pl.ds(0, inter)]
                    up = h_acc[:, pl.ds(inter, inter)]
                    a_q, a_s = _quantize_rows(gate * jax.nn.sigmoid(gate) * up)
                    row_scale = a_s * routing_weight(expert)
                    o = _dot(a_q, w2_buf[r])
                    acc[...] += (o * s2_buf[r]) * row_scale

        return step(e_idx, part, ring)

    lax.fori_loop(0, n_units, body, (jnp.int32(0), jnp.int32(0), jnp.int32(0)))


def _gather_rows_of(rows):
    """Rows of the gather-side operands: the routed rows padded to whole
    128-row tiles."""
    return max(_LANE, pl.cdiv(rows, _LANE) * _LANE)


def _gather_scratch(sc, rows, hidden, inter, fp8_rows):
    """The gather formulation's scratch, appended to `sc` in
    `_gather_prologue`'s order."""
    g_rows = _gather_rows_of(rows)
    max_pass = pl.cdiv(rows, GATHER_ROWS)
    sc.vmem((g_rows, hidden), jnp.bfloat16)  # the routed rows as the gather operand
    sc.vmem((g_rows, _LANE), jnp.float32)  # routing weights of the local lane block
    sc.vmem((g_rows, _LANE), jnp.float32)  # exclusive prefix count per local lane
    sc.vmem((_LANE, g_rows), jnp.float32)  # the same, transposed
    sc.vmem((_LANE, g_rows), jnp.float32)  # routing weights, transposed
    sc.vmem((8, g_rows), jnp.float32)  # per-row scales as a lane row (fp8 rows)
    sc.vmem((1, _LANE), jnp.int32)
    sc.add(pltpu.SMEM((1, _LANE), jnp.int32), pltpu.SemaphoreType.DMA((1,)))
    sc.vmem((max_pass, GATHER_ROWS, hidden), _FP8)  # gathered rows, quantized
    sc.vmem((max_pass, GATHER_ROWS, _LANE), jnp.float32)  # their scales
    sc.vmem((max_pass, GATHER_ROWS, _LANE), jnp.float32)  # their routing weights
    sc.vmem((max_pass, GATHER_ROWS, inter), jnp.float32)  # gate, then gate * up
    sc.vmem(
        (_LANE, hidden), jnp.bfloat16
    )  # the contribution pieces, K rows of the scatter
    return 14


def _gather_prologue(gid_s, wtab_ref, x_all, xs_all, g, rows):
    """Routing positions, transposed tables and the gather operand, once per call."""
    x_g, w_blk, pos, pos_t, w_t, xs_t, counts_v, counts_s, csem, *_ = g
    g_rows = x_g.shape[0]
    hidden = x_g.shape[1]
    first = gid_s[0, 0]
    blk = pl.multiple_of((first >> 7) << 7, _LANE)
    w_rows = wtab_ref[:, pl.ds(blk, _LANE)]
    if g_rows > rows:
        w_rows = jnp.concatenate(
            [w_rows, jnp.zeros((g_rows - rows, _LANE), jnp.float32)], axis=0
        )
        x_g[pl.ds(rows, g_rows - rows), :] = jnp.zeros(
            (g_rows - rows, hidden), jnp.bfloat16
        )
    w_blk[...] = w_rows
    mask = jnp.where(w_rows != 0, 1.0, 0.0)
    row_i = lax.broadcasted_iota(jnp.int32, (g_rows, g_rows), 0)
    col_i = lax.broadcasted_iota(jnp.int32, (g_rows, g_rows), 1)
    lower = jnp.where(row_i > col_i, 1.0, 0.0).astype(jnp.bfloat16)
    upper = jnp.where(row_i < col_i, 1.0, 0.0).astype(jnp.bfloat16)
    mask_bf = mask.astype(jnp.bfloat16)
    pos[...] = jnp.dot(lower, mask_bf, preferred_element_type=jnp.float32)
    pos_t[...] = jnp.dot(
        jnp.transpose(mask).astype(jnp.bfloat16),
        upper,
        preferred_element_type=jnp.float32,
    )
    w_t[...] = jnp.transpose(w_rows)
    counts_v[...] = jnp.sum(mask, axis=0, keepdims=True).astype(jnp.int32)
    copy = pltpu.make_async_copy(counts_v, counts_s, csem.at[0])
    copy.start()
    x_g[pl.ds(0, rows), :] = x_all[...].astype(jnp.bfloat16)
    if xs_all is not None:
        xs_rows = xs_all[...]
        if g_rows > rows:
            xs_rows = jnp.concatenate(
                [xs_rows, jnp.zeros((g_rows - rows, _LANE), jnp.float32)], axis=0
            )
        xs_t[...] = jnp.transpose(xs_rows)[0:8, :]
    g[-1][...] = jnp.zeros(g[-1].shape, g[-1].dtype)
    copy.wait()


def _gather_expert(
    gid_s,
    g,
    expert,
    part,
    slot_ok,
    w_gate,
    w_up,
    s_gate,
    s_up,
    w_down,
    s_down,
    acc,
    *,
    rows,
    fp8_rows,
):
    """One expert's unit on its routed rows: gather, matmul, scatter back."""
    (
        x_g,
        w_blk,
        pos,
        pos_t,
        w_t,
        xs_t,
        counts_v,
        counts_s,
        csem,
        xq_buf,
        xs_buf,
        wk_buf,
        h_g,
        pieces,
    ) = g
    g_rows = x_g.shape[0]
    first = gid_s[0, 0]
    blk = (first >> 7) << 7
    lane = expert + (first - blk)
    n_pass = (counts_s[0, lane] + (GATHER_ROWS - 1)) // GATHER_ROWS
    lane_i = lax.broadcasted_iota(jnp.int32, (g_rows, _LANE), 1)
    sub_i = lax.broadcasted_iota(jnp.int32, (_LANE, g_rows), 0)
    p_i = lax.broadcasted_iota(jnp.int32, (GATHER_ROWS, g_rows), 0).astype(jnp.float32)

    def row_tables():
        pos_row = jnp.sum(
            jnp.where(sub_i == lane, pos_t[...], 0.0), axis=0, keepdims=True
        )
        w_row = jnp.sum(jnp.where(sub_i == lane, w_t[...], 0.0), axis=0, keepdims=True)
        return pos_row, w_row

    def one_hot_t(k, pos_row, w_row):
        base = (k * GATHER_ROWS).astype(jnp.float32)
        return jnp.where((w_row != 0) & (pos_row - base == p_i), 1.0, 0.0)

    @pl.when((part == 0) & slot_ok)
    def _gate():
        pos_row, w_row = row_tables()

        def pass_k(k, carry):
            sel = one_hot_t(k, pos_row, w_row)
            w_k = jnp.sum(jnp.where(sel != 0, w_row, 0.0), axis=1, keepdims=True)
            x_rows = jnp.dot(
                sel.astype(jnp.bfloat16), x_g[...], preferred_element_type=jnp.float32
            )
            if fp8_rows:
                x_q = x_rows.astype(_FP8)
                x_s = jnp.sum(
                    jnp.where(sel != 0, xs_t[0:1, :], 0.0), axis=1, keepdims=True
                )
            else:
                x_q, x_s = _quantize_rows(x_rows)
            xq_buf[k] = x_q
            xs_buf[k] = jnp.broadcast_to(x_s, (GATHER_ROWS, _LANE))
            wk_buf[k] = jnp.broadcast_to(w_k, (GATHER_ROWS, _LANE))
            h = _dot(x_q, w_gate[...])
            gate = h * x_s * s_gate[...]
            h_g[k] = gate * jax.nn.sigmoid(gate)
            return carry

        lax.fori_loop(0, n_pass, pass_k, 0)

    @pl.when((part == 1) & slot_ok)
    def _up():
        def pass_k(k, carry):
            h = _dot(xq_buf[k], w_up[...])
            up = h * xs_buf[k][:, 0:1] * s_up[...]
            h_g[k] = h_g[k] * up
            return carry

        lax.fori_loop(0, n_pass, pass_k, 0)

    @pl.when((part == 2) & slot_ok)
    def _down():
        pos_col = jnp.sum(
            jnp.where(lane_i == lane, pos[...], 0.0), axis=1, keepdims=True
        )
        w_col = jnp.sum(
            jnp.where(lane_i == lane, w_blk[...], 0.0), axis=1, keepdims=True
        )
        piece_lane = (lane_i & (GATHER_ROWS - 1)).astype(jnp.float32)
        in_pieces = lane_i < _PIECES * GATHER_ROWS

        def pass_k(k, carry):
            a_q, a_s = _quantize_rows(h_g[k])
            row_scale = a_s * wk_buf[k][:, 0:1]
            o = _dot(a_q, w_down[...])
            contrib = (o * s_down[...]) * row_scale
            hi = contrib.astype(jnp.bfloat16)
            rest = contrib - hi.astype(jnp.float32)
            mid = rest.astype(jnp.bfloat16)
            lo = (rest - mid.astype(jnp.float32)).astype(jnp.bfloat16)
            pieces[pl.ds(0, GATHER_ROWS), :] = hi
            pieces[pl.ds(GATHER_ROWS, GATHER_ROWS), :] = mid
            pieces[pl.ds(2 * GATHER_ROWS, GATHER_ROWS), :] = lo
            base = (k * GATHER_ROWS).astype(jnp.float32)
            scatter = jnp.where(
                (w_col != 0) & (pos_col - base == piece_lane) & in_pieces, 1.0, 0.0
            ).astype(jnp.bfloat16)
            back = jnp.dot(scatter, pieces[...], preferred_element_type=jnp.float32)
            acc[...] += back[:rows]
            return carry

        lax.fori_loop(0, n_pass, pass_k, 0)


def _kernel(
    gid_s,
    *refs,
    rank_operand,
    n_experts,
    inter,
    w13_parts,
    topk,
    dp,
    tp,
    depth,
    dispatch,
    combine,
    dispatch_form,
    combine_form,
    dispatch_fp8,
    gather_rows,
):
    if rank_operand:
        rank_s, *refs = refs
        d, t = rank_s[0], rank_s[1]
    else:
        d, t = lax.axis_index(DP_AXIS), lax.axis_index(TP_AXIS)
    _kernel_body(
        d,
        t,
        gid_s,
        *refs,
        n_experts=n_experts,
        inter=inter,
        w13_parts=w13_parts,
        topk=topk,
        dp=dp,
        tp=tp,
        depth=depth,
        dispatch=dispatch,
        combine=combine,
        dispatch_form=dispatch_form,
        combine_form=combine_form,
        dispatch_fp8=dispatch_fp8,
        gather_rows=gather_rows,
    )


def _kernel_body(
    d,
    t,
    gid_s,
    x_ref,
    lg_ref,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    out_ref,
    *scratch,
    n_experts,
    inter,
    w13_parts,
    topk,
    dp,
    tp,
    depth,
    dispatch,
    combine,
    dispatch_form,
    combine_form,
    dispatch_fp8,
    gather_rows,
):
    scratch = list(scratch)

    def take(n):
        taken, scratch[:n] = scratch[:n], []
        return taken

    hidden = x_ref.shape[1]
    half = hidden // 2
    rows_l = x_ref.shape[0] if dispatch == "inkernel" else x_ref.shape[0] // dp
    xs_all = None
    if dispatch == "inkernel":
        x_all, lg_all, dsem, xrecv = take(4)
        if dispatch_fp8:
            xs_all = take(1)[0]
    else:
        x_all, lg_all = x_ref, lg_ref
    ids, hits_v, hits_s, hsem, wtab = take(5)
    ring = take(7)
    acc = take(1)[0]
    if combine == "inkernel":
        if combine_form == "flat":
            c_send, c_recv1, c_pair_send, c_pair_recv, csem = take(5)
        else:
            c_send, c_recv1, c_half, c_recv2, csem = take(5)
    gscratch = tuple(take(14)) if gather_rows else None
    assert not scratch

    if dispatch == "inkernel" or combine == "inkernel":
        _barrier(dp, tp)

    if dispatch == "inkernel":
        # Logits first: the top-k and the hit collection only need them.
        for j in range(dp):

            @pl.when(j != d)
            def _(j=j):
                _push(
                    lg_ref, lg_all.at[_rows_of(d, rows_l)], dsem.at[0], dsem.at[1], j, t
                )

        lg_all[_rows_of(d, rows_l), :] = lg_ref[...]
        mine = _rows_of(d, rows_l)
        if dispatch_fp8:
            # The rows quantized per row before the push: the same values the
            # loop would produce from the gathered bf16 rows, in half the bytes.
            x_q_l, x_s_l = _quantize_rows(x_ref[...].astype(jnp.float32))
            x_all[mine, :] = x_q_l
            xs_all[mine, :] = jnp.broadcast_to(x_s_l, (rows_l, _LANE))
            for j in range(dp):

                @pl.when(j != d)
                def _(j=j):
                    _push(
                        xs_all.at[mine], xs_all.at[mine], dsem.at[5], dsem.at[6], j, t
                    )

        else:
            x_all[mine, :] = x_ref[...]
        if dispatch_form == "flat":
            for j in range(dp):

                @pl.when(j != d)
                def _(j=j):
                    _push(x_all.at[mine], x_all.at[mine], dsem.at[2], xrecv.at[d], j, t)

        else:
            for u, cols in ((0, pl.ds(0, half)), (1, pl.ds(half, half))):

                @pl.when(t == u)
                def _(cols=cols):
                    for j in range(dp):

                        @pl.when(j != d)
                        def _(j=j):
                            _push(
                                x_all.at[mine, cols],
                                x_all.at[mine, cols],
                                dsem.at[2],
                                xrecv.at[d],
                                j,
                                t,
                            )

        _wait_bytes(lg_all.at[pl.ds(0, (dp - 1) * rows_l)], dsem.at[1])

    wtab[...] = _select_topk(lg_all[...], topk, from_logits=True)
    hit = jnp.any(wtab[...] != 0, axis=0, keepdims=True)
    hits_v[...] = jnp.where(hit, 1, 0).astype(jnp.int32)
    hcopy = pltpu.make_async_copy(hits_v, hits_s, hsem.at[0])
    hcopy.start()
    hcopy.wait()

    def rows_ready():
        if dispatch != "inkernel":
            return
        if dispatch_fp8:
            _wait_bytes(xs_all.at[pl.ds(0, (dp - 1) * rows_l)], dsem.at[6])
        if dispatch_form == "flat":
            for j in range(dp):

                @pl.when(j != d)
                def _(j=j):
                    _wait_bytes(x_all.at[_rows_of(j, rows_l)], xrecv.at[j])

            return
        # Each foreign block's column half lands from its DP peer and is
        # forwarded to the TP partner, whose other half arrives the same way.
        for u, cols in ((0, pl.ds(0, half)), (1, pl.ds(half, half))):

            @pl.when(t == u)
            def _(cols=cols):
                for j in range(dp):

                    @pl.when(j != d)
                    def _(j=j):
                        _wait_bytes(x_all.at[_rows_of(j, rows_l), cols], xrecv.at[j])
                        _push(
                            x_all.at[_rows_of(j, rows_l), cols],
                            x_all.at[_rows_of(j, rows_l), cols],
                            dsem.at[3],
                            dsem.at[4],
                            d,
                            1 - t,
                        )

                _wait_bytes(x_all.at[pl.ds(0, (dp - 1) * rows_l), cols], dsem.at[4])

    _stream_ring(
        gid_s,
        hits_s,
        wtab,
        x_all,
        s13_hbm,
        s2_hbm,
        w13_hbm,
        w2_hbm,
        acc,
        ids,
        *ring,
        n_experts=n_experts,
        inter=inter,
        w13_parts=w13_parts,
        depth=depth,
        before_compute=rows_ready,
        rows_fp8=(lambda: (x_all[...], xs_all[:, 0:1])) if dispatch_fp8 else None,
        gather=gscratch,
        xs_all=xs_all if dispatch_fp8 else None,
    )

    if combine != "inkernel":
        out_ref[...] = acc[...].astype(out_ref.dtype)
    elif combine_form == "flat":
        c_send[...] = acc[...].astype(c_send.dtype)
        c_recv1[pl.ds(d, 1)] = jnp.zeros((1, rows_l, hidden), c_recv1.dtype)
        for j in range(dp):

            @pl.when(j != d)
            def _(j=j):
                _push(
                    c_send.at[_rows_of(j, rows_l)],
                    c_recv1.at[d],
                    csem.at[0],
                    csem.at[1],
                    j,
                    t,
                )

        _wait_bytes(c_recv1.at[pl.ds(0, dp - 1)], csem.at[1])
        s = acc[_rows_of(d, rows_l), :]
        for j in range(dp):
            s = s + c_recv1[j].astype(jnp.float32)
        c_pair_send[...] = s.astype(c_pair_send.dtype)
        _push(c_pair_send, c_pair_recv, csem.at[2], csem.at[3], d, 1 - t)
        _wait_bytes(c_pair_recv, csem.at[3])
        out_ref[...] = (
            c_pair_send[...].astype(jnp.float32) + c_pair_recv[...].astype(jnp.float32)
        ).astype(out_ref.dtype)
        _wait_bytes(c_recv1.at[pl.ds(0, dp - 1)], csem.at[0])
        _wait_bytes(c_pair_recv, csem.at[2])
    else:
        c_send[...] = acc[...].astype(c_send.dtype)
        # Stage 1: the partner's half of my partial sums, summed intra-chip.
        for u, mine, theirs in (
            (0, pl.ds(0, half), pl.ds(half, half)),
            (1, pl.ds(half, half), pl.ds(0, half)),
        ):

            @pl.when(t == u)
            def _(theirs=theirs):
                _push(c_send.at[:, theirs], c_recv1, csem.at[0], csem.at[1], d, 1 - t)

        _wait_bytes(c_recv1, csem.at[1])
        for u, mine, theirs in (
            (0, pl.ds(0, half), pl.ds(half, half)),
            (1, pl.ds(half, half), pl.ds(0, half)),
        ):

            @pl.when(t == u)
            def _(mine=mine):
                c_half[...] = (acc[:, mine] + c_recv1[...].astype(jnp.float32)).astype(
                    c_half.dtype
                )

        # Stage 2: the chip partials of the foreign blocks to their owners.
        c_recv2[pl.ds(d, 1)] = jnp.zeros((1, rows_l, half), c_recv2.dtype)
        for j in range(dp):

            @pl.when(j != d)
            def _(j=j):
                _push(
                    c_half.at[_rows_of(j, rows_l)],
                    c_recv2.at[d],
                    csem.at[2],
                    csem.at[3],
                    j,
                    t,
                )

        _wait_bytes(c_recv2.at[pl.ds(0, dp - 1)], csem.at[3])
        fin = c_half[_rows_of(d, rows_l), :].astype(jnp.float32)
        for j in range(dp):
            fin = fin + c_recv2[j].astype(jnp.float32)
        fin = fin.astype(out_ref.dtype)
        # Stage 3: the finished halves exchanged intra-chip.
        for u, mine, theirs in (
            (0, pl.ds(0, half), pl.ds(half, half)),
            (1, pl.ds(half, half), pl.ds(0, half)),
        ):

            @pl.when(t == u)
            def _(mine=mine):
                out_ref[:, mine] = fin
                _push(
                    out_ref.at[:, mine],
                    out_ref.at[:, mine],
                    csem.at[4],
                    csem.at[5],
                    d,
                    1 - t,
                )

        _wait_bytes(out_ref.at[:, pl.ds(0, half)], csem.at[5])
        _wait_bytes(c_recv1, csem.at[0])
        _wait_bytes(c_recv2.at[pl.ds(0, dp - 1)], csem.at[2])
        _wait_bytes(out_ref.at[:, pl.ds(0, half)], csem.at[4])

    if dispatch == "inkernel":
        _wait_bytes(lg_all.at[pl.ds(0, (dp - 1) * rows_l)], dsem.at[0])
        if dispatch_fp8:
            _wait_bytes(xs_all.at[pl.ds(0, (dp - 1) * rows_l)], dsem.at[5])
        if dispatch_form == "flat":
            _wait_bytes(x_all.at[pl.ds(0, (dp - 1) * rows_l)], dsem.at[2])
        else:
            _wait_bytes(
                x_all.at[pl.ds(0, (dp - 1) * rows_l), pl.ds(0, half)], dsem.at[2]
            )
            _wait_bytes(
                x_all.at[pl.ds(0, (dp - 1) * rows_l), pl.ds(0, half)], dsem.at[3]
            )


class _Scratch:
    """The scratch list and the VMEM bytes it declares."""

    def __init__(self):
        self.shapes = []
        self.vmem_bytes = 0

    def vmem(self, shape, dtype):
        n = 1
        for dim in shape:
            n *= int(dim)
        self.vmem_bytes += n * jnp.dtype(dtype).itemsize
        self.shapes.append(pltpu.VMEM(tuple(shape), dtype))

    def add(self, *shapes):
        self.shapes.extend(shapes)

    def ring(self, hidden, inter, w13_parts, depth, rows):
        half = inter // w13_parts
        h_acc_cols = inter if w13_parts == 2 else 2 * inter
        self.vmem((depth * w13_parts, hidden, 2 * half), _FP8)
        self.vmem((depth, inter, hidden), _FP8)
        self.vmem((depth * w13_parts, 1, 2 * half), jnp.float32)
        self.vmem((depth, 1, hidden), jnp.float32)
        self.add(
            pltpu.SemaphoreType.DMA((depth * w13_parts,)),
            pltpu.SemaphoreType.DMA((depth,)),
        )
        self.vmem((rows, h_acc_cols), jnp.float32)


def vmem_budget(scratch_bytes, operand_bytes):
    """Declared scratch and VMEM operands plus a compiler margin, capped."""
    need = int((scratch_bytes + operand_bytes) * _VMEM_MARGIN)
    capacity = int(pltpu.get_tpu_info().vmem_capacity_bytes * _VMEM_FRACTION)
    return min(need, capacity)


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "dp",
        "tp",
        "dispatch",
        "combine",
        "dispatch_form",
        "combine_form",
        "depth",
        "dispatch_fp8",
        "gather_rows",
    ),
)
def moe_lowtok_fused(
    x,
    logits,
    w13,
    s13,
    w2,
    s2,
    experts_start=0,
    *,
    topk,
    dp,
    tp,
    dispatch="inkernel",
    combine="inkernel",
    dispatch_form="flat",
    combine_form="hier",
    depth=1,
    dispatch_fp8=False,
    rank=None,
    gather_rows=False,
):
    """The routed-expert sum of this engine's rows, with the collectives in the kernel.

    Inside a shard_map over a ("dp", "tp") mesh; `rank` ([2] int32: this core's
    index on each axis) when the caller's export path refuses `lax.axis_index`,
    as the torch bridge does. With `dispatch` "inkernel", `x`
    [rows, hidden] bf16 and `logits` [rows, experts] bf16 are this engine's own
    rows and the kernel gathers them over the DP axis; with "xla" they are the
    gathered rows of every engine. With `combine` "inkernel" the result is this
    engine's rows [rows, hidden] summed over every rank; with "xla" it is this
    rank's partial sum over all gathered rows, for the caller's combine. This
    rank holds experts [experts_start, experts_start + E). `dispatch_fp8` pushes
    the rows quantized per row to fp8 with their scale (half the bytes, the
    same values the kernel computes from bf16 rows); it needs row blocks of
    whole fp8 tiles (32 rows per engine). `gather_rows` applies each expert to its
    routed rows only (see `_stream_ring`).
    """
    if dispatch not in SIDES or combine not in SIDES:
        raise ValueError(f"dispatch and combine are one of {SIDES}")
    if dispatch_form not in FORMS or combine_form not in FORMS:
        raise ValueError(f"forms are one of {FORMS}")
    if tp != 2 and (
        (dispatch == "inkernel" and dispatch_form == "hier")
        or (combine == "inkernel" and combine_form == "hier")
    ):
        raise ValueError("the hier forms pair the two cores of a chip (tp == 2)")
    if depth < 1:
        raise ValueError("depth is at least 1")
    if dispatch_fp8 and dispatch != "inkernel":
        raise ValueError("dispatch_fp8 needs the dispatch in the kernel")
    tokens, hidden = x.shape
    if hidden % (2 * _LANE):
        raise ValueError(f"hidden {hidden} is not a multiple of {2 * _LANE}")
    n_experts, inter, w13_parts, _ = _layout(x, w13)
    lanes = pl.cdiv(logits.shape[1], _LANE) * _LANE
    if dispatch == "inkernel":
        rows_l = pl.cdiv(tokens, _ROW_TILE) * _ROW_TILE
        rows = rows_l * dp
    else:
        if tokens % dp:
            raise ValueError(f"{tokens} gathered rows do not split over dp={dp}")
        rows_l = tokens // dp
        if rows_l % _ROW_TILE:
            raise ValueError(
                f"{rows_l} rows per engine are not a multiple of {_ROW_TILE}"
            )
        rows = tokens
    if dispatch_fp8 and rows_l % 32:
        raise ValueError(
            f"dispatch_fp8 needs {rows_l} rows per engine to be a multiple of 32"
        )
    pad_rows = (rows_l if dispatch == "inkernel" else rows) - tokens
    x_rows = jnp.pad(x, ((0, pad_rows), (0, 0)))
    # The router logits keep their dtype, as moe_lowtok's scores entry does.
    if logits.dtype not in (jnp.bfloat16, jnp.float32):
        logits = logits.astype(jnp.float32)
    lg_rows = jnp.pad(
        logits,
        ((0, pad_rows), (0, lanes - logits.shape[1])),
        constant_values=jnp.asarray(-jnp.inf, dtype=logits.dtype),
    )
    s13 = s13.reshape(n_experts, 1, 2 * inter).astype(jnp.float32)
    s2 = s2.reshape(n_experts, 1, hidden).astype(jnp.float32)
    gid = (experts_start + jnp.arange(n_experts, dtype=jnp.int32)).reshape(1, n_experts)
    half = hidden // 2
    out_rows = rows_l if combine == "inkernel" else rows

    sc = _Scratch()
    if dispatch == "inkernel":
        sc.vmem((rows, hidden), _FP8 if dispatch_fp8 else jnp.bfloat16)
        sc.vmem((rows, lanes), lg_rows.dtype)
        sc.add(pltpu.SemaphoreType.DMA((7,)), pltpu.SemaphoreType.DMA((dp,)))
        if dispatch_fp8:
            sc.vmem((rows, _LANE), jnp.float32)
    sc.add(pltpu.SMEM((n_experts,), jnp.int32))
    sc.vmem((1, lanes), jnp.int32)
    sc.add(pltpu.SMEM((1, lanes), jnp.int32), pltpu.SemaphoreType.DMA((1,)))
    sc.vmem((rows, lanes), jnp.float32)
    sc.ring(hidden, inter, w13_parts, depth, rows)
    sc.vmem((rows, hidden), jnp.float32)
    if combine == "inkernel":
        sc.vmem((rows, hidden), jnp.bfloat16)
        if combine_form == "flat":
            sc.vmem((dp, rows_l, hidden), jnp.bfloat16)
            sc.vmem((rows_l, hidden), jnp.bfloat16)
            sc.vmem((rows_l, hidden), jnp.bfloat16)
        else:
            sc.vmem((rows, half), jnp.bfloat16)
            sc.vmem((rows, half), jnp.bfloat16)
            sc.vmem((dp, rows_l, half), jnp.bfloat16)
        sc.add(pltpu.SemaphoreType.DMA((6,)))
    if gather_rows:
        _gather_scratch(sc, rows, hidden, inter, dispatch_fp8)
    operand_bytes = (
        x_rows.size * 2
        + lg_rows.size * lg_rows.dtype.itemsize
        + out_rows * hidden * 2
        # x quantized to fp8 and its f32 cast, the dot results
        + rows * hidden * 5
        + rows * 2 * inter * 4
    )
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    dispatch_tag = dispatch_form[0] if dispatch == "inkernel" else ""
    name = (
        f"moe_lowtok_fused_d{dispatch[0]}{dispatch_tag}"
        f"{'q' if dispatch_fp8 else ''}"
        f"_c{combine[0]}{combine_form[0] if combine == 'inkernel' else ''}_r{depth}"
        f"{'_g' if gather_rows else ''}"
    )
    prefetch = (
        (gid,) if rank is None else (gid, jnp.asarray(rank, jnp.int32).reshape(2))
    )
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            rank_operand=rank is not None,
            n_experts=n_experts,
            inter=inter,
            w13_parts=w13_parts,
            topk=topk,
            dp=dp,
            tp=tp,
            depth=depth,
            dispatch=dispatch,
            combine=combine,
            dispatch_form=dispatch_form,
            combine_form=combine_form,
            dispatch_fp8=dispatch_fp8,
            gather_rows=gather_rows,
        ),
        out_shape=jax.ShapeDtypeStruct((out_rows, hidden), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(prefetch),
            in_specs=[vmem, vmem, hbm, hbm, hbm, hbm],
            out_specs=vmem,
            scratch_shapes=tuple(sc.shapes),
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_budget(sc.vmem_bytes, operand_bytes),
            **(
                {"collective_id": COLLECTIVE_ID}
                if "inkernel" in (dispatch, combine)
                else {}
            ),
        ),
        name=name,
    )(*prefetch, x_rows, lg_rows, s13, s2, w13, w2)
    if dispatch == "inkernel" and combine == "inkernel":
        return out[:tokens]
    # With the dispatch in the kernel and the combine outside, the partial
    # sums keep the gathered layout of rows_l padded rows per engine.
    return out
