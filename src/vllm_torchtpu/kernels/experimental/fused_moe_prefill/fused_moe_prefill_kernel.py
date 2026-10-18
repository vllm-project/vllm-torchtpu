# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE prefill kernel: routed experts, shared expert, combine and all-reduce.

Under MoE tensor parallelism every rank holds all experts with the
intermediate dimension sharded, and every rank sees every token. At prefill
sizes each active expert owns hundreds of rows, so unlike the decode kernel
(which computes every expert on every token) this one gathers each expert's
rows. Per rank the layer is

    routing plan (XLA) -> expert kernel (Pallas) -> combine (Pallas) -> psum

The routing plan orders the routes expert-major with two sorts (no scatter or
gather over the routes). The expert kernel visits the experts that received
rows: their weights stream through a ring of VMEM slots a few visits ahead,
each tile's FP8 activation rows are gathered from HBM by one DMA per row a few
tiles ahead, and each tile runs gate/up, the activation, the intermediate
requantization and down, then writes its unweighted rows into an expert-major
BF16 slab in HBM. The shared expert is one more visit over every token, with
contiguous rows. The combine is fused_moe/v2's destination combine: per token,
the router-weighted sum of its top-k routed rows (times `routed_scale`) plus
its shared row, gated unless the shared expert has no gate.
The sum over the ranks' intermediate shards is a reduce-scatter plus an
all-gather issued per combine chunk, so each chunk's reduction runs under the
next chunk's combine.

With expert parallelism and TP attention a rank holds whole experts instead of
slices and the call is the same, restricted to the rank's expert range.

Numerics: hidden states are quantized to FP8 once per call with one scale per
row; the activated intermediate is quantized per row; both matmuls are
FP8 x FP8 with FP32 accumulation. Each route's rows are BF16 in the slab (the
down projection's output is a partial sum over an intermediate shard; the
grouped GEMM path writes BF16 there too). Router top-k stays with the caller.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v2 import host as v2_host
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    DN,
    _intermediate_rows,
    _tile_row_scales,
    build_combine_kernel,
    combine_step_tokens,
)

_LANES = 128
# Weight DMAs run at this priority so the activation-row gathers, which the
# next tile waits on first, are not queued behind a whole expert's weights.
_WEIGHT_DMA_PRIORITY = 1
# Routed-expert weight ring depth: one slot computes while _NUM_BUFFERS - 1
# experts' weights stream in. Each slot holds a whole local expert (for
# Qwen3.5-397B, 1.5 MiB under TP8 and 12 MiB under EP8), so a deeper ring only
# helps when experts own too few rows to hide one transfer behind compute.
_NUM_BUFFERS = 3
# A tile computes on the smallest of _TILE_RUNGS heights (multiples of
# tile_rows / _TILE_RUNGS) that covers its live rows, so an expert's last,
# partial tile does not compute a whole tile of dead rows.
_TILE_RUNGS = 4
# Activation-row DMAs issued per loop iteration. Up to _GATHER_UNROLL - 1 dead
# rows ride along at a tile's end; they read in-bounds table entries.
_GATHER_UNROLL = 8
# Tiles whose activation rows are gathered ahead of the tile computing. One
# tile of lookahead leaves the row DMAs exposed at expert boundaries.
_LOOKAHEAD = 2
# Visit-table entries past the shared visit: the kernel reads a few tiles
# ahead (row lookahead plus index window) without branching on the count.
_VISIT_SLACK = 8
# Most tokens per combine chunk. The combine's position and coefficient tables
# live in SMEM (1 MiB): 16384 tokens with 11 slots each do not fit in one.
_COMBINE_CHUNK_TOKENS = 2048


def _cdiv(a, b):
    return (a + b - 1) // b


def _byte_transpose4(a):
    """Byte t of result j is byte j of a[t], for four u32 arrays: a 4x4 byte
    transpose per element, by shifts and masks only."""
    lo16, hi16 = jnp.uint32(0x0000FFFF), jnp.uint32(0xFFFF0000)
    even8, odd8 = jnp.uint32(0x00FF00FF), jnp.uint32(0xFF00FF00)
    b0 = (a[0] & lo16) | (a[2] << 16)
    b2 = (a[0] >> 16) | (a[2] & hi16)
    b1 = (a[1] & lo16) | (a[3] << 16)
    b3 = (a[1] >> 16) | (a[3] & hi16)
    return (
        (b0 & even8) | ((b1 & even8) << 8),
        ((b0 >> 8) & even8) | (b1 & odd8),
        (b2 & even8) | ((b3 & even8) << 8),
        ((b2 >> 8) & even8) | (b3 & odd8),
    )


def _tile_rows(tokens):
    """Tile height for a token count. Every tile pays a fixed cost (index
    window, scale window, waits), so tall tiles win once experts hold a few
    hundred rows each; below that a tall tile mostly computes dead rows."""
    return 256 if tokens >= 8192 else 128


@functools.cache
def _build_expert_kernel(
    *, experts, h, i, i_s, tokens, routes, tile_rows, weight_dtype
):
    """The per-rank expert kernel for a fixed token and route count.

    Slab layout (rows): [0, routes) routed rows expert-major, then the shared
    expert's `tokens` rows at `routes`, then tile_rows rows of slack. Returns
    the pallas call and the padded row counts of its scale and token tables.
    """
    n_act = _LOOKAHEAD + 1  # activation tile slots
    window = tile_rows + _LANES  # a tile's rows start anywhere in an aligned window
    scale_window = tile_rows // _LANES + 1
    rungs = [tile_rows * (r + 1) // _TILE_RUNGS for r in range(_TILE_RUNGS)]
    lane_blocks = h // _LANES
    # Layouts the tile body can move as whole u32 words (compute below): 8-bit
    # rows whose lane blocks come in fours, at rung heights of whole 32-row
    # tiles, for the lhs; pairs of bf16 lane blocks, at rung heights of whole
    # 8-row tiles, for the slab rows. The lhs loads have sublane stride L, and
    # libtpu splits a strided load into gcd(stride, 32) / 4 loads (at least
    # one) to avoid VMEM bank conflicts: 4 at L = 48, but 8 one-sublane loads
    # at L = 32 (h = 4096), more bundles than the retile they replace, so L
    # that are multiples of 32 keep the retile.
    lhs_from_words = (
        jnp.dtype(weight_dtype).itemsize == 1
        and lane_blocks % 4 == 0
        and tile_rows % (4 * _TILE_RUNGS * 8) == 0
        and math.gcd(lane_blocks, 32) <= 16
    )
    out_by_pairs = lane_blocks % 2 == 0 and tile_rows % (8 * _TILE_RUNGS) == 0
    shared = experts  # visit-table id of the shared expert
    slab_rows = routes + tokens + tile_rows
    scale_rows = _cdiv(slab_rows, _LANES) + 1
    token_rows = _cdiv(routes, _LANES) * _LANES + window
    nb = _NUM_BUFFERS

    def kernel(
        visit_expert,
        visit_start,
        visit_rows,
        num_visits,
        x_hbm,
        row_scale_hbm,
        row_token_hbm,
        w1_hbm,
        w2_hbm,
        w1_scale_hbm,
        w2_scale_hbm,
        shared_w1_hbm,
        shared_w2_hbm,
        shared_w1_scale_hbm,
        shared_w2_scale_hbm,
        slab_hbm,
        act_vmem,
        act_scale_vmem,
        out_vmem,
        w1_ring,
        w2_ring,
        w1_scale_ring,
        w2_scale_ring,
        token_window,
        act_sems,
        window_sems,
        out_sems,
        ring_sems,
        shared_w1_vmem,
        shared_w2_vmem,
        shared_w1_scale_vmem,
        shared_w2_scale_vmem,
        shared_sem,
    ):
        """One rank's expert loop. A visit is one expert's rows (or the shared
        expert's, every token); a tile is up to tile_rows rows of one visit.
        Shapes use E local experts, V = E + _VISIT_SLACK visit-table entries,
        T tokens, R routed slab rows, L = h // 128 lane blocks, nb ring slots
        and n_act = _LOOKAHEAD + 1 activation slots.

        Scalar prefetch (SMEM):
          visit_expert [V] int32: local expert of each visit (shared = E).
          visit_start [V] int32: first slab row of each visit.
          visit_rows [V] int32: rows of each visit.
          num_visits [1] int32: visits, the shared one included.
        Inputs in HBM:
          x_hbm [T, L, 128] fp8: row-quantized hidden states.
          row_scale_hbm [scale_rows, 128] f32: activation scale of each slab row.
          row_token_hbm [token_rows] int32: token of each routed slab row.
          w1_hbm [E, h, 2i], w2_hbm [E, i, h]: routed expert weights.
          w1_scale_hbm [E, 1, 2i], w2_scale_hbm [E, 1, h] f32: their scales.
          shared_w1_hbm [1, h, 2 i_s], shared_w2_hbm [1, i_s, h]: shared expert.
          shared_w1_scale_hbm [1, 1, 2 i_s], shared_w2_scale_hbm [1, 1, h] f32.
        Output in HBM:
          slab_hbm [slab_rows, L, 128] bf16: unweighted rows, slab order.
        Scratch:
          act_vmem [n_act, tile_rows, L, 128] fp8: gathered activation rows.
          act_scale_vmem [n_act, scale_window, 128] f32: their scales.
          out_vmem [2, tile_rows, L, 128] bf16: double-buffered tile output.
          w1_ring, w2_ring, w1_scale_ring, w2_scale_ring: [nb, ...] expert
            weight ring.
          token_window [2 * window] int32 (SMEM): token numbers of a tile's rows,
            window slot s at [s * window, (s + 1) * window).
          act_sems [n_act], window_sems [2], out_sems [2], ring_sems [nb]: DMA.
          shared_w1_vmem, shared_w2_vmem, shared_w1_scale_vmem,
            shared_w2_scale_vmem, shared_sem: the shared expert, loaded once.
        """
        n_visit = num_visits[0]

        def is_shared(v):
            return visit_expert[v] == shared

        # ---- weights: routed experts through a ring of nb slots, nb - 1
        # visits ahead; the shared expert (its own intermediate size) has its
        # own buffers, loaded once at entry ----
        def ring_copies(expert, slot):
            srcs = (
                w1_hbm.at[expert],
                w2_hbm.at[expert],
                w1_scale_hbm.at[expert],
                w2_scale_hbm.at[expert],
            )
            dsts = (
                w1_ring.at[slot],
                w2_ring.at[slot],
                w1_scale_ring.at[slot],
                w2_scale_ring.at[slot],
            )
            return [
                pltpu.make_async_copy(s, d, ring_sems.at[slot])
                for s, d in zip(srcs, dsts)
            ]

        shared_copies = [
            pltpu.make_async_copy(s, d, shared_sem)
            for s, d in zip(
                (
                    shared_w1_hbm.at[0],
                    shared_w2_hbm.at[0],
                    shared_w1_scale_hbm.at[0],
                    shared_w2_scale_hbm.at[0],
                ),
                (
                    shared_w1_vmem,
                    shared_w2_vmem,
                    shared_w1_scale_vmem,
                    shared_w2_scale_vmem,
                ),
            )
        ]

        def start_weights(v):
            @pl.when(jnp.logical_not(is_shared(v)))
            def _():
                for c in ring_copies(visit_expert[v], lax.rem(v, nb)):
                    c.start(priority=_WEIGHT_DMA_PRIORITY)

        def wait_weights(v):
            @pl.when(jnp.logical_not(is_shared(v)))
            def _():
                for c in ring_copies(0, lax.rem(v, nb)):
                    c.wait()

            @pl.when(is_shared(v))
            def _():
                for c in shared_copies:
                    c.wait()

        # ---- tile coordinates: (visit, tile within visit) -> next ----
        def next_tile(v, t):
            more = (t + 1) * tile_rows < visit_rows[v]
            return jnp.where(more, v, v + 1), jnp.where(more, t + 1, 0)

        def tile_start(v, t):
            return visit_start[v] + t * tile_rows

        def tile_live(v, t):
            return jnp.minimum(visit_rows[v] - t * tile_rows, tile_rows)

        # ---- token-number windows (routed tiles only) ----
        def window_copy(start, slot):
            aligned = pl.multiple_of(start - lax.rem(start, _LANES), _LANES)
            lo = slot * window  # the slot's first entry in the flat ring
            if not isinstance(lo, int):
                lo = pl.multiple_of(lo, _LANES)
            return pltpu.make_async_copy(
                row_token_hbm.at[pl.ds(aligned, window)],
                token_window.at[pl.ds(lo, window)],
                window_sems.at[slot],
            )

        def start_window(v, t, slot):
            @pl.when(jnp.logical_and(v < n_visit, jnp.logical_not(is_shared(v))))
            def _():
                window_copy(tile_start(v, t), slot).start()

        def wait_window(v, t, slot):
            @pl.when(jnp.logical_not(is_shared(v)))
            def _():
                window_copy(tile_start(v, t), slot).wait()

        # ---- activation rows ----
        def scale_copy(start, slot):
            return pltpu.make_async_copy(
                row_scale_hbm.at[pl.ds(start // _LANES, scale_window)],
                act_scale_vmem.at[slot],
                act_sems.at[slot],
            )

        def issue_tile(v, t, slot, wslot):
            """Start tile (v, t)'s rows into activation slot `slot`, reading
            token numbers from window slot `wslot`."""
            start = tile_start(v, t)

            # The shared expert's rows are contiguous tokens; x is not padded,
            # so the last tile moves only the rows it has.
            @pl.when(is_shared(v))
            def _():
                n = tile_live(v, t)
                pltpu.make_async_copy(
                    x_hbm.at[pl.ds(t * tile_rows, n)],
                    act_vmem.at[slot, pl.ds(0, n)],
                    act_sems.at[slot],
                ).start()

            @pl.when(jnp.logical_not(is_shared(v)))
            def _():
                # The carry is the flat-ring index of the block's first row,
                # so each row costs one SMEM load at a constant offset plus
                # its DMA (no 2-D index arithmetic per row).
                def block(b, ptr):
                    r0 = b * _GATHER_UNROLL
                    for u in range(_GATHER_UNROLL):
                        token = token_window[ptr + u]
                        pltpu.make_async_copy(
                            x_hbm.at[token],
                            act_vmem.at[slot, r0 + u],
                            act_sems.at[slot],
                        ).start()
                    return ptr + _GATHER_UNROLL

                ptr0 = wslot * window + lax.rem(start, _LANES)
                lax.fori_loop(0, _cdiv(tile_live(v, t), _GATHER_UNROLL), block, ptr0)

            scale_copy(start, slot).start()

        def wait_tile(v, t, slot):
            n = jnp.where(
                is_shared(v),
                tile_live(v, t),
                _cdiv(tile_live(v, t), _GATHER_UNROLL) * _GATHER_UNROLL,
            )
            # Semaphores count bytes: the wait only needs the extent.
            pltpu.make_async_copy(
                act_vmem.at[slot, pl.ds(0, n)],
                act_vmem.at[slot, pl.ds(0, n)],
                act_sems.at[slot],
            ).wait()
            scale_copy(tile_start(v, t), slot).wait()

        def out_copy(start, parity, n):
            return pltpu.make_async_copy(
                out_vmem.at[parity, pl.ds(0, n)],
                slab_hbm.at[pl.ds(start, n)],
                out_sems.at[parity],
            )

        def out_wait(parity, n):
            pltpu.make_async_copy(
                out_vmem.at[parity, pl.ds(0, n)],
                out_vmem.at[parity, pl.ds(0, n)],
                out_sems.at[parity],
            ).wait()

        # ---- prologue ----
        for c in shared_copies:
            c.start(priority=_WEIGHT_DMA_PRIORITY)
        for k in range(nb - 1):

            @pl.when(k < n_visit)
            def _(k=k):
                start_weights(jnp.int32(k))

        # Activation rows run _LOOKAHEAD tiles ahead of compute in a ring of
        # n_act slots; token windows run one tile ahead of the row issue in a
        # ring of 2. Tile j uses activation slot j % n_act and window slot j % 2.
        z = jnp.int32(0)
        start_window(z, z, 0)
        vi, ti = z, z  # issue cursor: the next tile whose rows to start
        for k in range(_LOOKAHEAD):

            @pl.when(vi < n_visit)
            def _(k=k, vi=vi, ti=ti):
                wait_window(vi, ti, k % 2)
                issue_tile(vi, ti, k % n_act, k % 2)
                vw, tw = next_tile(vi, ti)
                start_window(vw, tw, (k + 1) % 2)

            vi, ti = next_tile(vi, ti)

        # ---- main loop over tiles ----
        def tile_body(carry):
            v, t, g, n_out0, n_out1, vi, ti = carry
            p = lax.rem(g, 2)  # out_vmem parity
            aslot = lax.rem(g, n_act)

            @pl.when(t == 0)
            def _():
                wait_weights(v)

                @pl.when(v + nb - 1 < n_visit)
                def _():
                    start_weights(v + nb - 1)

            wait_tile(v, t, aslot)

            vn, tn = next_tile(v, t)
            gi = g + _LOOKAHEAD  # global index of the tile at the issue cursor

            @pl.when(vi < n_visit)
            def _():
                wait_window(vi, ti, lax.rem(gi, 2))
                issue_tile(vi, ti, lax.rem(gi, n_act), lax.rem(gi, 2))
                vw, tw = next_tile(vi, ti)
                start_window(vw, tw, lax.rem(gi + 1, 2))

            vi_n, ti_n = next_tile(vi, ti)
            vi_n = jnp.where(vi < n_visit, vi_n, vi)
            ti_n = jnp.where(vi < n_visit, ti_n, ti)

            slot = lax.rem(v, nb)
            start = tile_start(v, t)
            live = tile_live(v, t)

            # out_vmem[p] was last written two tiles ago; free it.
            n_prev = jnp.where(p == 0, n_out0, n_out1)

            @pl.when(n_prev > 0)
            def _():
                out_wait(p, n_prev)

            def compute(rows, w1, w2, w1_scale, w2_scale):
                if lhs_from_words:
                    # The matmul's 8-bit layout packs 4 consecutive rows into
                    # each u32 word (row 4s + t in byte t of sublane s). A
                    # gathered row is L / 4 u32 word-rows, word-row k packing
                    # lane blocks 4k..4k+3 (block 4k + j in byte j). So load
                    # rows t, t + 4, ... of word-row k with one sublane-strided
                    # load per t and transpose bytes across the four loads:
                    # the gathered bytes as they are, no relayout, no casts.
                    wpr = lane_blocks // 4
                    a32 = (
                        act_vmem.at[aslot]
                        .bitcast(jnp.uint32)
                        .reshape(tile_rows * wpr, _LANES)
                    )
                    blocks = []
                    for k in range(wpr):
                        quads = [
                            a32[pl.ds(t * wpr + k, rows // 4, stride=4 * wpr)]
                            for t in range(4)
                        ]
                        blocks += [
                            pltpu.bitcast(b, weight_dtype)
                            for b in _byte_transpose4(quads)
                        ]
                    act = jnp.concatenate(blocks, axis=1)
                else:
                    # Retile the gathered rows ([rows, L, 128] blocks) into the
                    # matmul layout [rows, h] as bf16, then cast back: cheaper
                    # than retiling the packed fp8 value; both casts are exact.
                    act = act_vmem[aslot, pl.ds(0, rows)].astype(jnp.bfloat16)
                    act = act.reshape(rows, h).astype(weight_dtype)
                act_scale = _tile_row_scales(act_scale_vmem[aslot], start, rows)
                # expert_ffn_fp8 up to its down matmul, which runs in the store
                # branch below. The weights are refs, loaded where they are used.
                inter = w1.shape[-1] // 2  # i, or i_s on the shared visit
                acc1 = lax.dot_general(
                    act, w1[...], DN, preferred_element_type=jnp.float32
                )
                mid_q, mid_scale = _intermediate_rows(
                    acc1, act_scale, inter, w1s=w1_scale[...], act_fn="silu", w1b=None
                )
                # The intermediate is requantized, so it has a scale.
                assert mid_scale is not None
                for par in range(2):

                    @pl.when(p == par)
                    def _(par=par):
                        # The down matmul and the epilogue in the branch that
                        # stores them, so the bf16 rows go straight into
                        # out_vmem[par] rather than through a value finished
                        # before the branch (same operations, same bits).
                        acc2 = lax.dot_general(
                            mid_q, w2[...], DN, preferred_element_type=jnp.float32
                        )
                        out = acc2 * mid_scale * w2_scale[...]
                        if out_by_pairs:
                            # Store straight into the per-row slab layout. In a
                            # u32 view of out_vmem[par], word-row r * L/2 + q
                            # holds lane blocks 2q (low half) and 2q + 1 (high
                            # half) of row r: pack each pair to bf16 elementwise
                            # and store it with one sublane-strided store per 8
                            # rows. Same products and rounding, no relayout.
                            half = lane_blocks // 2
                            out32 = (
                                out_vmem.at[par]
                                .bitcast(jnp.uint32)
                                .reshape(tile_rows * half, _LANES)
                            )
                            for q in range(half):
                                pair = [
                                    out[
                                        :,
                                        (2 * q + j) * _LANES : (2 * q + j + 1) * _LANES,
                                    ]
                                    for j in range(2)
                                ]
                                out32[pl.ds(q, rows, stride=half)] = (
                                    pltpu.pack_elementwise(
                                        pair, packed_dtype=jnp.bfloat16
                                    )
                                )
                        else:
                            # Retile into the per-row slab layout while still
                            # f32 (8x8 sublane blocks), then pack to bf16.
                            out = out.reshape(rows, lane_blocks, _LANES)
                            out_vmem[par, pl.ds(0, rows)] = out.astype(jnp.bfloat16)

            def compute_visit(rows):
                @pl.when(is_shared(v))
                def _():
                    compute(
                        rows,
                        shared_w1_vmem,
                        shared_w2_vmem,
                        shared_w1_scale_vmem,
                        shared_w2_scale_vmem,
                    )

                @pl.when(jnp.logical_not(is_shared(v)))
                def _():
                    compute(
                        rows,
                        w1_ring.at[slot],
                        w2_ring.at[slot],
                        w1_scale_ring.at[slot],
                        w2_scale_ring.at[slot],
                    )

            for r, rows in enumerate(rungs):
                below = rungs[r - 1] if r else 0

                @pl.when(jnp.logical_and(live > below, live <= rows))
                def _(rows=rows):
                    compute_visit(rows)

            out_copy(start, p, live).start()
            n_out0 = jnp.where(p == 0, live, n_out0)
            n_out1 = jnp.where(p == 1, live, n_out1)
            return vn, tn, g + 1, n_out0, n_out1, vi_n, ti_n

        _, _, _, n_out0, n_out1, _, _ = lax.while_loop(
            lambda carry: carry[0] < n_visit, tile_body, (z, z, z, z, z, vi, ti)
        )

        for par, n in ((0, n_out0), (1, n_out1)):

            @pl.when(n > 0)
            def _(par=par, n=n):
                out_wait(par, n)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[hbm] * 11,
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM((n_act, tile_rows, lane_blocks, _LANES), weight_dtype),
                pltpu.VMEM((n_act, scale_window, _LANES), jnp.float32),
                pltpu.VMEM((2, tile_rows, lane_blocks, _LANES), jnp.bfloat16),
                pltpu.VMEM((nb, h, 2 * i), weight_dtype),
                pltpu.VMEM((nb, i, h), weight_dtype),
                pltpu.VMEM((nb, 1, 2 * i), jnp.float32),
                pltpu.VMEM((nb, 1, h), jnp.float32),
                pltpu.SMEM((2 * window,), jnp.int32),
                pltpu.SemaphoreType.DMA((n_act,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((nb,)),
                pltpu.VMEM((h, 2 * i_s), weight_dtype),
                pltpu.VMEM((i_s, h), weight_dtype),
                pltpu.VMEM((1, 2 * i_s), jnp.float32),
                pltpu.VMEM((1, h), jnp.float32),
                pltpu.SemaphoreType.DMA(()),
            ),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((slab_rows, lane_blocks, _LANES), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=v2_host.vmem_limit(),
            disable_bounds_checks=True,
        ),
        name=(f"fused_moe_prefill-t_{tokens}-e_{experts}-i_{i}-is_{i_s}-m_{tile_rows}"),
    )
    return call, scale_rows, token_rows


def _routing_plan(topk_ids, route_valid, experts, row_scale):
    """Expert-major slab order for the routed rows, with no scatter or gather
    over the routes.

    One sort orders the routes by expert, carrying each route's id and its
    token's activation scale; a second sort of that permutation gives its
    inverse (each route's slab row). While the ids fit, each is a single
    unstable sort of packed uint32 keys: the keys are distinct, so it gives
    the stable key/value sort's permutation, at a lower cost. Routes where
    route_valid [T, K] is False get the sentinel expert E, sort last and are
    never visited.

    Returns the token of each slab row [R], the slab row of each route [T, K],
    the activation scale of each slab row [R], the visit tables (expert, first
    row, rows; E + _VISIT_SLACK entries each) and the routed visit count.
    """
    tokens, topk = topk_ids.shape
    routes = tokens * topk
    route_expert = jnp.where(
        route_valid.reshape(-1), topk_ids.reshape(-1).astype(jnp.int32), experts
    )
    iota = jnp.arange(routes, dtype=jnp.int32)
    route_scale = jnp.broadcast_to(row_scale[:, None], (tokens, topk)).reshape(-1)
    if routes < 1 << 20 and experts < 1 << 11:
        # One sort on (expert << 20) | route (expert ids up to E < 2^11, route
        # ids < 2^20): the route id breaks ties within an expert, so this is
        # the stable expert order.
        key = (route_expert.astype(jnp.uint32) << 20) | iota.astype(jnp.uint32)
        key, sorted_scale = lax.sort((key, route_scale), num_keys=1, is_stable=False)
        order = (key & ((1 << 20) - 1)).astype(jnp.int32)
    else:
        _, order, sorted_scale = lax.sort(
            (route_expert, iota, route_scale), num_keys=1, is_stable=True
        )
    if routes <= 1 << 16:
        # One sort on (route << 16) | slab row, the route being order[slab row]:
        # entry r ends up holding route r's slab row in its low 16 bits.
        key = (order.astype(jnp.uint32) << 16) | iota.astype(jnp.uint32)
        slab_row = (lax.sort(key, is_stable=False) & 0xFFFF).astype(jnp.int32)
    else:
        _, slab_row = lax.sort((order, iota), num_keys=1, is_stable=False)
    counts = jnp.sum(
        route_expert[None, :] == jnp.arange(experts, dtype=jnp.int32)[:, None],
        axis=1,
        dtype=jnp.int32,
    )
    first = jnp.cumsum(counts) - counts
    active = counts > 0
    (visit_expert,) = jnp.nonzero(
        active, size=experts + _VISIT_SLACK, fill_value=experts
    )
    visit_expert = visit_expert.astype(jnp.int32)
    counts = jnp.concatenate([counts, jnp.zeros((1,), jnp.int32)])
    first = jnp.concatenate([first, jnp.zeros((1,), jnp.int32)])
    return (
        order // topk,
        slab_row.reshape(tokens, topk),
        sorted_scale,
        visit_expert,
        first[visit_expert],
        counts[visit_expert],
        jnp.sum(active, dtype=jnp.int32),
    )


def _quantize_rows(x, dtype):
    """Per-row quantization straight into the [T, h / 128, 128] layout the
    expert kernel gathers rows from (one XLA fusion, no separate retile).
    Returns (q, scale [T]); an all-zero row gets scale 0."""
    tokens, h = x.shape
    x3 = x.reshape(tokens, h // _LANES, _LANES)
    amax = jnp.max(jnp.abs(x3), axis=(1, 2), keepdims=True)
    scale = amax.astype(jnp.float32) / float(jnp.finfo(dtype).max)
    usable = jnp.isfinite(scale) & (scale > 0)
    scale = jnp.where(usable, scale, 0.0)
    inv = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0)
    q = jnp.where(usable, x3 * inv.astype(x.dtype), jnp.zeros_like(x3)).astype(dtype)
    return q, scale.reshape(-1)


def _psum_rows(y, axis_name):
    """psum(y, axis_name) as a reduce-scatter over the rows plus an all-gather.

    The same sum up to rounding order, but never one all-reduce of the whole
    chunk: offloaded to the SparseCore, that all-reduce rounded the sum so that
    the first token's attention sink formed too weakly for some GLM-5.2 prompts
    (fp8 KV cache) and generation degenerated; this form served them correctly.
    Rows are zero-padded to a multiple of the axis size and sliced back, so
    every chunk shape takes this form."""
    rows = y.shape[0]
    pad = -rows % lax.axis_size(axis_name)
    if pad:
        y = jnp.pad(y, ((0, pad),) + ((0, 0),) * (y.ndim - 1))
    y = lax.psum_scatter(y, axis_name, scatter_dimension=0, tiled=True)
    y = lax.all_gather(y, axis_name, axis=0, tiled=True)
    return y[:rows] if pad else y


def _combine(slab, slab_row, coeff, axis_name):
    """out[t] = psum(sum_k coeff[t, k] * slab[slab_row[t, k]]) in equal token
    chunks of at most _COMBINE_CHUNK_TOKENS, each chunk reduced by _psum_rows."""
    tokens, slots = slab_row.shape
    h = slab.shape[1] * _LANES
    chunks = next(
        n
        for n in range(1, tokens + 1)
        if not tokens % n and tokens // n <= _COMBINE_CHUNK_TOKENS
    )
    chunk = tokens // chunks
    outs = []
    for c in range(chunks):
        rows = slice(c * chunk, (c + 1) * chunk)
        if combine_step_tokens(chunk):
            y = build_combine_kernel(
                t_local=chunk,
                topk=slots,
                hidden=h,
                lane_blocks=h // _LANES,
                wire_dtype=jnp.bfloat16,
                out_dtype=jnp.bfloat16,
            )(slab, coeff[rows], slab_row[rows])
        else:  # chunk sizes the Pallas combine does not tile
            gathered = slab.reshape(-1, h)[slab_row[rows]].astype(jnp.float32)
            y = jnp.sum(gathered * coeff[rows][..., None], axis=1)
            y = y.astype(jnp.bfloat16)
        if outs:
            # Tie chunk c's input to chunk c - 1's reduced output so XLA keeps
            # the reductions separate and each runs under the next combine.
            y, _ = lax.optimization_barrier((y, outs[-1]))
        outs.append(_psum_rows(y, axis_name))
    return outs[0] if chunks == 1 else jnp.concatenate(outs, 0)


def fused_moe_prefill(
    x,
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
    axis_name,
    expert_start=None,
    routed_scale=1.0,
):
    """One rank's MoE block for prefill-sized token counts, under shard_map.

    Args:
      x: [T, h] bf16 hidden states, the same on every rank.
      w1: [E, h, 2i] FP8 gate/up weights of the rank's E local experts.
      w2: [E, i, h] FP8 down weights.
      w1_scale: per-channel f32 scales of w1, [E, 1, 1, 2i] (any shape with
        E * 2i elements).
      w2_scale: per-channel f32 scales of w2, [E, 1, 1, h] likewise.
      shared_w1: [h, 2 i_s] FP8 gate/up of the rank's shared-expert slice.
      shared_w1_scale: [2 i_s] f32.
      shared_w2: [i_s, h] FP8.
      shared_w2_scale: [h] f32.
      shared_gate_weight: shared-expert gate weight, h elements (the gate is
        sigmoid(x . w)), or None for a shared expert without a gate
        (GLM-5.2), whose rows enter the combine with coefficient 1.
      topk_weights: [T, K] router weights. Routes with weight 0 (padded rows
        under TPU_MOE_SKIP_PADDED_TOKENS) are not computed.
      topk_ids: [T, K] global expert ids.
      axis_name: mesh axis the per-rank partials are summed over.
      expert_start: None when every rank holds all experts (TP experts);
        with expert parallelism the global id of the rank's first expert,
        and routes to other ranks' experts are not computed here.
      routed_scale: multiplies the routed coefficients (the router weights,
        in f32) and not the shared row: GLM-5.2's routed_scaling_factor.

    Returns:
      [T, h] bf16: shared + routed_scale * routed output summed over
      `axis_name`.
    """
    tokens, h = x.shape
    experts, _, two_i = w1.shape
    i_s = shared_w2.shape[0]
    routes = tokens * topk_ids.shape[1]
    weight_dtype = w1.dtype
    call, scale_rows, token_rows = _build_expert_kernel(
        experts=experts,
        h=h,
        i=two_i // 2,
        i_s=i_s,
        tokens=tokens,
        routes=routes,
        tile_rows=_tile_rows(tokens),
        weight_dtype=weight_dtype,
    )

    x_q, x_scale = _quantize_rows(x, weight_dtype)
    route_valid = topk_weights != 0
    if expert_start is not None:
        topk_ids = topk_ids - expert_start
        local = (topk_ids >= 0) & (topk_ids < experts)
        route_valid = route_valid & local
        topk_ids = jnp.where(local, topk_ids, 0)
    (
        row_token,
        slab_row,
        sorted_scale,
        visit_expert,
        visit_start,
        visit_rows,
        routed_visits,
    ) = _routing_plan(topk_ids, route_valid, experts, x_scale)
    # The shared expert is one more visit, over every token.
    visit_expert = visit_expert.at[routed_visits].set(experts)
    visit_start = visit_start.at[routed_visits].set(routes)
    visit_rows = visit_rows.at[routed_visits].set(tokens)
    row_scale = jnp.concatenate([sorted_scale, x_scale])
    row_scale = jnp.pad(row_scale, (0, scale_rows * _LANES - row_scale.shape[0]))
    slab = call(
        visit_expert,
        visit_start,
        visit_rows,
        (routed_visits + 1).reshape(1),
        x_q,
        row_scale.reshape(-1, _LANES),
        jnp.pad(row_token, (0, token_rows - routes)),
        w1,
        w2,
        w1_scale.reshape(experts, 1, two_i).astype(jnp.float32),
        w2_scale.reshape(experts, 1, h).astype(jnp.float32),
        shared_w1.reshape(1, h, 2 * i_s),
        shared_w2.reshape(1, i_s, h),
        shared_w1_scale.reshape(1, 1, 2 * i_s).astype(jnp.float32),
        shared_w2_scale.reshape(1, 1, h).astype(jnp.float32),
    )

    # ---- combine: top-k routed rows + the gated shared row per token ----
    if shared_gate_weight is None:  # no gate: the shared row is added as is
        gate = jnp.ones((tokens, 1), jnp.float32)
    else:
        gate = jax.nn.sigmoid(
            jnp.dot(
                x,
                shared_gate_weight.reshape(h, 1).astype(x.dtype),
                preferred_element_type=jnp.float32,
            )
        )
    shared_row = routes + jnp.arange(tokens, dtype=jnp.int32)[:, None]
    # Rows of routes that were not computed are never written; point those
    # routes at a written row so coefficient 0 never meets uninitialized memory.
    slab_row = jnp.where(route_valid, slab_row, shared_row)
    coeff = jnp.where(route_valid, topk_weights.astype(jnp.float32), 0.0)
    # A Python-level branch: at 1.0 the traced program is the unscaled one.
    if routed_scale != 1.0:
        coeff = coeff * routed_scale
    return _combine(
        slab,
        jnp.concatenate([slab_row, shared_row], 1),
        jnp.concatenate([coeff, gate], 1),
        axis_name,
    )
