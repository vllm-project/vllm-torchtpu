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
the router-weighted sum of its top-k routed rows plus its shared row. The sum
over the ranks is a reduce-scatter plus an all-gather issued per combine chunk,
so each chunk's reduction runs under the next chunk's combine.

With expert parallelism and TP attention a rank holds whole experts instead of
slices and the call is the same, restricted to the rank's expert range.
Experts too large for three whole-expert slots (GLM-5.2 at EP16: 37.7 MB of
FP8 each) run blocked mode instead: each expert's rows in chunks with an FP32
running sum in VMEM while the expert streams through a ring of intermediate
blocks, and a combine that reads only the live slots of each token
(plan_blocks).

Numerics: hidden states are quantized to FP8 once per call with one scale per
row per 512 columns, as gmm_v2 quantizes them with the kernel off; the
activated intermediate likewise per row per 512 columns of the expert (per
row per block in blocked mode, in 512-column pieces of wider blocks). Both
matmuls are FP8 x FP8; each 512-column block's partial is scaled and summed in
FP32. Each route's
rows are BF16 in the slab (the down projection's output is a partial sum over
an intermediate shard; the grouped GEMM path writes BF16 there too). Router
top-k stays with the caller.
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
    _absmax_rows,
    _intermediate_rows,
    _quant_apply,
    _row_scale,
    _tile_row_scales,
    build_combine_kernel,
    combine_step_tokens,
)

_LANES = 128
# Columns per FP8 activation scale: each row of the hidden states (and of the
# activated intermediate) gets one scale per block of this many columns, the
# block gmm_v2 uses on the GMM path, so one outlier channel coarsens only its
# own block. Each block's partial product is scaled before it is summed.
_Q_BLOCK = 512
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

# Blocked mode. (block width, ring depth) candidates in order of preference,
# from the GLM-5.2 EP16 sweep: 128-row tiles (T < 8192) stream 512-wide blocks
# through 3 slots; 256-row tiles 1024-wide blocks through 2, half the
# running-sum passes, which dominate once experts hold hundreds of rows.
_BLOCKS_TALL_TILES = ((1024, 2),)
_BLOCKS = ((512, 3), (256, 3), (128, 3))
# Most rows of one chunk; the planner takes the tallest multiple of the tile
# height up to this that fits VMEM.
_MAX_CHUNK_ROWS = 512
# Output columns per down-projection matmul: bounds the live f32 result,
# which goes straight into the running sum.
_DOWN_CHUNK = 512
# VMEM held back from the planned buffers for Mosaic's internal scratch (the
# live f32 matmul results of a tile).
_BLOCKED_RESERVE = 6 << 20

# Sparse combine (blocked mode): tokens per step, tokens per relayout group,
# row buffers, and steps of rows in flight ahead of the reduction.
_SC_TOKENS = 64
_SC_GROUP = 16
_SC_PARITIES = 4
_SC_LEAD = 2


def _unpack_scales(packed, n_q):
    """The n_q activation scales of a tile as [m, 1] f32 columns, from the
    [m, 1] int32 columns that carry them two per word (_pack_scales): the
    first scale in the high 16 bits, the second in the low, each a bf16."""
    out = []
    for v in packed:
        out.append(lax.bitcast_convert_type(v & jnp.int32(-65536), jnp.float32))
        out.append(lax.bitcast_convert_type(lax.shift_left(v, 16), jnp.float32))
    return out[:n_q]


def _gate_up(act, act_scales, w1):
    """act [m, h] fp8 @ w1 [h, 2 width] in f32, each _Q_BLOCK-row slab of w1
    against its column block of act, scaled by that block's row scales
    (act_scales: one [m, 1] column per block) before the sum."""
    acc = None
    for k, scale in enumerate(act_scales):
        cols = slice(k * _Q_BLOCK, (k + 1) * _Q_BLOCK)
        part = lax.dot_general(
            act[:, cols], w1[cols, :], DN, preferred_element_type=jnp.float32
        )
        part = part * scale
        acc = part if acc is None else acc + part
    return acc


def _quantize_mid(acc1, width, w1_scale):
    """acc1's gate and up halves (x scales applied) through SiLU and the
    weight scale, quantized per row per min(_Q_BLOCK, width) columns.
    Returns [(fp8 block, its [m, 1] row scale)] in column order."""
    mid, _ = _intermediate_rows(
        acc1, None, width, w1s=w1_scale, act_fn="silu", w1b=None, requantize=False
    )
    qa = min(_Q_BLOCK, width)
    blocks = []
    for k in range(0, width, qa):
        blk = mid[:, k : k + qa]
        scale, sinv, usable = _row_scale(_absmax_rows(blk))
        blocks.append((_quant_apply(blk, sinv.astype(blk.dtype), usable), scale))
    return blocks


def _down(blocks, w2, cols):
    """The down projection of _quantize_mid's blocks onto w2's columns `cols`,
    each block's partial scaled by its row scales, summed in f32 (the weight
    scale still owed)."""
    acc, k = None, 0
    for mq, scale in blocks:
        n = mq.shape[1]
        part = lax.dot_general(
            mq, w2[k : k + n, cols], DN, preferred_element_type=jnp.float32
        )
        part = part * scale
        acc = part if acc is None else acc + part
        k += n
    return acc


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
    n_q = h // _Q_BLOCK
    n_p = _cdiv(n_q, 2)
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
          x_hbm [T, L, 128] fp8: hidden states quantized per row per
            _Q_BLOCK columns.
          row_scale_hbm [P, scale_rows, 128] int32: the activation scales of
            each slab row, two bf16 per word (_pack_scales), one plane per
            pair of column blocks (P = ceil(h / (2 * _Q_BLOCK))).
          row_token_hbm [token_rows] int32: token of each routed slab row.
          w1_hbm [E, h, 2i], w2_hbm [E, i, h]: routed expert weights.
          w1_scale_hbm [E, 1, 2i], w2_scale_hbm [E, 1, h] f32: their scales.
          shared_w1_hbm [1, h, 2 i_s], shared_w2_hbm [1, i_s, h]: shared expert.
          shared_w1_scale_hbm [1, 1, 2 i_s], shared_w2_scale_hbm [1, 1, h] f32.
        Output in HBM:
          slab_hbm [slab_rows, L, 128] bf16: unweighted rows, slab order.
        Scratch:
          act_vmem [n_act, tile_rows, L, 128] fp8: gathered activation rows.
          act_scale_vmem [n_act, P, scale_window, 128] int32: their scales.
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

        shared_copies = _shared_copies(
            (shared_w1_hbm, shared_w2_hbm, shared_w1_scale_hbm, shared_w2_scale_hbm),
            (
                shared_w1_vmem,
                shared_w2_vmem,
                shared_w1_scale_vmem,
                shared_w2_scale_vmem,
            ),
            shared_sem,
        )

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
                row_scale_hbm.at[:, pl.ds(start // _LANES, scale_window)],
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
                _gather_rows(
                    x_hbm,
                    act_vmem.at[slot],
                    act_sems.at[slot],
                    token_window,
                    wslot * window + lax.rem(start, _LANES),
                    tile_live(v, t),
                )

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
                    # lane blocks 4k..4k+3 (block 4k + j in byte j), which are
                    # also the 512 columns of activation scale block k. So load
                    # rows t, t + 4, ... of word-row k with one sublane-strided
                    # load per t and transpose bytes across the four loads:
                    # the gathered bytes as they are, no relayout, no casts.
                    wpr = lane_blocks // 4
                    a32 = (
                        act_vmem.at[aslot]
                        .bitcast(jnp.uint32)
                        .reshape(tile_rows * wpr, _LANES)
                    )
                    act_lanes = []
                    for k in range(wpr):
                        quads = [
                            a32[pl.ds(t * wpr + k, rows // 4, stride=4 * wpr)]
                            for t in range(4)
                        ]
                        act_lanes += [
                            pltpu.bitcast(b, weight_dtype)
                            for b in _byte_transpose4(quads)
                        ]
                    act = jnp.concatenate(act_lanes, axis=1)
                else:
                    # Retile the gathered rows ([rows, L, 128] blocks) into the
                    # matmul layout [rows, h] as bf16, then cast back: cheaper
                    # than retiling the packed fp8 value; both casts are exact.
                    act = act_vmem[aslot, pl.ds(0, rows)].astype(jnp.bfloat16)
                    act = act.reshape(rows, h).astype(weight_dtype)
                act_scales = _unpack_scales(
                    [
                        _tile_row_scales(act_scale_vmem[aslot, k], start, rows)
                        for k in range(n_p)
                    ],
                    n_q,
                )
                # Gate/up and the requantized intermediate; the down matmul
                # runs in the store branch below. The weights are refs, loaded
                # where they are used.
                acc1 = _gate_up(act, act_scales, w1)
                inter = w1.shape[-1] // 2  # i, or i_s on the shared visit
                blocks = _quantize_mid(acc1, inter, w1_scale[...])
                for par in range(2):

                    @pl.when(p == par)
                    def _(par=par):
                        # The down matmul and the epilogue in the branch that
                        # stores them, so the bf16 rows go straight into
                        # out_vmem[par] rather than through a value finished
                        # before the branch (same operations, same bits).
                        acc2 = _down(blocks, w2, slice(None))
                        out = acc2 * w2_scale[...]
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
                pltpu.VMEM((n_act, n_p, scale_window, _LANES), jnp.int32),
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
                *_shared_scratch(h, i_s, weight_dtype),
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


def _shared_copies(srcs, dsts, sem):
    """The shared expert's four transfers (w1, w2 and their scales)."""
    return [pltpu.make_async_copy(s.at[0], d, sem) for s, d in zip(srcs, dsts)]


def _shared_scratch(h, i_s, weight_dtype):
    """VMEM for the shared expert, loaded once, and its DMA semaphore."""
    return (
        pltpu.VMEM((h, 2 * i_s), weight_dtype),
        pltpu.VMEM((i_s, h), weight_dtype),
        pltpu.VMEM((1, 2 * i_s), jnp.float32),
        pltpu.VMEM((1, h), jnp.float32),
        pltpu.SemaphoreType.DMA(()),
    )


def _gather_rows(x_hbm, dst, sem, token_window, ptr0, live):
    """Start one DMA per activation row: row r of `dst` gets token
    token_window[ptr0 + r] (the flat token ring), for whole blocks of
    _GATHER_UNROLL rows covering `live` (dead rows read in-bounds table
    entries)."""

    # The carry is the flat-ring index of the block's first row, so each row
    # costs one SMEM load at a constant offset plus its DMA (no 2-D index
    # arithmetic per row).
    def block(b, ptr):
        r0 = b * _GATHER_UNROLL
        for u in range(_GATHER_UNROLL):
            token = token_window[ptr + u]
            pltpu.make_async_copy(x_hbm.at[token], dst.at[r0 + u], sem).start()
        return ptr + _GATHER_UNROLL

    lax.fori_loop(0, _cdiv(live, _GATHER_UNROLL), block, ptr0)


# ---- Blocked mode: experts streamed in intermediate blocks ----------------
#
# For experts whose whole-expert weight slots do not fit VMEM (GLM-5.2 at EP:
# h 6144, i 2048). Each routed visit is cut into chunks of at most chunk_rows
# rows; a chunk's activation rows stay resident (double-buffered across
# chunks) with an f32 running sum of its rows in VMEM, and the expert streams
# through a ring of (expert, block) units: block b holds w1's gate columns
# [b ib, (b + 1) ib) beside the matching up columns and w2's rows
# [b ib, (b + 1) ib). Every block adds its partial down product for every tile
# of the chunk; the last block finishes each tile into the bf16 slab, which
# keeps the whole-expert mode's layout. Exact in real arithmetic (SiLU(gate) *
# up is elementwise in the intermediate and the down projection sums over it).
# Each chunk reads its expert's weights once. The shared expert is loaded once
# and runs as one unit per chunk of its rows, after the routed chunks.


@functools.cache
def _build_blocked_expert_kernel(
    *, experts, h, i, i_s, tokens, routes, ib, nb, chunk_rows, tile_rows, weight_dtype
):
    """The per-rank blocked expert kernel; same operands, slab layout and
    returns as _build_expert_kernel, except that the visit tables hold the
    routed visits only."""
    n_blk = i // ib
    tiles = chunk_rows // tile_rows
    window = chunk_rows + _LANES  # a chunk's rows start anywhere in an aligned window
    scale_window = tile_rows // _LANES + 1
    n_q = h // _Q_BLOCK
    n_p = _cdiv(n_q, 2)
    lane_blocks = h // _LANES
    shared = experts
    slab_rows = routes + tokens + tile_rows
    scale_rows = _cdiv(slab_rows, _LANES) + 1
    token_rows = _cdiv(routes, _LANES) * _LANES + window
    max_chunks = (
        experts + _cdiv(routes, chunk_rows) + _cdiv(tokens, chunk_rows) + _VISIT_SLACK
    )

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
        acc_vmem,
        out_vmem,
        w1_ring,
        w2_ring,
        w1_scale_ring,
        w2_scale_ring,
        token_window,
        act_sems,
        window_sems,
        out_sem,
        ring_sems,
        shared_w1_vmem,
        shared_w2_vmem,
        shared_w1_scale_vmem,
        shared_w2_scale_vmem,
        shared_sem,
        chunk_expert,
        chunk_start,
        chunk_live,
    ):
        """One rank's chunk loop. num_visits counts routed visits. Scratch
        beyond the whole-expert kernel's: act_vmem [2, chunk_rows, L, 128]
        and act_scale_vmem [2, tiles, P, scale_window, 128] (a chunk's rows,
        double-buffered), acc_vmem [chunk_rows, h] f32 (the running sum),
        out_vmem [chunk_rows, L, 128] bf16, a ring of nb (expert, block)
        units, and the chunk table chunk_expert / chunk_start / chunk_live
        [max_chunks] int32 (SMEM)."""

        # ---- chunk table: every routed visit cut into chunks of at most
        # chunk_rows rows, in visit order, then the shared expert's rows
        # (slab rows routes + [0, tokens)) ----
        def put(c, e, start, rows):
            chunk_expert[c] = e
            chunk_start[c] = start
            chunk_live[c] = rows
            return c + 1

        def fill_visit(v, c):
            e, start, rows = visit_expert[v], visit_start[v], visit_rows[v]

            def fill(q, c):
                return put(
                    c,
                    e,
                    start + q * chunk_rows,
                    jnp.minimum(chunk_rows, rows - q * chunk_rows),
                )

            return lax.fori_loop(0, _cdiv(rows, chunk_rows), fill, c)

        n_routed = lax.fori_loop(0, num_visits[0], fill_visit, jnp.int32(0))

        def fill_shared(q, c):
            return put(
                c,
                shared,
                routes + q * chunk_rows,
                jnp.minimum(chunk_rows, tokens - q * chunk_rows),
            )

        n_chunk = lax.fori_loop(0, _cdiv(tokens, chunk_rows), fill_shared, n_routed)
        n_units = n_routed * n_blk

        def is_shared(c):
            return chunk_expert[c] == shared

        # ---- weights: unit u = block u % n_blk of routed chunk u // n_blk ----
        def unit_copies(u):
            e = chunk_expert[u // n_blk]
            b = lax.rem(u, n_blk)
            slot = lax.rem(u, nb)
            gate = pl.multiple_of(b * ib, ib)
            up = pl.multiple_of(i + b * ib, ib)
            sem = ring_sems.at[slot]
            copies = []
            for dst, src in ((0, gate), (ib, up)):
                copies.append(
                    pltpu.make_async_copy(
                        w1_hbm.at[e, :, pl.ds(src, ib)],
                        w1_ring.at[slot, :, pl.ds(dst, ib)],
                        sem,
                    )
                )
                copies.append(
                    pltpu.make_async_copy(
                        w1_scale_hbm.at[e, :, pl.ds(src, ib)],
                        w1_scale_ring.at[slot, :, pl.ds(dst, ib)],
                        sem,
                    )
                )
            copies.append(
                pltpu.make_async_copy(
                    w2_hbm.at[e, pl.ds(gate, ib)], w2_ring.at[slot], sem
                )
            )
            copies.append(
                pltpu.make_async_copy(w2_scale_hbm.at[e], w2_scale_ring.at[slot], sem)
            )
            return copies

        def start_unit(u):
            @pl.when(u < n_units)
            def _():
                for c in unit_copies(u):
                    c.start(priority=_WEIGHT_DMA_PRIORITY)

        def wait_unit(u):
            for c in unit_copies(u):
                c.wait()

        shared_copies = _shared_copies(
            (shared_w1_hbm, shared_w2_hbm, shared_w1_scale_hbm, shared_w2_scale_hbm),
            (
                shared_w1_vmem,
                shared_w2_vmem,
                shared_w1_scale_vmem,
                shared_w2_scale_vmem,
            ),
            shared_sem,
        )

        # ---- a chunk's rows: token window, row gather, per-tile scales ----
        def window_copy(c, slot):
            start = chunk_start[c]
            aligned = pl.multiple_of(start - lax.rem(start, _LANES), _LANES)
            lo = slot * window  # the slot's first entry in the flat ring
            if not isinstance(lo, int):
                lo = pl.multiple_of(lo, _LANES)
            return pltpu.make_async_copy(
                row_token_hbm.at[pl.ds(aligned, window)],
                token_window.at[pl.ds(lo, window)],
                window_sems.at[slot],
            )

        def start_window(c, slot):
            @pl.when(jnp.logical_and(c < n_chunk, jnp.logical_not(is_shared(c))))
            def _():
                window_copy(c, slot).start()

        def n_tiles(c):
            return _cdiv(chunk_live[c], tile_rows)

        def scale_copy(c, buf, t):
            row = (chunk_start[c] + t * tile_rows) // _LANES
            return pltpu.make_async_copy(
                row_scale_hbm.at[:, pl.ds(row, scale_window)],
                act_scale_vmem.at[buf, t],
                act_sems.at[buf],
            )

        def issue_chunk(c, buf):
            """Start chunk c's rows and scale windows into buffer buf (its
            token window, routed chunks only, is in window slot buf)."""
            start, live = chunk_start[c], chunk_live[c]

            @pl.when(is_shared(c))
            def _():
                pltpu.make_async_copy(
                    x_hbm.at[pl.ds(start - routes, live)],
                    act_vmem.at[buf, pl.ds(0, live)],
                    act_sems.at[buf],
                ).start()

            @pl.when(jnp.logical_not(is_shared(c)))
            def _():
                window_copy(c, buf).wait()
                _gather_rows(
                    x_hbm,
                    act_vmem.at[buf],
                    act_sems.at[buf],
                    token_window,
                    buf * window + lax.rem(start, _LANES),
                    live,
                )

            def scales(t, carry):
                scale_copy(c, buf, t).start()
                return carry

            lax.fori_loop(0, n_tiles(c), scales, 0)

        def wait_chunk(c, buf):
            live = chunk_live[c]
            n = jnp.where(
                is_shared(c), live, _cdiv(live, _GATHER_UNROLL) * _GATHER_UNROLL
            )
            pltpu.make_async_copy(
                act_vmem.at[buf, pl.ds(0, n)],
                act_vmem.at[buf, pl.ds(0, n)],
                act_sems.at[buf],
            ).wait()
            nt = n_tiles(c)
            pltpu.make_async_copy(
                act_scale_vmem.at[buf, pl.ds(0, nt)],
                act_scale_vmem.at[buf, pl.ds(0, nt)],
                act_sems.at[buf],
            ).wait()

        def out_copy(start, n):
            return pltpu.make_async_copy(
                out_vmem.at[pl.ds(0, n)], slab_hbm.at[pl.ds(start, n)], out_sem
            )

        def out_wait(n):
            pltpu.make_async_copy(
                out_vmem.at[pl.ds(0, n)], out_vmem.at[pl.ds(0, n)], out_sem
            ).wait()

        def tile_ffn(c, buf, t, w1, w2, w1_scale, w2_scale, mode):
            """Tile t of chunk c through one unit: gate/up, the activation and
            its requantization per row per 512 columns, then the down
            projection in column
            chunks of _DOWN_CHUNK straight into the running sum (mode "first"
            stores, "mid" adds) or, finished with the down scale, into the
            bf16 output rows ("last" adds the running sum first, "only" has
            none)."""
            lo = pl.multiple_of(t * tile_rows, tile_rows)
            act = act_vmem[buf, pl.ds(lo, tile_rows)].reshape(tile_rows, h)
            act_scales = _unpack_scales(
                [
                    _tile_row_scales(
                        act_scale_vmem[buf, t, k], chunk_start[c] + lo, tile_rows
                    )
                    for k in range(n_p)
                ],
                n_q,
            )
            acc1 = _gate_up(act, act_scales, w1)
            blocks = _quantize_mid(acc1, w1.shape[-1] // 2, w1_scale)
            rows = pl.ds(lo, tile_rows)
            nc = min(_DOWN_CHUNK, h)
            for n in range(0, h, nc):
                cols = slice(n, n + nc)
                part = _down(blocks, w2, cols)
                if mode == "first":
                    acc_vmem[rows, cols] = part
                elif mode == "mid":
                    acc_vmem[rows, cols] += part
                else:
                    if mode == "last":
                        part = acc_vmem[rows, cols] + part
                    done = part * w2_scale[:, cols]
                    out_vmem[rows, n // _LANES : (n + nc) // _LANES] = done.reshape(
                        tile_rows, nc // _LANES, _LANES
                    ).astype(jnp.bfloat16)

        # ---- prologue ----
        for c in shared_copies:
            c.start(priority=_WEIGHT_DMA_PRIORITY)
        for k in range(nb - 1):
            start_unit(jnp.int32(k))
        z = jnp.int32(0)
        start_window(z, 0)
        issue_chunk(z, 0)  # every call has at least its shared chunk
        start_window(jnp.int32(1), 1)

        def chunk_body(c, n_prev):
            buf = lax.rem(c, 2)
            nt = n_tiles(c)
            wait_chunk(c, buf)

            @pl.when(c + 1 < n_chunk)
            def _():
                issue_chunk(c + 1, 1 - buf)

            start_window(c + 2, buf)

            @pl.when(jnp.logical_not(is_shared(c)))
            def _routed():
                def unit_wait(b):
                    u = c * n_blk + b
                    wait_unit(u)
                    start_unit(u + nb - 1)
                    return lax.rem(u, nb)

                def block_step(b, carry):
                    slot = unit_wait(b)

                    def tile_sum(t, carry):
                        args = (w1_ring.at[slot], w2_ring.at[slot], w1_scale_ring[slot])

                        @pl.when(b == 0)
                        def _():
                            tile_ffn(c, buf, t, *args, None, "first")

                        @pl.when(b > 0)
                        def _():
                            tile_ffn(c, buf, t, *args, None, "mid")

                        return carry

                    lax.fori_loop(0, nt, tile_sum, 0)
                    return carry

                lax.fori_loop(0, n_blk - 1, block_step, 0)
                slot = unit_wait(n_blk - 1)

                @pl.when(n_prev > 0)
                def _():
                    out_wait(n_prev)

                def last_tile(t, carry):
                    tile_ffn(
                        c,
                        buf,
                        t,
                        w1_ring.at[slot],
                        w2_ring.at[slot],
                        w1_scale_ring[slot],
                        w2_scale_ring[slot],
                        "only" if n_blk == 1 else "last",
                    )
                    return carry

                lax.fori_loop(0, nt, last_tile, 0)

            @pl.when(is_shared(c))
            def _shared():
                @pl.when(c == n_routed)
                def _():
                    for cp in shared_copies:
                        cp.wait()

                @pl.when(n_prev > 0)
                def _():
                    out_wait(n_prev)

                def shared_tile(t, carry):
                    tile_ffn(
                        c,
                        buf,
                        t,
                        shared_w1_vmem,
                        shared_w2_vmem,
                        shared_w1_scale_vmem[...],
                        shared_w2_scale_vmem[...],
                        "only",
                    )
                    return carry

                lax.fori_loop(0, nt, shared_tile, 0)

            live = chunk_live[c]
            out_copy(chunk_start[c], live).start()
            return live

        n_last = lax.fori_loop(0, n_chunk, chunk_body, z)

        @pl.when(n_last > 0)
        def _():
            out_wait(n_last)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[hbm] * 11,
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM((2, chunk_rows, lane_blocks, _LANES), weight_dtype),
                pltpu.VMEM((2, tiles, n_p, scale_window, _LANES), jnp.int32),
                pltpu.VMEM((chunk_rows, h), jnp.float32),
                pltpu.VMEM((chunk_rows, lane_blocks, _LANES), jnp.bfloat16),
                pltpu.VMEM((nb, h, 2 * ib), weight_dtype),
                pltpu.VMEM((nb, ib, h), weight_dtype),
                pltpu.VMEM((nb, 1, 2 * ib), jnp.float32),
                pltpu.VMEM((nb, 1, h), jnp.float32),
                pltpu.SMEM((2 * window,), jnp.int32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA(()),
                pltpu.SemaphoreType.DMA((nb,)),
                *_shared_scratch(h, i_s, weight_dtype),
                pltpu.SMEM((max_chunks,), jnp.int32),
                pltpu.SMEM((max_chunks,), jnp.int32),
                pltpu.SMEM((max_chunks,), jnp.int32),
            ),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((slab_rows, lane_blocks, _LANES), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=v2_host.vmem_limit(),
            disable_bounds_checks=True,
        ),
        name=(
            f"fused_moe_prefill-t_{tokens}-e_{experts}-i_{i}-is_{i_s}-m_{tile_rows}"
            f"-ib_{ib}-nb_{nb}-r_{chunk_rows}"
        ),
    )
    return call, scale_rows, token_rows


def _unit_bytes(h, width, wt):
    """One weight-ring slot: w1 [h, 2 width] + w2 [width, h] and f32 scales."""
    return 3 * h * width * wt + (2 * width + h) * 4


def _blocked_bytes(h, ib, i_s, nb, tile_rows, chunk_rows, wt):
    """VMEM of the blocked kernel: the unit ring, a chunk's double-buffered
    rows, running sum and output, its scale windows and the shared expert."""
    rows = chunk_rows * h * (2 * wt + 4 + 2)
    scales = (
        2
        * (chunk_rows // tile_rows)
        * _cdiv(h // _Q_BLOCK, 2)
        * (tile_rows // _LANES + 1)
        * _LANES
        * 4
    )
    return nb * _unit_bytes(h, ib, wt) + rows + scales + _unit_bytes(h, i_s, wt)


def plan_blocks(tokens, h, i, i_s, weight_dtype, i_block=None):
    """(block width, ring depth, chunk rows) of blocked mode for a call of
    `tokens` tokens, or None for whole experts; chosen at trace time.

    Whole experts whenever three whole-expert slots fit VMEM, the decode
    kernel's rule (plan_weight_ring): it depends on the expert's size only, so
    small experts (Qwen3.5, GLM-5.2 at TP) never switch mode with the token
    count. Otherwise the first of the preferred (width, depth) pairs whose
    width is narrower than and divides i, at the tallest chunk (a multiple of
    the tile height, at most _MAX_CHUNK_ROWS) that fits, falling back to
    narrower blocks if none does. No such width: whole experts. i_block forces
    blocked mode with that width (tests).
    """
    wt = jnp.dtype(weight_dtype).itemsize
    tile_rows = _tile_rows(tokens)
    if i_block is None:
        prefs = (_BLOCKS_TALL_TILES if tile_rows >= 256 else ()) + _BLOCKS
        candidates = [(w, n) for w, n in prefs if w < i and i % w == 0]
        capacity = pltpu.get_tpu_info().vmem_capacity_bytes
        if 3 * _unit_bytes(h, i, wt) <= capacity or not candidates:
            return None
    elif i % i_block or i_block % _LANES:
        raise ValueError(f"i_block {i_block} must divide i={i} in lanes of 128")
    else:
        candidates = [(i_block, 3)]
    budget = v2_host.vmem_limit() - _BLOCKED_RESERVE
    for k, (width, nb) in enumerate(candidates):
        rows = (_MAX_CHUNK_ROWS // tile_rows) * tile_rows
        while (
            rows > tile_rows
            and _blocked_bytes(h, width, i_s, nb, tile_rows, rows, wt) > budget
        ):
            rows -= tile_rows
        fits = _blocked_bytes(h, width, i_s, nb, tile_rows, rows, wt) <= budget
        if fits or k == len(candidates) - 1:
            return width, nb, rows
    raise AssertionError("unreachable")


def _routing_plan(topk_ids, route_valid, experts, row_scale):
    """Expert-major slab order for the routed rows, with no scatter or gather
    over the routes.

    One sort orders the routes by expert, carrying each route's id and its
    token's packed activation scales (row_scale [T, P] int32, one payload per
    word; a gather instead would be a slow TensorCore index gather); a second
    sort of that permutation gives its inverse (each route's slab row). While
    the ids fit, each is a single unstable sort of packed uint32 keys: the
    keys are distinct, so it gives the stable key/value sort's permutation, at
    a lower cost. Routes where route_valid [T, K] is False get the sentinel
    expert E, sort last and are never visited.

    Returns the token of each slab row [R], the slab row of each route [T, K],
    the packed activation scales of each slab row [P, R], the visit tables
    (expert, first row, rows; E + _VISIT_SLACK entries each) and the routed
    visit count.
    """
    tokens, topk = topk_ids.shape
    routes = tokens * topk
    route_expert = jnp.where(
        route_valid.reshape(-1), topk_ids.reshape(-1).astype(jnp.int32), experts
    )
    iota = jnp.arange(routes, dtype=jnp.int32)
    route_scales = [
        jnp.broadcast_to(row_scale[:, k, None], (tokens, topk)).reshape(-1)
        for k in range(row_scale.shape[1])
    ]
    if routes < 1 << 20 and experts < 1 << 11:
        # One sort on (expert << 20) | route (expert ids up to E < 2^11, route
        # ids < 2^20): the route id breaks ties within an expert, so this is
        # the stable expert order.
        key = (route_expert.astype(jnp.uint32) << 20) | iota.astype(jnp.uint32)
        key, *sorted_scales = lax.sort(
            (key, *route_scales), num_keys=1, is_stable=False
        )
        order = (key & ((1 << 20) - 1)).astype(jnp.int32)
    else:
        _, order, *sorted_scales = lax.sort(
            (route_expert, iota, *route_scales), num_keys=1, is_stable=True
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
        jnp.stack(sorted_scales),
        visit_expert,
        first[visit_expert],
        counts[visit_expert],
        jnp.sum(active, dtype=jnp.int32),
    )


def _quantize_rows(x, dtype):
    """Quantization per row per _Q_BLOCK columns straight into the
    [T, h / 128, 128] layout the expert kernel gathers rows from. The block
    maxima reduce over lanes first, in that layout (no relayout of x), then
    over each block's lane blocks. Each scale is rounded up to bf16, so it
    travels in 16 bits exactly and x / scale stays within the fp8 range.
    Returns (q, scales [T, ceil(h / (2 * _Q_BLOCK))] int32 packed by
    _pack_scales); an all-zero block gets scale 0."""
    tokens, h = x.shape
    per_block = _Q_BLOCK // _LANES
    x3 = x.reshape(tokens, h // _LANES, _LANES)
    amax = jnp.max(jnp.abs(x3), axis=2).reshape(tokens, -1, per_block).max(-1)
    scale = amax.astype(jnp.float32) / float(jnp.finfo(dtype).max)
    usable = jnp.isfinite(scale) & (scale > 0)
    bits = lax.bitcast_convert_type(jnp.where(usable, scale, 0.0), jnp.uint32)
    bits = (bits + jnp.uint32(0xFFFF)) & jnp.uint32(0xFFFF0000)  # round up to bf16
    scale = lax.bitcast_convert_type(bits, jnp.float32)
    inv = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0)
    inv = jnp.repeat(inv.astype(x.dtype), per_block, axis=1)[:, :, None]
    q = jnp.where(inv != 0, x3 * inv, jnp.zeros_like(x3)).astype(dtype)
    return q, _pack_scales(bits)


def _pack_scales(bits):
    """[T, Q] uint32 bf16-valued scale bits -> [T, ceil(Q / 2)] int32, two
    per word: block 2j in the high 16 bits, block 2j + 1 in the low."""
    if bits.shape[1] % 2:
        bits = jnp.pad(bits, ((0, 0), (0, 1)))
    return lax.bitcast_convert_type(bits[:, 0::2] | (bits[:, 1::2] >> 16), jnp.int32)


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


def _chunked_psum(tokens, combine_chunk, axis_name):
    """psum(combine_chunk(rows)) over equal token chunks of at most
    _COMBINE_CHUNK_TOKENS, concatenated. Chunk c's input is tied to chunk
    c - 1's reduced output so XLA keeps the reductions separate and each runs
    under the next chunk's combine."""
    chunks = next(
        n
        for n in range(1, tokens + 1)
        if not tokens % n and tokens // n <= _COMBINE_CHUNK_TOKENS
    )
    chunk = tokens // chunks
    outs = []
    for c in range(chunks):
        y = combine_chunk(slice(c * chunk, (c + 1) * chunk))
        if outs:
            y, _ = lax.optimization_barrier((y, outs[-1]))
        outs.append(_psum_rows(y, axis_name))
    return outs[0] if chunks == 1 else jnp.concatenate(outs, 0)


def _combine(slab, slab_row, coeff, axis_name):
    """out[t] = psum(sum_k coeff[t, k] * slab[slab_row[t, k]]), [T, K + 1]
    tables, with fused_moe/v2's combine kernel."""
    slots = slab_row.shape[1]
    h = slab.shape[1] * _LANES

    def combine_chunk(rows):
        chunk = rows.stop - rows.start
        if combine_step_tokens(chunk):
            return build_combine_kernel(
                t_local=chunk,
                topk=slots,
                hidden=h,
                lane_blocks=h // _LANES,
                wire_dtype=jnp.bfloat16,
                out_dtype=jnp.bfloat16,
            )(slab, coeff[rows], slab_row[rows])
        # Chunk sizes the Pallas combine does not tile.
        gathered = slab.reshape(-1, h)[slab_row[rows]].astype(jnp.float32)
        y = jnp.sum(gathered * coeff[rows][..., None], axis=1)
        return y.astype(jnp.bfloat16)

    return _chunked_psum(slab_row.shape[0], combine_chunk, axis_name)


# ---- Sparse combine (blocked mode) ----------------------------------------
#
# v2's combine reads K + 1 slab rows per token whatever their weight. Under
# expert parallelism a token has K / EP local routes on average (GLM-5.2 at
# EP16: 0.5 of 8), so most of those reads fetch a row again for a zero
# weight. This combine reads only the live slots: each token's slot list
# holds its shared row first, then its local routed rows, and a per-token
# count says how many are live. Same step structure as v2's (steps of
# _SC_TOKENS tokens, rows staged _SC_LEAD steps ahead, the sum in the rows'
# layout, the bf16 tile relaid out in VMEM); a step's row wait counts the
# bytes its live slots started.


def _sparse_combine_tokens(tokens):
    """Tokens per step of the sparse combine kernel, or 0 for its XLA form."""
    steps = tokens // _SC_TOKENS
    if tokens % _SC_TOKENS or steps < _SC_PARITIES or steps % _SC_PARITIES:
        return 0
    return _SC_TOKENS


@functools.cache
def _build_sparse_combine_kernel(*, tokens, slots, h):
    tb = _SC_TOKENS
    steps = tokens // tb
    lane_blocks = h // _LANES
    groups = tb // _SC_GROUP

    def kernel(
        count,
        step_rows,
        slab_row,
        coeff,
        slab_hbm,
        out_hbm,
        rows_vmem,
        tile_vmem,
        out_vmem,
        row_sems,
        out_sems,
    ):
        """Scalar prefetch: count [T] (live slots per token), step_rows
        [steps] (live slots per step), slab_row / coeff [S * T] slot-major
        (slot j of token t at j * T + t, live slots first; slot 0, the shared
        row, always live). Only live slots are touched: a scalar loop per
        token, so the cost follows the live rows, not K."""

        def issue_step(step, parity):
            """Start the live rows of step `step` into row buffer `parity`."""
            base = step * tb

            def start(i, t, j, priority):
                pltpu.make_async_copy(
                    slab_hbm.at[slab_row[j * tokens + t]],
                    rows_vmem.at[parity, j * tb + i],
                    row_sems.at[parity],
                ).start(priority=priority)

            def token(i, carry):
                t = base + i
                # Shared rows and routed rows on separate DMA queues.
                start(i, t, 0, 0)

                def slot(j, carry):
                    start(i, t, j, 1)
                    return carry

                return lax.fori_loop(1, count[t], slot, carry)

            lax.fori_loop(0, tb, token, jnp.int32(0))

        def one_step(step, parity):
            base = step * tb
            out_par = parity % 2
            # Semaphores count bytes: wait for exactly the rows started.
            n = step_rows[step]
            pltpu.make_async_copy(
                rows_vmem.at[parity, pl.ds(0, n)],
                rows_vmem.at[parity, pl.ds(0, n)],
                row_sems.at[parity],
            ).wait()

            @pl.when(step + _SC_LEAD < steps)
            def _():
                issue_step(step + _SC_LEAD, (parity + _SC_LEAD) % _SC_PARITIES)

            @pl.when(step >= 2)
            def _():
                pltpu.make_async_copy(
                    out_vmem.at[out_par], out_vmem.at[out_par], out_sems.at[out_par]
                ).wait()

            def token(i, carry):
                t = base + i

                def slot(j, acc):
                    row = rows_vmem[parity, j * tb + i].astype(jnp.float32)
                    return acc + row * coeff[j * tokens + t]

                acc = rows_vmem[parity, i].astype(jnp.float32) * coeff[t]
                acc = lax.fori_loop(1, count[t], slot, acc)
                tile_vmem[i] = acc.astype(jnp.bfloat16)
                return carry

            lax.fori_loop(0, tb, token, jnp.int32(0))
            for g in range(groups):
                out_vmem[out_par, pl.ds(g * _SC_GROUP, _SC_GROUP)] = tile_vmem[
                    pl.ds(g * _SC_GROUP, _SC_GROUP)
                ].reshape(_SC_GROUP, h)
            pltpu.make_async_copy(
                out_vmem.at[out_par], out_hbm.at[pl.ds(base, tb)], out_sems.at[out_par]
            ).start()

        def step_group(g, carry):
            for parity in range(_SC_PARITIES):
                one_step(g * _SC_PARITIES + parity, parity)
            return carry

        for p in range(_SC_LEAD):
            issue_step(jnp.int32(p), p)
        lax.fori_loop(0, steps // _SC_PARITIES, step_group, jnp.int32(0))
        for p in range(2):
            pltpu.make_async_copy(out_vmem.at[p], out_vmem.at[p], out_sems.at[p]).wait()

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[hbm],
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM(
                    (_SC_PARITIES, slots * tb, lane_blocks, _LANES), jnp.bfloat16
                ),
                pltpu.VMEM((tb, lane_blocks, _LANES), jnp.bfloat16),
                pltpu.VMEM((2, tb, h), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((_SC_PARITIES,)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, h), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=v2_host.vmem_limit(),
            disable_bounds_checks=True,
        ),
        name=f"fused_moe_prefill_combine-t_{tokens}-s_{slots}",
    )


def _sparse_combine(slab, slab_row, coeff, live, axis_name):
    """_combine over only the live slots, from [S, T] SLOT-MAJOR tables (a
    [T, S] table pads its minor dimension to a lane block; [S, T] does not):
    slot 0 the shared row (always live), `live` marking the routed slots that
    carry a local route. One stable sort along the slot axis moves each
    token's live slots to the front, so the kernel reads count[t] rows for
    token t."""
    _, slab_row, coeff = lax.sort(
        (
            jnp.logical_not(live).astype(jnp.int32),
            slab_row.astype(jnp.int32),
            coeff.astype(jnp.float32),
        ),
        dimension=0,
        num_keys=1,
        is_stable=True,
    )
    count = jnp.sum(live, axis=0).astype(jnp.int32)
    slots = slab_row.shape[0]
    h = slab.shape[1] * _LANES

    def combine_chunk(rows):
        chunk = rows.stop - rows.start
        if _sparse_combine_tokens(chunk):
            step_rows = jnp.sum(count[rows].reshape(-1, _SC_TOKENS), axis=1)
            return _build_sparse_combine_kernel(tokens=chunk, slots=slots, h=h)(
                count[rows],
                step_rows.astype(jnp.int32),
                slab_row[:, rows].reshape(-1),
                coeff[:, rows].reshape(-1),
                slab,
            )
        # Chunk sizes the kernel does not tile.
        mask = jnp.arange(slots)[:, None] < count[rows][None, :]
        gathered = slab.reshape(-1, h)[slab_row[:, rows]].astype(jnp.float32)
        weight = jnp.where(mask, coeff[:, rows], 0.0)
        y = jnp.sum(jnp.where(mask[..., None], gathered, 0.0) * weight[..., None], 0)
        return y.astype(jnp.bfloat16)

    return _chunked_psum(slab_row.shape[1], combine_chunk, axis_name)


def fused_moe_prefill(
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
    axis_name,
    expert_start=None,
    routed_scale=1.0,
    i_block=None,
):
    """Per-rank body; call inside shard_map over the mesh.

    Returns the (shared + routed) MoE output summed over `axis_name`, BF16
    [tokens, H], identical on every rank.

    Args:
      hidden: [tokens, H] BF16 hidden states, the same on every rank.
      w1: [E, H, 2 i] this rank's gate and up projections of its E experts,
        concatenated on the last axis; i is the rank's intermediate slice.
      w2: [E, i, H] the down projections.
      w1_scale: FP32 per-output-channel scales of w1, [E, 1, 1, 2 i] (any
        shape with E * 2 i elements).
      w2_scale: FP32 per-output-channel scales of w2, [E, 1, 1, H] likewise.
      shared_w1: [H, 2 i_s] the shared expert's gate and up slice.
      shared_w1_scale: FP32 scales of shared_w1, 2 i_s elements.
      shared_w2: [i_s, H] the shared expert's down slice.
      shared_w2_scale: FP32 scales of shared_w2, H elements.
      shared_gate_weight: [H, 1] the replicated shared_expert_gate; the shared
        expert's output is scaled by sigmoid(hidden @ shared_gate_weight)
        (Qwen3.5). None for an ungated shared expert (GLM-5.2, DeepSeek-V3).
      topk_weights: [tokens, k] route weights from the caller's router, used in
        their dtype. Routes of weight 0 (padded rows under
        TPU_MOE_SKIP_PADDED_TOKENS) are not computed.
      topk_ids: [tokens, k] int32 expert ids of those routes.
      axis_name: the mesh axis the per-rank partials are summed over.
      expert_start: with expert parallelism, the global id of this rank's
        first expert; the rank holds experts [expert_start, expert_start + E)
        and topk_ids are global. Routes to other ranks' experts are not
        computed here. None for TP experts.
      routed_scale: multiplies the routed experts' sum only, never the shared
        expert, as vLLM's MoERunner applies routed_scaling_factor (GLM-5.2:
        2.5). Folded into the route weights.
      i_block: force blocked mode with this block width (tests); by default
        plan_blocks picks it for experts too large to stream whole.

    The weights' dtype (FP8 in production) is also the dtype the kernel
    quantizes the hidden states and the activated intermediate to, one scale
    per row per 512 columns. The hidden size must be a multiple of 512.
    """
    tokens, h = hidden.shape
    if h % _Q_BLOCK:
        raise ValueError(f"hidden size {h} must be a multiple of {_Q_BLOCK}")
    experts, _, two_i = w1.shape
    i = two_i // 2
    i_s = shared_w2.shape[0]
    routes = tokens * topk_ids.shape[1]
    weight_dtype = w1.dtype
    tile_rows = _tile_rows(tokens)
    blocks = plan_blocks(tokens, h, i, i_s, weight_dtype, i_block)
    if blocks is None:
        call, scale_rows, token_rows = _build_expert_kernel(
            experts=experts,
            h=h,
            i=i,
            i_s=i_s,
            tokens=tokens,
            routes=routes,
            tile_rows=tile_rows,
            weight_dtype=weight_dtype,
        )
    else:
        ib, nb, chunk_rows = blocks
        call, scale_rows, token_rows = _build_blocked_expert_kernel(
            experts=experts,
            h=h,
            i=i,
            i_s=i_s,
            tokens=tokens,
            routes=routes,
            ib=ib,
            nb=nb,
            chunk_rows=chunk_rows,
            tile_rows=tile_rows,
            weight_dtype=weight_dtype,
        )

    x_q, x_scale = _quantize_rows(hidden, weight_dtype)
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
    if blocks is None:
        # The shared expert is one more visit, over every token. (The blocked
        # kernel cuts the shared rows into chunks itself.)
        visit_expert = visit_expert.at[routed_visits].set(experts)
        visit_start = visit_start.at[routed_visits].set(routes)
        visit_rows = visit_rows.at[routed_visits].set(tokens)
    # [P, slab rows]: per packed scale word, the routed rows' then the shared
    # rows' (every token), each plane in the flat lane-dense layout.
    row_scale = jnp.concatenate([sorted_scale, x_scale.T], 1)
    row_scale = jnp.pad(
        row_scale, ((0, 0), (0, scale_rows * _LANES - row_scale.shape[1]))
    )
    slab = call(
        visit_expert,
        visit_start,
        visit_rows,
        (routed_visits + 1 if blocks is None else routed_visits).reshape(1),
        x_q,
        row_scale.reshape(row_scale.shape[0], -1, _LANES),
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

    # ---- combine: top-k routed rows + the shared row per token ----
    if shared_gate_weight is None:
        gate = jnp.ones((tokens, 1), jnp.float32)
    else:
        gate = jax.nn.sigmoid(
            jnp.dot(
                hidden,
                shared_gate_weight.reshape(h, 1).astype(hidden.dtype),
                preferred_element_type=jnp.float32,
            )
        )
    shared_row = routes + jnp.arange(tokens, dtype=jnp.int32)[:, None]
    # Rows of routes that were not computed are never written; point those
    # routes at a written row so coefficient 0 never meets uninitialized memory.
    slab_row = jnp.where(route_valid, slab_row, shared_row)
    coeff = jnp.where(route_valid, topk_weights.astype(jnp.float32), 0.0)
    if routed_scale != 1.0:
        coeff = coeff * jnp.float32(routed_scale)
    if blocks is None:
        return _combine(
            slab,
            jnp.concatenate([slab_row, shared_row], 1),
            jnp.concatenate([coeff, gate], 1),
            axis_name,
        )
    return _sparse_combine(
        slab,
        jnp.concatenate([shared_row.T, slab_row.T], 0),
        jnp.concatenate([gate.T, coeff.T], 0),
        jnp.concatenate([jnp.ones((1, tokens), jnp.bool_), route_valid.T], 0),
        axis_name,
    )
