# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""FUSED_MOE_PREFILL_KERNEL_V2: the fused MoE prefill kernel with the t4-moe-prefill
stage-A levers.

The op and slab layout of fused_moe_prefill_kernel, with per-row activation
quantization: one scale per token row (and per intermediate row), as the prefill kernel
quantized before it moved to one scale per row per 512 columns. That per-row
quantization and the routing plan that carries one scale per row are kept here
(`_quantize_rows`, `_routing_plan`); the other helpers are fused_moe_prefill_kernel's.
Not bit-identical to the default build, whose activation scales are per 512 columns.
Whole experts only (no blocked mode: callers check plan_blocks). Beyond the
quantization, what changes is how the expert kernel and the combine spend their issue
slots. Build arguments (the per-row default build's value first):

  nb        routed weight ring depth (3). On v7x three slots stream 2.56-2.59 TB/s and
            four reach the roofline's 3.34 TB/s, so experts that own few rows stop
            waiting on their weights.
  down      "dot" | "xmxu": both matmuls on the explicit MXU. The down projection runs
            one 256-column N-tile at a time and each N-tile's epilogue (scales, bf16
            pair pack, strided store) follows its pop >= 32 vmatmul slots later, so the
            [rows, h] f32 accumulator never exists and does not spill. Mosaic refuses
            explicit MXU ops beside a dot in one program, so gate/up is explicit too,
            scheduled as Mosaic's dot schedules it: 256-deep K-tiles alternate MXUs,
            each accumulating its own in order, and the two pops are added. The down
            contraction is padded from i to 256 with zero weight rows (written once; the
            ring DMAs never touch them).
  issue     "loop" | "inline": the token table is resident (scalar prefetch) and tile g
            + 2's activation-row DMAs are issued straight-line inside tile g's down
            N-tiles (predicated, not branched), in slots the vector-extended work leaves
            free; the per-tile token-window DMA is gone.
  tile_min  token count from which tiles are 256 rows (8192).
  flat      combine descriptors indexed from a carried flat pointer (False).
  sum_in_kernel_max_tokens
            largest token count at which the expert kernel sums each token's rows
            itself (0); no slab in HBM and no combine kernel. Where the f32 sums of all
            h columns fit in 24 MiB (1024 tokens on GLM-5.2's TP16 slice, 47 MiB of
            VMEM in all) it is one pass: each row is scaled by its routing weight in
            the down epilogue and added into an f32 [T, h] array at its token, and the
            shared expert's tiles, which come last and cover the tokens in order, add
            their rows and write the bf16 output. Up to 4096 tokens it is two phases.
            Phase 1 runs the gate/up loop alone and keeps every tile's bf16
            intermediate rows in VMEM (10.5 MiB at 4096 tokens). Phase 2 runs once per
            2048-column slice of h: the slice of each visit's w2 streams through the
            ring, each tile's down projection is added at its tokens, weighted from
            SMEM, into an f32 [T, 2048] array (32 MiB), and the shared expert's tiles
            write the finished tokens' slice. Each weight is still read once; 58.5 MiB
            of VMEM at 4096 tokens.
  sum_overlap_adds
            in one pass: each tile's staged rows are added into the sums during the
            next tile's gate/up matmuls, straight-line in the issue slots they leave
            free, instead of in a loop after the tile (False). A routed tile stages its
            rows in the slot of its parity and reads its predecessor's from the other,
            so the two never alias; the shared expert's tiles use slot 0.
  row_tokens_resident_max_tokens
            largest token count at which loop issue keeps the token table resident
            in SMEM (scalar prefetch), as inline issue does, and reads each route's
            token from it (0). Below it the per-tile token-window DMA, its wait and
            the fence after the wait are gone; the same tokens feed the same row
            DMAs. The table takes 4 bytes per route: 132.6 KB at 4096 tokens, top-8.
            The in-kernel sum builds keep the table resident already; the op
            passes it only to the builds that write the slab.
  scatter   return only this core's rows of the summed output, the reduce-scatter
            without the all-gather after it (False). The rows a core keeps are its
            block of every combine chunk (`own_rows`); TPU_PREFILL_SEQ_PARALLEL keeps
            the residual stream split that way between the layers.

Explicit-MXU forms follow jax tests/pallas/tpu_pallas_test.py ExplicitMXUTest (jax
0.11.2 ref form). The measurements behind each setting are in the t4-moe-prefill
campaign (WHITEBOARD.v1, probe 02).
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental import rotated_collectives
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import combine_scatter
from vllm_torchtpu.kernels.experimental.fused_moe_prefill.fused_moe_prefill_kernel import (  # noqa: E501
    _GATHER_UNROLL,
    _LANES,
    _LOOKAHEAD,
    _TILE_RUNGS,
    _VISIT_SLACK,
    _WEIGHT_DMA_PRIORITY,
    _byte_transpose4,
    _cdiv,
    _psum_rows,
)
from vllm_torchtpu.kernels.fused_moe.v2 import host as v2_host
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    COMBINE_CHUNK,
    COMBINE_GROUP,
    COMBINE_LEAD,
    COMBINE_PARITIES,
    DN,
    HIDDEN_LANE_BLOCK,
    QCHUNK,
    _absmax_rows,
    _intermediate_rows,
    _quant_apply,
    _row_scale,
    _tile_row_scales,
    build_combine_kernel,
    combine_step_tokens,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


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


def _routing_plan(topk_ids, route_valid, experts, row_scale, route_coeff=None):
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
    row, rows; E + _VISIT_SLACK entries each), the routed visit count and, when
    route_coeff [T, K] is given, the routing weight of each slab row [R] (the
    first sort carries it beside the scale), else None.
    """
    tokens, topk = topk_ids.shape
    routes = tokens * topk
    route_expert = jnp.where(
        route_valid.reshape(-1), topk_ids.reshape(-1).astype(jnp.int32), experts
    )
    iota = jnp.arange(routes, dtype=jnp.int32)
    route_scale = jnp.broadcast_to(row_scale[:, None], (tokens, topk)).reshape(-1)
    carried = (route_scale,)
    if route_coeff is not None:
        carried += (route_coeff.reshape(-1),)
    if routes < 1 << 20 and experts < 1 << 11:
        # One sort on (expert << 20) | route (expert ids up to E < 2^11, route
        # ids < 2^20): the route id breaks ties within an expert, so this is
        # the stable expert order.
        key = (route_expert.astype(jnp.uint32) << 20) | iota.astype(jnp.uint32)
        key, *sorted_rows = lax.sort((key, *carried), num_keys=1, is_stable=False)
        order = (key & ((1 << 20) - 1)).astype(jnp.int32)
    else:
        _, order, *sorted_rows = lax.sort(
            (route_expert, iota, *carried), num_keys=1, is_stable=True
        )
    sorted_scale = sorted_rows[0]
    sorted_coeff = sorted_rows[1] if route_coeff is not None else None
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
        sorted_coeff,
    )


_COMBINE_CHUNK_TOKENS = 2048
# What FUSED_MOE_PREFILL_KERNEL_V2 serves (t4-moe-prefill probes 03-04, v7x, GLM-5.2
# TP16 per-rank shapes; device time per call of the whole op against the default build).
# 4096 tokens: -134 us uniform routing, -85 Zipf s 0.3, -78 s 0.6, -72 s 1.0, -65
# top-heavy; 2048 -26 / -23; 1024 -7; 512 and 256 unchanged. down="xmxu" carries 2048
# and below and 32-52 us of 4096; tile_min=4096 on 32-row rungs (tall_rung_rows) the
# rest of 4096 -- on v1's 64-row rungs the taller tile pads visit tails and lost 18 us
# under top-heavy routing; flat 0.1-3.6 us. issue="inline" stays off (+7..+18 us: its
# predicated issue takes bundles for every block of every tile); nb 3 and 4 measure
# within 3 us. Shapes the explicit-MXU body does not handle run the per-row default
# build.
STAGE_A = dict(nb=4, down="xmxu", tile_min=4096, flat=True, tall_rung_rows=32)

_NT = 256  # explicit-MXU N-tile width (MXU column size on v7x)
_MID_BLOCK = 16
_ADD_BATCH = 4
_FINAL_ROWS = 64
_SUMS_ONE_PASS_BYTES = 24 << 20
_SUMS_SLICED_BYTES = 32 << 20
_SUM_SLICE_COLS = 2048


def sum_columns(tokens, h, i, i_s):
    """Columns of the f32 per-token sums the expert kernel holds at a time when it
    sums each token's rows itself: all of h in one pass while they fit in 24 MiB
    (1024 tokens at h 6144), else 2048-column slices of up to 32 MiB (4096 tokens)
    over two phases; 0 where neither fits."""
    if tokens * h * 4 <= _SUMS_ONE_PASS_BYTES:
        return h
    if (
        h % _SUM_SLICE_COLS == 0
        and h > _SUM_SLICE_COLS
        and tokens * _SUM_SLICE_COLS * 4 <= _SUMS_SLICED_BYTES
        and i == i_s <= QCHUNK
    ):
        return _SUM_SLICE_COLS
    return 0


def _quantize_mid(mid):
    """fp8 rows and row scale of bf16 intermediate rows, as _intermediate_rows
    quantizes them when they fit one QCHUNK."""
    mid_scale, mid_sinv, usable = _row_scale(_absmax_rows(mid))
    return _quant_apply(mid, mid_sinv.astype(jnp.bfloat16), usable), mid_scale


def tile_rows_for(tokens, tile_min=8192):
    return 256 if tokens >= tile_min else 128


def _lag_for(rows, nt):
    """N-tiles between an N-tile's accumulate and its pop: >= 32 vmatmul slots (32-row
    vmatmuls) [hardware-facts §3: pops ~8 slots after the last acc stall; >= 32 later
    did not]."""
    return min(nt - 1, max(2, _cdiv(1024, rows)))


@functools.cache
def build_expert_kernel(
    *,
    experts,
    h,
    i,
    i_s,
    tokens,
    routes,
    tile_rows,
    weight_dtype,
    nb=3,
    down="dot",
    issue="loop",
    lag=None,
    rung_rows=None,
    sum_cols=0,
    overlap_adds=False,
    resident_tokens=False,
):
    assert down in ("dot", "xmxu") and issue in ("loop", "inline")
    assert not (issue == "inline" and down != "xmxu"), (
        "inline issue lives in the xmxu N-tile sequence"
    )
    sum_in_kernel = sum_cols > 0
    single_pass = sum_cols == h
    two_phase = sum_in_kernel and not single_pass
    assert not overlap_adds or single_pass
    assert not sum_in_kernel or tokens % tile_rows == 0, (
        "the shared expert writes the output in whole tiles"
    )
    inline = issue == "inline"
    resident = inline or sum_in_kernel or resident_tokens
    n_act = _LOOKAHEAD + 1
    window = tile_rows + _LANES
    scale_window = tile_rows // _LANES + 1
    # rung_rows: the height step of the rungs a tile computes on (None: tile_rows /
    # _TILE_RUNGS, v1). A 256-row tile on 32-row rungs pads a visit's tail exactly as a
    # 128-row tile on 32-row rungs does, so a taller tile only ever saves tiles.
    n_rungs = _TILE_RUNGS if rung_rows is None else tile_rows // rung_rows
    rungs = [tile_rows * (r + 1) // n_rungs for r in range(n_rungs)]
    lane_blocks = h // _LANES
    lhs_from_words = (
        jnp.dtype(weight_dtype).itemsize == 1
        and lane_blocks % 4 == 0
        and tile_rows % (4 * n_rungs * 8) == 0
        and math.gcd(lane_blocks, 32) <= 16
    )
    out_by_pairs = lane_blocks % 2 == 0 and tile_rows % (8 * n_rungs) == 0
    xmxu = down == "xmxu"
    if xmxu:
        assert out_by_pairs and h % _NT == 0 and _NT == 2 * _LANES
    nt = h // _NT
    rung_step = tile_rows // n_rungs
    slices = h // sum_cols if sum_in_kernel else 1
    if two_phase:
        assert h % sum_cols == 0 and (sum_cols // _NT) % 8 == 0
        assert i == i_s and i <= QCHUNK and rung_step % _MID_BLOCK == 0
        assert not inline, "inline issue lives in the down N-tiles phase 1 drops"
    stage_cols = sum_cols if two_phase else h
    mid_blocks = _cdiv(routes + tokens, _MID_BLOCK) + experts * (
        rung_step // _MID_BLOCK
    )
    # padded contraction of the down matmul (the explicit MXU takes K = 256); unpadded
    # under "dot"
    kp = _cdiv(i, _NT) * _NT if xmxu else i
    kp_s = _cdiv(i_s, _NT) * _NT if xmxu else i_s
    shared = experts
    slab_rows = routes + tokens + tile_rows
    scale_rows = _cdiv(slab_rows, _LANES) + 1
    token_rows = _cdiv(routes + (tokens if two_phase else 0), _LANES) * _LANES + window
    n_blk_max = tile_rows // _GATHER_UNROLL
    # inline issue: block b of the next tile is issued after N-tile blocks_at.index(b)
    blocks_at = [
        [b for b in range(n_blk_max) if b * nt // n_blk_max == n] for n in range(nt)
    ]

    def kernel(*refs):
        refs = list(refs)
        visit_expert, visit_start, visit_rows, num_visits = refs[:4]
        k0 = 4
        row_token_sm = coeff_sm = None
        if resident:
            row_token_sm = refs[4]
            k0 = 5
        if two_phase:
            coeff_sm = refs[5]
            k0 = 6
        x_hbm, row_scale_hbm = refs[k0], refs[k0 + 1]
        k0 += 2
        row_coeff_hbm = None
        if single_pass:
            row_coeff_hbm = refs[k0]
            k0 += 1
        row_token_hbm = None
        if not resident:
            row_token_hbm = refs[k0]
            k0 += 1
        (
            w1_hbm,
            w2_hbm,
            w1_scale_hbm,
            w2_scale_hbm,
            shared_w1_hbm,
            shared_w2_hbm,
            shared_w1_scale_hbm,
            shared_w2_scale_hbm,
            out_hbm,
        ) = refs[k0 : k0 + 9]
        k0 += 9
        (
            act_vmem,
            act_scale_vmem,
            out_vmem,
            w1_ring,
            w2_ring,
            w1_scale_ring,
            w2_scale_ring,
        ) = refs[k0 : k0 + 7]
        k0 += 7
        token_window = window_sems = None
        if not resident:
            token_window = refs[k0]
            k0 += 1
        act_sems = refs[k0]
        k0 += 1
        if not resident:
            window_sems = refs[k0]
            k0 += 1
        if two_phase:
            (
                out_sems,
                ring_sems,
                shared_w1_vmem,
                shared_w1_scale_vmem,
                shared_sem,
            ) = refs[k0 : k0 + 5]
            shared_w2_vmem = shared_w2_scale_vmem = None
            k0 += 5
        else:
            (
                out_sems,
                ring_sems,
                shared_w1_vmem,
                shared_w2_vmem,
                shared_w1_scale_vmem,
                shared_w2_scale_vmem,
                shared_sem,
            ) = refs[k0 : k0 + 7]
            k0 += 7
        coeff_vmem = mid_vmem = sums_vmem = final_vmem = None
        if single_pass:
            coeff_vmem, sums_vmem, final_vmem = refs[k0 : k0 + 3]
            k0 += 3
        if two_phase:
            mid_vmem, sums_vmem, final_vmem = refs[k0 : k0 + 3]
            k0 += 3
        assert k0 == len(refs), (k0, len(refs))

        n_visit = num_visits[0]
        # ACC: 128 entries of 8 x 256 f32 per MXU (1024 rows). Mosaic's ACC allocator is
        # a bump pointer that only pl.run_scoped rewinds, and it does so for the
        # run_scoped's own context only; so every compute copy allocates its
        # accumulators at the top of its own run_scoped (copies are mutually exclusive
        # branches and alias the same entries), never inside a nested pl.when.
        acc_entries_rows = 1024

        def is_shared(v):
            return visit_expert[v] == shared

        def ring_copies(expert, slot):
            w2_dst = w2_ring.at[slot] if kp == i else w2_ring.at[slot, pl.ds(0, i)]
            srcs = (
                w1_hbm.at[expert],
                w2_hbm.at[expert],
                w1_scale_hbm.at[expert],
                w2_scale_hbm.at[expert],
            )
            dsts = (
                w1_ring.at[slot],
                w2_dst,
                w1_scale_ring.at[slot],
                w2_scale_ring.at[slot],
            )
            if two_phase:
                srcs, dsts = srcs[::2], dsts[::2]
            return [
                pltpu.make_async_copy(s, d, ring_sems.at[slot])
                for s, d in zip(srcs, dsts)
            ]

        if two_phase:
            shared_copies = [
                pltpu.make_async_copy(s, d, shared_sem)
                for s, d in zip(
                    (shared_w1_hbm.at[0], shared_w1_scale_hbm.at[0]),
                    (shared_w1_vmem, shared_w1_scale_vmem),
                )
            ]
        else:
            assert shared_w2_vmem is not None
            sw2_dst = (
                shared_w2_vmem if kp_s == i_s else shared_w2_vmem.at[pl.ds(0, i_s)]
            )
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
                        sw2_dst,
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

        def next_tile(v, t):
            more = (t + 1) * tile_rows < visit_rows[v]
            return jnp.where(more, v, v + 1), jnp.where(more, t + 1, 0)

        def tile_start(v, t):
            return visit_start[v] + t * tile_rows

        def tile_live(v, t):
            return jnp.minimum(visit_rows[v] - t * tile_rows, tile_rows)

        # ---- token-number windows (loop issue without the resident table) ----
        def window_copy(start, slot):
            assert row_token_hbm is not None
            assert token_window is not None and window_sems is not None
            aligned = pl.multiple_of(start - lax.rem(start, _LANES), _LANES)
            lo = slot * window
            if not isinstance(lo, int):
                lo = pl.multiple_of(lo, _LANES)
            return pltpu.make_async_copy(
                row_token_hbm.at[pl.ds(aligned, window)],
                token_window.at[pl.ds(lo, window)],
                window_sems.at[slot],
            )

        def start_window(v, t, slot):
            if resident:
                return

            @pl.when(jnp.logical_and(v < n_visit, jnp.logical_not(is_shared(v))))
            def _():
                window_copy(tile_start(v, t), slot).start()

        def wait_window(v, t, slot):
            if resident:
                return

            @pl.when(jnp.logical_not(is_shared(v)))
            def _():
                window_copy(tile_start(v, t), slot).wait()

        def scale_copy(start, slot):
            return pltpu.make_async_copy(
                row_scale_hbm.at[pl.ds(start // _LANES, scale_window)],
                act_scale_vmem.at[slot],
                act_sems.at[slot],
            )

        def coeff_copy(start, slot):
            assert row_coeff_hbm is not None and coeff_vmem is not None
            return pltpu.make_async_copy(
                row_coeff_hbm.at[pl.ds(start // _LANES, scale_window)],
                coeff_vmem.at[slot],
                act_sems.at[slot],
            )

        def row_copy(token, slot, r):
            return pltpu.make_async_copy(
                x_hbm.at[token], act_vmem.at[slot, r], act_sems.at[slot]
            )

        def issue_tile(v, t, slot, wslot, routed_rows=True):
            """Start tile (v, t)'s rows into activation slot `slot`. routed_rows=False
            leaves the routed per-row DMAs to the caller (inline issue)."""
            start = tile_start(v, t)

            @pl.when(is_shared(v))
            def _():
                n = tile_live(v, t)
                pltpu.make_async_copy(
                    x_hbm.at[pl.ds(t * tile_rows, n)],
                    act_vmem.at[slot, pl.ds(0, n)],
                    act_sems.at[slot],
                ).start()

            if routed_rows:

                @pl.when(jnp.logical_not(is_shared(v)))
                def _():
                    def block(b, ptr):
                        r0 = b * _GATHER_UNROLL
                        for u in range(_GATHER_UNROLL):
                            if resident:
                                assert row_token_sm is not None
                                token = row_token_sm[ptr + u]
                            else:
                                assert token_window is not None
                                token = token_window[ptr + u]
                            row_copy(token, slot, r0 + u).start()
                        return ptr + _GATHER_UNROLL

                    ptr0 = (
                        start if resident else wslot * window + lax.rem(start, _LANES)
                    )
                    lax.fori_loop(
                        0, _cdiv(tile_live(v, t), _GATHER_UNROLL), block, ptr0
                    )

            scale_copy(start, slot).start()
            if single_pass:
                coeff_copy(start, slot).start()

        def wait_tile(v, t, slot):
            n = jnp.where(
                is_shared(v),
                tile_live(v, t),
                _cdiv(tile_live(v, t), _GATHER_UNROLL) * _GATHER_UNROLL,
            )

            pltpu.make_async_copy(
                act_vmem.at[slot, pl.ds(0, n)],
                act_vmem.at[slot, pl.ds(0, n)],
                act_sems.at[slot],
            ).wait()
            scale_copy(tile_start(v, t), slot).wait()
            if single_pass:
                coeff_copy(tile_start(v, t), slot).wait()

        def out_copy(start, parity, n):
            return pltpu.make_async_copy(
                out_vmem.at[parity, pl.ds(0, n)],
                out_hbm.at[pl.ds(start, n)],
                out_sems.at[parity],
            )

        def final_copy(t, parity):
            assert final_vmem is not None
            return pltpu.make_async_copy(
                final_vmem.at[parity],
                out_hbm.at[pl.ds(t * tile_rows, tile_rows)],
                out_sems.at[parity],
            )

        def out_wait(parity, n):
            if single_pass:
                final_copy(0, parity).wait()
                return
            pltpu.make_async_copy(
                out_vmem.at[parity, pl.ds(0, n)],
                out_vmem.at[parity, pl.ds(0, n)],
                out_sems.at[parity],
            ).wait()

        def add_rows(start, n, parity):
            """Adds the tile's n expert rows into their tokens' sums, weighted here when
            the routing weights are in SMEM (two phases) and in the epilogue
            otherwise."""
            assert row_token_sm is not None and sums_vmem is not None

            def add_row(r, carry):
                assert sums_vmem is not None  # pyrefly drops the narrowing on `+=`
                token = row_token_sm[start + r]
                if coeff_sm is None:
                    sums_vmem[token] += out_vmem[parity, r].astype(jnp.float32)
                else:
                    sums_vmem[token] += (
                        out_vmem[parity, r].astype(jnp.float32) * coeff_sm[start + r]
                    )
                return carry

            def add_block(b, carry):
                for u in range(_GATHER_UNROLL):
                    add_row(b * _GATHER_UNROLL + u, carry)
                return carry

            whole = n // _GATHER_UNROLL
            lax.fori_loop(0, whole, add_block, 0)
            lax.fori_loop(whole * _GATHER_UNROLL, n, add_row, 0)

        def finish_tokens(t, parity):
            """The shared expert's tile t holds tokens t * tile_rows onward, in order:
            adds its rows to those tokens' sums and writes them out as bf16."""
            assert sums_vmem is not None and final_vmem is not None

            def group(g, carry):
                first = pl.multiple_of(g * COMBINE_GROUP, COMBINE_GROUP)
                token = pl.multiple_of(t * tile_rows + first, COMBINE_GROUP)
                rows = pl.ds(first, COMBINE_GROUP)
                y = sums_vmem[pl.ds(token, COMBINE_GROUP)]
                y = y + out_vmem[0 if overlap_adds else parity, rows].astype(
                    jnp.float32
                )
                final_vmem[parity, rows] = y.astype(jnp.bfloat16).reshape(
                    COMBINE_GROUP, h
                )
                return carry

            lax.fori_loop(0, tile_rows // COMBINE_GROUP, group, 0)
            final_copy(t, parity).start()

        def zero_sums(b, carry):
            assert sums_vmem is not None
            sums_vmem[pl.ds(b * _GATHER_UNROLL, _GATHER_UNROLL)] = jnp.zeros(
                (_GATHER_UNROLL, *sums_vmem.shape[1:]), jnp.float32
            )
            return carry

        def add_staged_rows(start, batches, slot, live=None):
            """Adds rows of the previous tile in staging slot `slot` (slab rows from
            `start`) into their tokens' sums, straight-line, one batch of rows at a
            time; with `live`, rows from `live` on go to the spare row past the
            tokens. The rows of one tile belong to distinct tokens, so a batch loads
            all its sums before it stores them."""
            assert row_token_sm is not None and sums_vmem is not None
            for q in batches:
                to = [row_token_sm[start + r] for r in q]
                if live is not None:
                    to = [jnp.where(r < live, token, tokens) for r, token in zip(q, to)]
                sums = [
                    sums_vmem[token] + out_vmem[slot, r].astype(jnp.float32)
                    for token, r in zip(to, q)
                ]
                for token, row_sum in zip(to, sums):
                    sums_vmem[token] = row_sum

        def add_left_rows(start, live, first, slot):
            """Adds rows first .. live of the previous tile in staging slot `slot`,
            those the next tile's gate/up does not cover: _GATHER_UNROLL at a time,
            then one at a time."""
            assert row_token_sm is not None and sums_vmem is not None

            def add_block(b, carry):
                r0 = first + b * _GATHER_UNROLL
                rows = [r0 + u for u in range(_GATHER_UNROLL)]
                add_staged_rows(start, [rows[:_ADD_BATCH], rows[_ADD_BATCH:]], slot)
                return carry

            def add_row(r, carry):
                add_staged_rows(start, [[r]], slot)
                return carry

            whole = jnp.maximum(live - first, 0) // _GATHER_UNROLL
            lax.fori_loop(0, whole, add_block, 0)
            lax.fori_loop(first + whole * _GATHER_UNROLL, live, add_row, 0)

        def mid_blocks_of(live):
            return _cdiv(live, rung_step) * (rung_step // _MID_BLOCK)

        def scoped(f):
            if xmxu:
                pl.run_scoped(f)
            else:
                f()

        def down_accumulators(rows, n_tiles):
            """(lag, accumulators per MXU, the ACC accumulators) of one compute copy's
            down matmul; allocated inside that copy's run_scoped."""
            L = lag if lag is not None else _lag_for(rows, n_tiles)
            L = min(
                L, 2 * (acc_entries_rows // rows - 1) - 1
            )  # one tile of ACC stays with gate/up
            S = L // 2 + 1
            return (
                L,
                S,
                [
                    [
                        jax.empty_ref(
                            jax.ShapeDtypeStruct((rows, _NT), jnp.float32),
                            memory_space=pltpu.ACC(m),
                        )
                        for _ in range(S)
                    ]
                    for m in range(2)
                ],
            )

        def down_xmxu(
            rows, mid_q, mid_scale, w2, w2_scale, out32, half, accumulators, after_acc
        ):
            """out[:, n-tile] = pop(acc_n) * mid_scale * w2s[n-tile], one 256-column
            N-tile at a time on alternating MXUs; the pop of N-tile n waits `lag`
            N-tiles (>= 32 vmatmul slots)."""
            k_in = mid_q.shape[1]
            k_pad, n_tiles = w2.shape[0], w2.shape[1] // _NT
            lhs = (
                mid_q
                if k_in == k_pad
                else jnp.concatenate(
                    [mid_q, jnp.zeros((rows, k_pad - k_in), mid_q.dtype)], axis=1
                )
            )
            L, S, accs = accumulators

            def acc_of(n):
                return accs[n % 2][(n // 2) % S]

            def push(n):
                pltpu.matmul_push_rhs(
                    w2[:, n * _NT : (n + 1) * _NT],
                    staging_register=0,
                    mxu_index=n % 2,
                )

            def epi(n):
                r = pltpu.matmul_pop(acc_of(n))
                o = r * mid_scale * w2_scale[:, n * _NT : (n + 1) * _NT]
                out32[pl.ds(n, rows, stride=half)] = pltpu.pack_elementwise(
                    [o[:, :_LANES], o[:, _LANES:]], packed_dtype=jnp.bfloat16
                )

            push(0)
            for n in range(n_tiles):
                if n + 1 < n_tiles:
                    push(n + 1)
                pltpu.matmul_acc_lhs(acc_of(n), lhs=lhs, load_staged_rhs=0)
                if n >= L:
                    epi(n - L)
                after_acc(n)
            for n in range(n_tiles - L, n_tiles):
                epi(n)

        # ---- phase 2 of the two-phase sum: the down projection per column slice ----
        def slice_copies(w2_rows, w2_scale_row, s, slot):
            cols = pl.ds(pl.multiple_of(s * sum_cols, sum_cols), sum_cols)
            w2_dst = w2_ring.at[slot] if kp == i else w2_ring.at[slot, pl.ds(0, i)]
            return [
                pltpu.make_async_copy(w2_rows.at[:, cols], w2_dst, ring_sems.at[slot]),
                pltpu.make_async_copy(
                    w2_scale_row.at[:, cols], w2_scale_ring.at[slot], ring_sems.at[slot]
                ),
            ]

        def start_slice_weights(u):
            """Starts the w2 slice of visit u: slice u // visits, visit u % visits."""
            s, v, slot = u // n_visit, lax.rem(u, n_visit), lax.rem(u, nb)

            @pl.when(is_shared(v))
            def _():
                for c in slice_copies(
                    shared_w2_hbm.at[0], shared_w2_scale_hbm.at[0], s, slot
                ):
                    c.start(priority=_WEIGHT_DMA_PRIORITY)

            @pl.when(jnp.logical_not(is_shared(v)))
            def _():
                expert = visit_expert[v]
                for c in slice_copies(
                    w2_hbm.at[expert], w2_scale_hbm.at[expert], s, slot
                ):
                    c.start(priority=_WEIGHT_DMA_PRIORITY)

        def wait_slice_weights(u):
            for c in slice_copies(
                w2_hbm.at[0], w2_scale_hbm.at[0], jnp.int32(0), lax.rem(u, nb)
            ):
                c.wait()

        def slice_compute(rows, block, slot):
            """One tile's down projection onto the current column slice, from the
            intermediate rows phase 1 left at `block`, into out_vmem[0]."""
            assert mid_vmem is not None
            accumulators = down_accumulators(rows, sum_cols // _NT) if xmxu else None
            mid = mid_vmem[pl.ds(block, rows // _MID_BLOCK)].reshape(rows, i)
            mid_q, mid_scale = _quantize_mid(mid)
            w2, w2_scale = w2_ring.at[slot], w2_scale_ring.at[slot]
            if xmxu:
                half = sum_cols // _NT
                out32 = (
                    out_vmem.at[0].bitcast(jnp.uint32).reshape(tile_rows * half, _LANES)
                )
                down_xmxu(
                    rows,
                    mid_q,
                    mid_scale,
                    w2,
                    w2_scale,
                    out32,
                    half,
                    accumulators,
                    lambda n: None,
                )
                return
            acc2 = lax.dot_general(
                mid_q, w2[...], DN, preferred_element_type=jnp.float32
            )
            out = acc2 * mid_scale * w2_scale[...]
            out = out.reshape(rows, sum_cols // _LANES, _LANES)
            out_vmem[0, pl.ds(0, rows)] = out.astype(jnp.bfloat16)

        def write_slice(t, s):
            """Writes tokens t * tile_rows onward, whose sums the shared expert's tile
            t has just completed, to column slice s of the output, _FINAL_ROWS rows per
            DMA."""
            assert sums_vmem is not None and final_vmem is not None
            cols = pl.ds(pl.multiple_of(s * sum_cols, sum_cols), sum_cols)
            chunks = tile_rows // _FINAL_ROWS

            def copy(c):
                first = pl.multiple_of(t * tile_rows + c * _FINAL_ROWS, _FINAL_ROWS)
                return pltpu.make_async_copy(
                    final_vmem.at[c % 2],
                    out_hbm.at[pl.ds(first, _FINAL_ROWS), cols],
                    out_sems.at[c % 2],
                )

            for c in range(chunks):
                if c >= 2:
                    copy(c - 2).wait()

                def group(g, carry, c=c):
                    first = pl.multiple_of(g * COMBINE_GROUP, COMBINE_GROUP)
                    token = pl.multiple_of(
                        t * tile_rows + c * _FINAL_ROWS + first, COMBINE_GROUP
                    )
                    y = sums_vmem[pl.ds(token, COMBINE_GROUP)].astype(jnp.bfloat16)
                    final_vmem[c % 2, pl.ds(first, COMBINE_GROUP)] = y.reshape(
                        COMBINE_GROUP, sum_cols
                    )
                    return carry

                lax.fori_loop(0, _FINAL_ROWS // COMBINE_GROUP, group, 0)
                copy(c).start()
            for c in range(max(0, chunks - 2), chunks):
                copy(c).wait()

        def run_slices():
            """Phase 2: per column slice, every visit's down projection from the stored
            intermediate rows, summed per token; the shared expert's tiles, last in
            each slice, write the finished tokens."""
            for k in range(nb - 1):

                @pl.when(k < slices * n_visit)
                def _(k=k):
                    start_slice_weights(jnp.int32(k))

            def slice_body(carry):
                s, v, t, block = carry
                u = s * n_visit + v

                @pl.when(t == 0)
                def _():
                    wait_slice_weights(u)

                    @pl.when(u + nb - 1 < slices * n_visit)
                    def _():
                        start_slice_weights(u + nb - 1)

                @pl.when(jnp.logical_and(v == 0, t == 0))
                def _():
                    lax.fori_loop(0, tokens // _GATHER_UNROLL, zero_sums, 0)

                live = tile_live(v, t)
                for r, rows in enumerate(rungs):
                    below = rungs[r - 1] if r else 0

                    @pl.when(jnp.logical_and(live > below, live <= rows))
                    def _(rows=rows):
                        scoped(lambda: slice_compute(rows, block, lax.rem(u, nb)))

                add_rows(tile_start(v, t), live, 0)

                @pl.when(is_shared(v))
                def _():
                    write_slice(t, s)

                vn, tn = next_tile(v, t)
                wrapped = vn == n_visit
                return (
                    jnp.where(wrapped, s + 1, s),
                    jnp.where(wrapped, 0, vn),
                    tn,
                    jnp.where(wrapped, 0, block + mid_blocks_of(live)),
                )

            z = jnp.int32(0)
            lax.while_loop(lambda carry: carry[0] < slices, slice_body, (z, z, z, z))

        # ---- prologue ----
        # zero the padded contraction rows of the down weights once (never DMA'd over)
        if kp != i:
            w2_ring[:, pl.ds(i, kp - i), :] = jnp.zeros(
                (nb, kp - i, stage_cols), w2_ring.dtype
            )
        if kp_s != i_s and not two_phase:
            assert shared_w2_vmem is not None
            shared_w2_vmem[pl.ds(i_s, kp_s - i_s), :] = jnp.zeros(
                (kp_s - i_s, h), shared_w2_vmem.dtype
            )
        for c in shared_copies:
            c.start(priority=_WEIGHT_DMA_PRIORITY)
        for k in range(nb - 1):

            @pl.when(k < n_visit)
            def _(k=k):
                start_weights(jnp.int32(k))

        z = jnp.int32(0)
        start_window(z, z, 0)
        vi, ti = z, z
        for k in range(_LOOKAHEAD):

            @pl.when(vi < n_visit)
            def _(k=k, vi=vi, ti=ti):
                wait_window(vi, ti, k % 2)
                issue_tile(vi, ti, k % n_act, k % 2)
                vw, tw = next_tile(vi, ti)
                start_window(vw, tw, (k + 1) % 2)

            vi, ti = next_tile(vi, ti)

        if single_pass:
            lax.fori_loop(0, tokens // _GATHER_UNROLL, zero_sums, 0)

        def tile_body(carry):
            v, t, g, n_out0, n_out1, vi, ti, *extra = carry
            p = lax.rem(g, 2)
            aslot = lax.rem(g, n_act)

            @pl.when(t == 0)
            def _():
                wait_weights(v)

                @pl.when(v + nb - 1 < n_visit)
                def _():
                    start_weights(v + nb - 1)

            wait_tile(v, t, aslot)

            vn, tn = next_tile(v, t)
            gi = g + _LOOKAHEAD
            islot = lax.rem(gi, n_act)

            @pl.when(vi < n_visit)
            def _():
                wait_window(vi, ti, lax.rem(gi, 2))
                issue_tile(vi, ti, islot, lax.rem(gi, 2), routed_rows=not inline)
                vw, tw = next_tile(vi, ti)
                start_window(vw, tw, lax.rem(gi + 1, 2))

            if (
                inline
            ):  # the routed rows of tile g + 2, issued inside this tile's down N-tiles
                n_iss = jnp.where(
                    jnp.logical_and(vi < n_visit, jnp.logical_not(is_shared(vi))),
                    _cdiv(tile_live(vi, ti), _GATHER_UNROLL),
                    0,
                )
                i_base = tile_start(vi, ti)

            def issue_at(n):
                if not inline:
                    return
                assert row_token_sm is not None
                for b in blocks_at[n]:

                    @pl.when(b < n_iss)
                    def _(b=b):
                        for u in range(_GATHER_UNROLL):
                            r = b * _GATHER_UNROLL + u
                            row_copy(row_token_sm[i_base + r], islot, r).start()

            vi_n, ti_n = next_tile(vi, ti)
            vi_n = jnp.where(vi < n_visit, vi_n, vi)
            ti_n = jnp.where(vi < n_visit, ti_n, ti)

            slot = lax.rem(v, nb)
            start = tile_start(v, t)
            live = tile_live(v, t)
            if not two_phase:
                n_prev = jnp.where(p == 0, n_out0, n_out1)

                @pl.when(n_prev > 0)
                def _():
                    out_wait(p, n_prev)

            def build_lhs(rows, as_blocks=False):
                """The gate/up LHS [rows, h] (or, as_blocks, its 128-column blocks in
                order)."""
                if lhs_from_words:
                    wpr = lane_blocks // 4
                    a32 = (
                        act_vmem.at[aslot]
                        .bitcast(jnp.uint32)
                        .reshape(tile_rows * wpr, _LANES)
                    )
                    blocks = []
                    for k in range(wpr):
                        quads = [
                            a32[pl.ds(t_ * wpr + k, rows // 4, stride=4 * wpr)]
                            for t_ in range(4)
                        ]
                        blocks += [
                            pltpu.bitcast(b, weight_dtype)
                            for b in _byte_transpose4(quads)
                        ]
                    return blocks if as_blocks else jnp.concatenate(blocks, axis=1)
                act = act_vmem[aslot, pl.ds(0, rows)].astype(jnp.bfloat16)
                act = act.reshape(rows, h).astype(weight_dtype)
                if as_blocks:
                    return [
                        act[:, c * _LANES : (c + 1) * _LANES]
                        for c in range(lane_blocks)
                    ]
                return act

            def gateup_xmxu(rows, blocks, w1, after_acc):
                """acc1 = act @ w1 on the explicit MXU, as Mosaic's dot schedules it
                [static: V0 final bundles]: 256-deep K-tiles alternate between the two
                MXUs, each accumulating its own K-tiles in order, and the two pops are
                added. RHS of K-tile k + 2 is staged right after K-tile k's accumulate
                (one staging register per MXU)."""
                n_out = w1.shape[1]
                assert n_out == _NT, "gate|up is one 256-column N-tile here"
                kt = h // _NT
                per = _NT // _LANES
                accs = [
                    jax.empty_ref(
                        jax.ShapeDtypeStruct((rows, n_out), jnp.float32),
                        memory_space=pltpu.ACC(m),
                    )
                    for m in range(2)
                ]

                def push(k):
                    pltpu.matmul_push_rhs(
                        w1[k * _NT : (k + 1) * _NT, :],
                        staging_register=0,
                        mxu_index=k % 2,
                    )

                push(0)
                push(1)
                for k in range(kt):
                    lhs = jnp.concatenate(blocks[k * per : (k + 1) * per], axis=1)
                    pltpu.matmul_acc_lhs(accs[k % 2], lhs=lhs, load_staged_rhs=0)
                    after_acc(k)
                    if k + 2 < kt:
                        push(k + 2)
                return pltpu.matmul_pop(accs[0]) + pltpu.matmul_pop(accs[1])

            def staged_batches(rows):
                return [
                    range(q, min(q + _ADD_BATCH, rows))
                    for q in range(0, rows, _ADD_BATCH)
                ]

            def gateup_adds(rows, slot):
                """The K-tile hook that adds the first `rows` rows of the previous tile
                in staging slot `slot`, spread evenly over the gate/up K-tiles."""
                kt = h // _NT
                batches = staged_batches(rows)

                def at(k):
                    n = len(batches)
                    add_staged_rows(
                        extra[0],
                        [q for j, q in enumerate(batches) if j * kt // n == k],
                        slot,
                        extra[1],
                    )

                return at

            down_accs = {}

            def compute(
                rows, w1, w2, w1_scale, w2_scale, inter, stage=None, added_rows=0
            ):
                """The tile's FFN into staging slot `stage` (None: the tile's parity,
                chosen at run time). `added_rows` rows of the previous tile, staged in
                the other slot, are added into their tokens' sums beside the gate/up
                matmuls."""
                if xmxu and not two_phase:
                    # allocated here (inside this copy's run_scoped), shared by both
                    # parity branches
                    down_accs[rows] = down_accumulators(rows, nt)
                # the other slot; rows are only ever added with a static stage
                added_slot = None if stage is None else 1 - stage
                act_scale = _tile_row_scales(act_scale_vmem[aslot], start, rows)
                if xmxu:  # Mosaic refuses explicit MXU ops beside a dot in one program
                    acc1 = gateup_xmxu(
                        rows,
                        build_lhs(rows, as_blocks=True),
                        w1,
                        gateup_adds(added_rows, added_slot)
                        if added_rows
                        else lambda k: None,
                    )
                else:
                    acc1 = lax.dot_general(
                        # pyrefly: ignore[bad-argument-type]
                        build_lhs(rows),
                        w1[...],
                        DN,
                        preferred_element_type=jnp.float32,
                    )
                    if added_rows:
                        add_staged_rows(
                            extra[0], staged_batches(added_rows), added_slot, extra[1]
                        )
                if two_phase:
                    assert mid_vmem is not None
                    mid, _ = _intermediate_rows(
                        acc1,
                        act_scale,
                        inter,
                        w1s=w1_scale[...],
                        act_fn="silu",
                        w1b=None,
                        requantize=False,
                    )
                    mid_vmem[pl.ds(extra[0], rows // _MID_BLOCK)] = mid.reshape(
                        rows // _MID_BLOCK, _MID_BLOCK, inter
                    )
                    return
                mid_q, mid_scale = _intermediate_rows(
                    acc1, act_scale, inter, w1s=w1_scale[...], act_fn="silu", w1b=None
                )
                assert mid_scale is not None
                if single_pass:
                    assert coeff_vmem is not None
                    mid_scale = mid_scale * _tile_row_scales(
                        coeff_vmem[aslot], start, rows
                    )
                half = lane_blocks // 2

                def down(slot):
                    out32 = (
                        out_vmem.at[slot]
                        .bitcast(jnp.uint32)
                        .reshape(tile_rows * half, _LANES)
                        if out_by_pairs
                        else None
                    )
                    if xmxu:
                        down_xmxu(
                            rows,
                            mid_q,
                            mid_scale,
                            w2,
                            w2_scale,
                            out32,
                            half,
                            down_accs[rows],
                            issue_at,
                        )
                        return
                    acc2 = lax.dot_general(
                        mid_q, w2[...], DN, preferred_element_type=jnp.float32
                    )
                    out = acc2 * mid_scale * w2_scale[...]
                    if out_by_pairs:
                        assert out32 is not None
                        for q in range(half):
                            pair = [
                                out[:, (2 * q + j) * _LANES : (2 * q + j + 1) * _LANES]
                                for j in range(2)
                            ]
                            out32[pl.ds(q, rows, stride=half)] = pltpu.pack_elementwise(
                                pair, packed_dtype=jnp.bfloat16
                            )
                    else:
                        out = out.reshape(rows, lane_blocks, _LANES)
                        out_vmem[slot, pl.ds(0, rows)] = out.astype(jnp.bfloat16)

                if stage is not None:
                    down(stage)
                    return
                for par in range(2):
                    pl.when(p == par)(functools.partial(down, par))

            def compute_visit(rows):
                @pl.when(is_shared(v))
                def _():
                    scoped(
                        lambda: compute(
                            rows,
                            shared_w1_vmem,
                            shared_w2_vmem,
                            shared_w1_scale_vmem,
                            shared_w2_scale_vmem,
                            i_s,
                            0 if overlap_adds else None,
                        )
                    )

                def routed(stage=None):
                    scoped(
                        lambda: compute(
                            rows,
                            w1_ring.at[slot],
                            w2_ring.at[slot],
                            w1_scale_ring.at[slot],
                            w2_scale_ring.at[slot],
                            i,
                            stage,
                            rows if overlap_adds else 0,
                        )
                    )

                @pl.when(jnp.logical_not(is_shared(v)))
                def _():
                    if not overlap_adds:
                        routed()
                        return
                    for par in range(2):
                        pl.when(p == par)(functools.partial(routed, par))

            if overlap_adds:
                covered = jnp.where(is_shared(v), 0, _cdiv(live, rung_step) * rung_step)
                add_left_rows(extra[0], extra[1], covered, 1 - p)

            for r, rows in enumerate(rungs):
                below = rungs[r - 1] if r else 0

                @pl.when(jnp.logical_and(live > below, live <= rows))
                def _(rows=rows):
                    compute_visit(rows)

            if two_phase:
                next_block = extra[0] + mid_blocks_of(live)
                return vn, tn, g + 1, n_out0, n_out1, vi_n, ti_n, next_block
            if single_pass:

                @pl.when(is_shared(v))
                def _():
                    finish_tokens(t, p)

                if not overlap_adds:

                    @pl.when(jnp.logical_not(is_shared(v)))
                    def _():
                        add_rows(start, live, p)

                emitted = jnp.where(is_shared(v), live, 0)
            else:
                out_copy(start, p, live).start()
                emitted = live
            n_out0 = jnp.where(p == 0, emitted, n_out0)
            n_out1 = jnp.where(p == 1, emitted, n_out1)
            if overlap_adds:
                staged = (
                    jnp.where(is_shared(v), 0, start),
                    jnp.where(is_shared(v), 0, live),
                )
                return vn, tn, g + 1, n_out0, n_out1, vi_n, ti_n, *staged
            return vn, tn, g + 1, n_out0, n_out1, vi_n, ti_n

        _, _, _, n_out0, n_out1, *_ = lax.while_loop(
            lambda carry: carry[0] < n_visit,
            tile_body,
            (z, z, z, z, z, vi, ti)
            + ((z,) if two_phase else ())
            + ((z, z) if overlap_adds else ()),
        )

        if two_phase:
            run_slices()
            return

        for par, n in ((0, n_out0), (1, n_out1)):

            @pl.when(n > 0)
            def _(par=par, n=n):
                out_wait(par, n)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    n_in = 11 - (1 if resident else 0) + (1 if single_pass else 0)
    scratch = [
        pltpu.VMEM((n_act, tile_rows, lane_blocks, _LANES), weight_dtype),
        pltpu.VMEM((n_act, scale_window, _LANES), jnp.float32),
        pltpu.VMEM(
            (1 if two_phase else 2, tile_rows, stage_cols // _LANES, _LANES),
            jnp.bfloat16,
        ),
        pltpu.VMEM((nb, h, 2 * i), weight_dtype),
        pltpu.VMEM((nb, kp, stage_cols), weight_dtype),
        pltpu.VMEM((nb, 1, 2 * i), jnp.float32),
        pltpu.VMEM((nb, 1, stage_cols), jnp.float32),
    ]
    if not resident:
        scratch.append(pltpu.SMEM((2 * window,), jnp.int32))
    scratch.append(pltpu.SemaphoreType.DMA((n_act,)))
    if not resident:
        scratch.append(pltpu.SemaphoreType.DMA((2,)))
    if two_phase:
        scratch += [
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((nb,)),
            pltpu.VMEM((h, 2 * i_s), weight_dtype),
            pltpu.VMEM((1, 2 * i_s), jnp.float32),
            pltpu.SemaphoreType.DMA(()),
        ]
    else:
        scratch += [
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((nb,)),
            pltpu.VMEM((h, 2 * i_s), weight_dtype),
            pltpu.VMEM((kp_s, h), weight_dtype),
            pltpu.VMEM((1, 2 * i_s), jnp.float32),
            pltpu.VMEM((1, h), jnp.float32),
            pltpu.SemaphoreType.DMA(()),
        ]
    tag = f"nb{nb}-{down}-{issue}" + ("" if rung_rows is None else f"-r{rung_rows}")
    out_shape = (slab_rows, lane_blocks, _LANES)
    if single_pass:
        scratch += [
            pltpu.VMEM((n_act, scale_window, _LANES), jnp.float32),
            pltpu.VMEM(
                (tokens + (1 if overlap_adds else 0), lane_blocks, _LANES),
                jnp.float32,
            ),
            pltpu.VMEM((2, tile_rows, h), jnp.bfloat16),
        ]
        out_shape = (tokens, h)
        tag += "-sumk"
    if overlap_adds:
        tag += "-pipe"
    if two_phase:
        scratch += [
            pltpu.VMEM((mid_blocks, _MID_BLOCK, i), jnp.bfloat16),
            pltpu.VMEM((tokens, sum_cols // _LANES, _LANES), jnp.float32),
            pltpu.VMEM((2, _FINAL_ROWS, sum_cols), jnp.bfloat16),
        ]
        out_shape = (tokens, h)
        tag += f"-sumk{slices}"
    if resident_tokens and not inline and not sum_in_kernel:
        tag += "-rt"
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=(5 if resident else 4) + (1 if two_phase else 0),
            in_specs=[hbm] * n_in,
            out_specs=hbm,
            scratch_shapes=tuple(scratch),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=v2_host.vmem_limit(), disable_bounds_checks=True
        ),
        name=f"fused_moe_prefill-t_{tokens}-e_{experts}-i_{i}-is_{i_s}-m_{tile_rows}-{tag}",
    )
    return call, scale_rows, token_rows


# ------------------------------------------------------------------------------------ #
# Combine (A5): fused_moe/v2's destination combine, with the descriptor index walked by
# a carried flat pointer when flat=True. flat=False is build_combine_kernel's body
# operation for operation.
# ------------------------------------------------------------------------------------ #
@functools.cache
def build_combine(*, t_local, topk, hidden, lane_blocks, wire_dtype, out_dtype, flat):
    if not flat:
        return build_combine_kernel(
            t_local=t_local,
            topk=topk,
            hidden=hidden,
            lane_blocks=lane_blocks,
            wire_dtype=wire_dtype,
            out_dtype=out_dtype,
        )
    tb = combine_step_tokens(t_local)
    steps = t_local // tb
    gt = COMBINE_GROUP
    groups = tb // gt
    lanes = HIDDEN_LANE_BLOCK
    CH = COMBINE_CHUNK

    def kernel(
        pos_sm, coef_sm, arr_hbm, out_hbm, rows_vm, tile_vm, out_vm, row_sems, out_sems
    ):
        def start_row(pidx, parity, i, k):
            pltpu.make_async_copy(
                arr_hbm.at[pos_sm[pidx]],
                rows_vm.at[parity, k * tb + i],
                row_sems.at[parity],
            ).start(priority=k % 2)

        def one_step(step, parity):
            nxt = jnp.minimum(step + COMBINE_LEAD, steps - 1) * tb
            base = step * tb
            npar = (parity + COMBINE_LEAD) % COMBINE_PARITIES
            pltpu.make_async_copy(
                rows_vm.at[parity], rows_vm.at[parity], row_sems.at[parity]
            ).wait()

            @pl.when(step >= COMBINE_PARITIES)
            def _():
                pltpu.make_async_copy(
                    out_vm.at[parity], out_vm.at[parity], out_sems.at[parity]
                ).wait()

            def one_chunk(chunk, carry):
                # flat table index of the chunk's first token (next / this step);
                # its row
                pn, pb, r0 = carry
                for u in range(CH):
                    for k in range(topk):
                        start_row(pn + (u * topk + k), npar, r0 + u, k)
                    acc = None
                    for k in range(topk):
                        term = (
                            rows_vm[parity, k * tb + r0 + u].astype(jnp.float32)
                            * coef_sm[pb + (u * topk + k)]
                        )
                        acc = term if acc is None else acc + term
                    assert acc is not None
                    tile_vm[r0 + u] = acc.astype(out_dtype)
                return pn + CH * topk, pb + CH * topk, r0 + CH

            lax.fori_loop(
                0, tb // CH, one_chunk, (nxt * topk, base * topk, jnp.int32(0))
            )
            for g in range(groups):
                out_vm[parity, pl.ds(g * gt, gt)] = tile_vm[pl.ds(g * gt, gt)].reshape(
                    gt, hidden
                )
            pltpu.make_async_copy(
                out_vm.at[parity], out_hbm.at[pl.ds(step * tb, tb)], out_sems.at[parity]
            ).start()

        def step_pair(pair, carry):
            for parity in range(COMBINE_PARITIES):
                one_step(pair * COMBINE_PARITIES + parity, parity)
            return carry

        for p in range(COMBINE_LEAD):
            for i in range(tb):
                for k in range(topk):
                    start_row((p * tb + i) * topk + k, p, i, k)
        lax.fori_loop(0, steps // COMBINE_PARITIES, step_pair, jnp.int32(0))
        for p in range(COMBINE_LEAD):
            pltpu.make_async_copy(rows_vm.at[p], rows_vm.at[p], row_sems.at[p]).wait()
        for p in range(COMBINE_PARITIES):
            pltpu.make_async_copy(out_vm.at[p], out_vm.at[p], out_sems.at[p]).wait()

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[hbm],
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM(
                    (COMBINE_PARITIES, topk * tb, lane_blocks, lanes), wire_dtype
                ),
                pltpu.VMEM((tb, lane_blocks, lanes), out_dtype),
                pltpu.VMEM((COMBINE_PARITIES, tb, hidden), out_dtype),
                pltpu.SemaphoreType.DMA((COMBINE_PARITIES,)),
                pltpu.SemaphoreType.DMA((COMBINE_PARITIES,)),
            ),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((t_local, hidden), out_dtype),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=v2_host.vmem_limit(), disable_bounds_checks=True
        ),
        name=f"moe_v2_combine_t{t_local}_k{topk}_b{tb}_flat",
    )

    def fn(arrivals, coef, pos):
        assert pos.shape == (t_local, topk) and coef.shape == (t_local, topk)
        return call(pos.astype(jnp.int32).reshape(-1), coef.reshape(-1), arrivals)

    return fn


def token_chunks(tokens):
    """How many equal chunks of at most _COMBINE_CHUNK_TOKENS rows the combine
    sums a step of `tokens` tokens in, each with its own reduction."""
    return next(
        n
        for n in range(1, tokens + 1)
        if not tokens % n and tokens // n <= _COMBINE_CHUNK_TOKENS
    )


def _rank_blocks(x, ranks):
    """x [tokens, ...] as [chunks, ranks, rows, ...]: block (c, r) is what the
    reduce-scatter of combine chunk c leaves on core r."""
    tokens = x.shape[0]
    chunks = token_chunks(tokens)
    if tokens % (chunks * ranks):
        raise ValueError(
            f"{tokens} tokens in {chunks} chunks do not split over {ranks} cores"
        )
    return x.reshape(chunks, ranks, tokens // (chunks * ranks), *x.shape[1:])


def own_rows(x, index, ranks):
    """Core `index`'s rows of the step's rows x [tokens, ...]: its block of
    every combine chunk, in token order."""
    mine = lax.dynamic_index_in_dim(_rank_blocks(x, ranks), index, 1, keepdims=False)
    return mine.reshape(-1, *x.shape[1:])


def scatter_rows(x, axis_name):
    """The sum over `axis_name` of x [tokens, ...], this core's rows only
    (`own_rows`): one reduce-scatter per combine chunk."""
    blocks = _rank_blocks(x, lax.axis_size(axis_name))
    chunk = blocks.shape[1] * blocks.shape[2]
    outs = [
        lax.psum_scatter(
            x[c * chunk : (c + 1) * chunk], axis_name, scatter_dimension=0, tiled=True
        )
        for c in range(blocks.shape[0])
    ]
    return outs[0] if len(outs) == 1 else jnp.concatenate(outs, 0)


def gather_rows(x, axis_name):
    """Every core's rows (`own_rows`) put back in token order: one all-gather
    per combine chunk."""
    ranks = lax.axis_size(axis_name)
    chunks = token_chunks(x.shape[0] * ranks)
    rows = x.shape[0] // chunks
    outs = [
        lax.all_gather(x[c * rows : (c + 1) * rows], axis_name, axis=0, tiled=True)
        for c in range(chunks)
    ]
    return outs[0] if chunks == 1 else jnp.concatenate(outs, 0)


def in_token_order(x, ranks):
    """Rows x [tokens, ...] gathered in core order (core 0's `own_rows`, then
    core 1's, ...) put back in token order, by a reshape and an axis swap (no
    gather; the identity when the step has one combine chunk)."""
    tokens = x.shape[0]
    chunks = token_chunks(tokens)
    blocks = x.reshape(ranks, chunks, tokens // (chunks * ranks), *x.shape[1:])
    return jnp.swapaxes(blocks, 0, 1).reshape(x.shape)


def _as_words(x):
    """x [n, k] of a 32-bit type, or of bf16 (widened to f32 exactly), as int32
    [n, k] with one value per word."""
    if x.dtype == jnp.bfloat16:
        x = x.astype(jnp.float32)
    return lax.bitcast_convert_type(x, jnp.int32)


def _from_words(w, dtype):
    """The inverse of _as_words: int32 words as `dtype` values."""
    if jnp.dtype(dtype) == jnp.bfloat16:
        return lax.bitcast_convert_type(w, jnp.float32).astype(jnp.bfloat16)
    return lax.bitcast_convert_type(w, dtype)


def gather_routes(weights, ids, scale, axis_name):
    """Every core's routing weights [n, k], expert ids [n, k] and row scales [n]
    in core order, from one all-gather of int32 rows that hold one value per
    word (2 k + 1 words)."""
    k = ids.shape[1]
    packed = jnp.concatenate(
        [ids.astype(jnp.int32), _as_words(weights), _as_words(scale.reshape(-1, 1))], 1
    )
    packed = lax.all_gather(packed, axis_name, axis=0, tiled=True)
    return (
        _from_words(packed[:, k : 2 * k], weights.dtype),
        packed[:, :k],
        _from_words(packed[:, 2 * k], scale.dtype),
    )


def reduce_scatter_rows(y, axis_name, rank=None, rs_table=None):
    """lax.psum_scatter(y, tiled=True) over `axis_name`: this core's block of the
    sum of y [rows, h] bf16. With `rs_table` (rotated_collectives.peer_table
    "reduce_scatter") and `rank` (this core's mesh index, int32 [1]) it runs as
    rotated_collectives.reduce_scatter on the TensorCore when that takes y's
    shape (TPU_MOE_PREFILL_ROTATED_COLLECTIVES); the same bf16 terms, added in
    another order."""
    if (
        rs_table is not None
        and y.dtype == jnp.bfloat16
        and rotated_collectives.reduce_scatter_supports(*y.shape)
    ):
        return rotated_collectives.reduce_scatter(
            y, rank, jnp.asarray(rs_table), axis=axis_name
        )
    return lax.psum_scatter(y, axis_name, scatter_dimension=0, tiled=True)


def combine(
    slab,
    slab_row,
    coeff,
    axis_name,
    flat=False,
    scatter=False,
    rank=None,
    rs_table=None,
    cs_table=None,
):
    """Each token's sum of its slab rows times their coefficients, over the
    cores: all-reduced, or with `scatter` this core's rows only, one
    reduce-scatter per combine chunk. With `rs_table` and `rank` the last
    chunk's reduce-scatter is reduce_scatter_rows' rotated one; the earlier
    chunks' stay XLA's, which run beside the next chunk's combine. With
    `cs_table` (combine_scatter.scatter_table) every chunk the fused kernel
    takes runs its combine and reduce-scatter as combine_scatter."""
    tokens, slots = slab_row.shape
    h = slab.shape[1] * _LANES
    chunks = token_chunks(tokens)
    chunk = tokens // chunks
    fused = scatter and cs_table is not None and combine_scatter.supports(chunk, h)
    outs = []
    for c in range(chunks):
        rows = slice(c * chunk, (c + 1) * chunk)
        if fused:
            y = combine_scatter.combine_scatter(
                slab,
                coeff[rows],
                slab_row[rows],
                rank,
                jnp.asarray(cs_table),
                axis=axis_name,
            )
            if outs:
                y, _ = lax.optimization_barrier((y, outs[-1]))
            outs.append(y)
            continue
        if combine_step_tokens(chunk):
            y = build_combine(
                t_local=chunk,
                topk=slots,
                hidden=h,
                lane_blocks=h // _LANES,
                wire_dtype=jnp.bfloat16,
                out_dtype=jnp.bfloat16,
                flat=flat,
            )(slab, coeff[rows], slab_row[rows])
        else:
            gathered = slab.reshape(-1, h)[slab_row[rows]].astype(jnp.float32)
            y = jnp.sum(gathered * coeff[rows][..., None], axis=1).astype(jnp.bfloat16)
        if outs:
            y, _ = lax.optimization_barrier((y, outs[-1]))
        if scatter and c == chunks - 1:
            outs.append(reduce_scatter_rows(y, axis_name, rank, rs_table))
        elif scatter:
            outs.append(lax.psum_scatter(y, axis_name, scatter_dimension=0, tiled=True))
        else:
            # Reduce-scatter + all-gather, not one all-reduce of the chunk (#2196).
            outs.append(_psum_rows(y, axis_name))
    return outs[0] if chunks == 1 else jnp.concatenate(outs, 0)


# ------------------------------------------------------------------------------------ #
# The op: fused_moe_prefill's body with the levers as arguments. x_q / x_scale (A6)
# replace the in-op quantization when the caller's RMSNorm already emitted them.
# ------------------------------------------------------------------------------------ #
_FALLBACK_LOGGED = set()


def xmxu_supported(h, i, i_s):
    """Shapes the explicit-MXU body is written for: gate|up of the routed and the shared
    expert each one 256-column N-tile (2 i = 2 i_s = 256: GLM-5.2's TP16 slice), the
    down contraction padded up to one 256-deep K-tile, and h whole 256-column
    N-tiles."""
    return 2 * i == _NT and 2 * i_s == _NT and h % _NT == 0 and (h // _LANES) % 2 == 0


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
    nb=3,
    down="dot",
    issue="loop",
    tile_min=8192,
    flat=False,
    lag=None,
    x_q=None,
    x_scale=None,
    tall_rung_rows=None,
    sum_in_kernel_max_tokens=0,
    sum_overlap_adds=False,
    row_tokens_resident_max_tokens=0,
    scatter=False,
    core_major=False,
    rank=None,
    rs_table=None,
    cs_table=None,
):
    """fused_moe_prefill_kernel.fused_moe_prefill's op with per-row activation
    scales and the stage-A levers as arguments (the defaults are the per-row
    default build). x_q [T, h/128, 128] fp8 and x_scale [T] f32 replace the op's
    own quantization when given; they must be _quantize_rows of the op's input,
    bit for bit. Up to sum_in_kernel_max_tokens tokens (in whole tiles) the
    expert kernel applies the routing weights and sums each token's rows itself,
    and no slab or combine kernel is built; with sum_overlap_adds the one-pass
    build adds each tile's rows beside the next tile's gate/up matmuls. Up to
    row_tokens_resident_max_tokens tokens a build that writes the slab reads
    each route's token from the token table resident in SMEM instead of from a
    copied window of it: the same row DMAs, the same output (the in-kernel sum
    builds read it from SMEM already). With `scatter` it returns this core's
    rows of the sum only (`scatter_rows`); x may then be None when x_q is given
    and there is no shared expert gate. With `core_major` (and `scatter`) the
    rows arrive gathered in core order (`in_token_order`, TPU_MOE_PREFILL_GATHER_ONCE):
    the routing plan and the expert kernel take them so, the combine reads its
    row tables back in token order, and an in-kernel sum, whose rows stay in
    core order, ends in one reduce-scatter that leaves each core its own rows.
    With core_major, `rs_table` (rotated_collectives.peer_table
    "reduce_scatter") and `rank` (this core's mesh index, int32 [1]) the last
    combine chunk's reduce-scatter, or the one after an in-kernel sum, runs as
    rotated_collectives.reduce_scatter (TPU_MOE_PREFILL_ROTATED_COLLECTIVES)."""
    if core_major and not scatter:
        raise ValueError(
            "core-major rows are only reduce-scattered back to their cores"
        )
    if rs_table is not None and (not core_major or rank is None):
        raise ValueError("the rotated reduce-scatter needs core-major rows and a rank")
    if x_q is None:
        tokens, h = x.shape
    else:
        tokens, h = x_q.shape[0], x_q.shape[1] * x_q.shape[2]
    experts, _, two_i = w1.shape
    i_s = shared_w2.shape[0]
    routes = tokens * topk_ids.shape[1]
    weight_dtype = w1.dtype
    if down == "xmxu" and not xmxu_supported(h, two_i // 2, i_s):
        # Static shapes only: another TP degree (e.g. TP8, i = 256) gets the
        # per-row default build's levers, still per-row quantized.
        key = (h, two_i // 2, i_s)
        if key not in _FALLBACK_LOGGED:
            _FALLBACK_LOGGED.add(key)
            logger.warning(
                "FUSED_MOE_PREFILL_KERNEL_V2: h=%d i=%d i_s=%d is not a shape the "
                "stage-A body handles "
                "(gate|up must be one 256-column tile); running the per-row "
                "default build",
                h,
                two_i // 2,
                i_s,
            )
        nb, down, issue, tile_min, flat, tall_rung_rows = (
            3,
            "dot",
            "loop",
            8192,
            False,
            None,
        )
    tile_rows = tile_rows_for(tokens, tile_min)
    sum_cols = 0
    if tokens <= sum_in_kernel_max_tokens and tokens % tile_rows == 0:
        sum_cols = sum_columns(tokens, h, two_i // 2, i_s)
    # The in-kernel sum builds keep the token table resident on their own.
    resident_tokens = (
        issue == "loop" and not sum_cols and tokens <= row_tokens_resident_max_tokens
    )
    call, scale_rows, token_rows = build_expert_kernel(
        experts=experts,
        h=h,
        i=two_i // 2,
        i_s=i_s,
        tokens=tokens,
        routes=routes,
        tile_rows=tile_rows,
        weight_dtype=weight_dtype,
        nb=nb,
        down=down,
        issue=issue,
        lag=lag,
        rung_rows=tall_rung_rows if tile_rows > 128 else None,
        sum_cols=sum_cols,
        overlap_adds=sum_overlap_adds and sum_cols == h,
        resident_tokens=resident_tokens,
    )
    if x_q is None:
        x_q, x_scale = _quantize_rows(x, weight_dtype)
    route_valid = topk_weights != 0
    if expert_start is not None:
        topk_ids = topk_ids - expert_start
        local = (topk_ids >= 0) & (topk_ids < experts)
        route_valid = route_valid & local
        topk_ids = jnp.where(local, topk_ids, 0)
    coeff = jnp.where(route_valid, topk_weights.astype(jnp.float32), 0.0)
    if routed_scale != 1.0:
        coeff = coeff * routed_scale
    if shared_gate_weight is None:
        gate = jnp.ones((tokens, 1), jnp.float32)
    else:
        gate = jax.nn.sigmoid(
            jnp.dot(
                x,
                shared_gate_weight.reshape(h, 1).astype(x.dtype),
                preferred_element_type=jnp.float32,
            )
        )
    (
        row_token,
        slab_row,
        sorted_scale,
        visit_expert,
        visit_start,
        visit_rows,
        routed_visits,
        sorted_coeff,
    ) = _routing_plan(
        topk_ids, route_valid, experts, x_scale, coeff if sum_cols else None
    )
    visit_expert = visit_expert.at[routed_visits].set(experts)
    visit_start = visit_start.at[routed_visits].set(routes)
    visit_rows = visit_rows.at[routed_visits].set(tokens)
    # pyrefly: ignore[bad-argument-type]
    row_scale = jnp.concatenate([sorted_scale, x_scale])
    row_scale = jnp.pad(row_scale, (0, scale_rows * _LANES - row_scale.shape[0]))
    if 0 < sum_cols < h:
        row_token = jnp.concatenate([row_token, jnp.arange(tokens, dtype=jnp.int32)])
    row_token = jnp.pad(row_token, (0, token_rows - row_token.shape[0]))
    args = [visit_expert, visit_start, visit_rows, (routed_visits + 1).reshape(1)]
    if sum_cols:
        assert sorted_coeff is not None
        row_coeff = jnp.concatenate([sorted_coeff, gate.reshape(-1)])
        if sum_cols == h:
            row_coeff = jnp.pad(
                row_coeff, (0, scale_rows * _LANES - row_coeff.shape[0])
            )
            args += [
                row_token,
                x_q,
                row_scale.reshape(-1, _LANES),
                row_coeff.reshape(-1, _LANES),
            ]
        else:
            row_coeff = jnp.pad(row_coeff, (0, token_rows - row_coeff.shape[0]))
            args += [row_token, row_coeff, x_q, row_scale.reshape(-1, _LANES)]
    elif issue == "inline" or resident_tokens:
        args += [row_token, x_q, row_scale.reshape(-1, _LANES)]
    else:
        args += [x_q, row_scale.reshape(-1, _LANES), row_token]
    out = call(
        *args,
        w1,
        w2,
        w1_scale.reshape(experts, 1, two_i).astype(jnp.float32),
        w2_scale.reshape(experts, 1, h).astype(jnp.float32),
        shared_w1.reshape(1, h, 2 * i_s),
        shared_w2.reshape(1, i_s, h),
        shared_w1_scale.reshape(1, 1, 2 * i_s).astype(jnp.float32),
        shared_w2_scale.reshape(1, 1, h).astype(jnp.float32),
    )
    if sum_cols and core_major:
        return reduce_scatter_rows(out, axis_name, rank, rs_table)
    if sum_cols:
        return scatter_rows(out, axis_name) if scatter else _psum_rows(out, axis_name)
    shared_row = routes + jnp.arange(tokens, dtype=jnp.int32)[:, None]
    slab_row = jnp.where(route_valid, slab_row, shared_row)
    slab_row = jnp.concatenate([slab_row, shared_row], 1)
    coeff = jnp.concatenate([coeff, gate], 1)
    if core_major:
        ranks = lax.axis_size(axis_name)
        slab_row, coeff = in_token_order(slab_row, ranks), in_token_order(coeff, ranks)
    return combine(
        out,
        slab_row,
        coeff,
        axis_name,
        flat=flat,
        scatter=scatter,
        rank=rank,
        rs_table=rs_table,
        cs_table=cs_table,
    )


def fused_moe_prefill_stage_a(*args, **kwargs):
    """FUSED_MOE_PREFILL_KERNEL_V2: `fused_moe_prefill` at STAGE_A, with the
    same arguments as fused_moe_prefill_kernel.fused_moe_prefill, summing each
    token's expert rows in the kernel up to
    TPU_MOE_PREFILL_SUM_IN_KERNEL_MAX_TOKENS tokens and keeping the token table
    resident in SMEM up to TPU_MOE_PREFILL_ROW_TOKENS_RESIDENT_MAX_TOKENS
    tokens."""
    return fused_moe_prefill(
        *args,
        **kwargs,
        **dict(STAGE_A, tile_min=envs.TPU_MOE_PREFILL_TILE_MIN or STAGE_A["tile_min"]),
        sum_in_kernel_max_tokens=envs.TPU_MOE_PREFILL_SUM_IN_KERNEL_MAX_TOKENS,
        sum_overlap_adds=envs.TPU_MOE_PREFILL_SUM_IN_KERNEL_OVERLAP_ADDS,
        row_tokens_resident_max_tokens=envs.TPU_MOE_PREFILL_ROW_TOKENS_RESIDENT_MAX_TOKENS,
    )
