# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""FUSED_MOE_PREFILL_KERNEL_V2: the fused MoE prefill kernel with the t4-moe-prefill
stage-A levers.

Same op, slab layout and numerics as fused_moe_prefill_kernel (its helpers are imported,
not copied); what changes is how the expert kernel and the combine spend their issue
slots. Build arguments (v1's value first):

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

from vllm_torchtpu.kernels.experimental.fused_moe_prefill.fused_moe_prefill_kernel import (  # noqa: E501
    _GATHER_UNROLL,
    _LANES,
    _LOOKAHEAD,
    _TILE_RUNGS,
    _WEIGHT_DMA_PRIORITY,
    _byte_transpose4,
    _cdiv,
    _psum_rows,
    _quantize_rows,
    _routing_plan,
)
from vllm_torchtpu.kernels.fused_moe.v2 import host as v2_host
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    COMBINE_CHUNK,
    COMBINE_GROUP,
    COMBINE_LEAD,
    COMBINE_PARITIES,
    DN,
    HIDDEN_LANE_BLOCK,
    _intermediate_rows,
    _tile_row_scales,
    build_combine_kernel,
    combine_step_tokens,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_COMBINE_CHUNK_TOKENS = 2048
# What FUSED_MOE_PREFILL_KERNEL_V2 serves (t4-moe-prefill probes 03-04, v7x, GLM-5.2
# TP16 per-rank shapes; device time per call of the whole op against the default build).
# 4096 tokens: -134 us uniform routing, -85 Zipf s 0.3, -78 s 0.6, -72 s 1.0, -65
# top-heavy; 2048 -26 / -23; 1024 -7; 512 and 256 unchanged. down="xmxu" carries 2048
# and below and 32-52 us of 4096; tile_min=4096 on 32-row rungs (tall_rung_rows) the
# rest of 4096 -- on v1's 64-row rungs the taller tile pads visit tails and lost 18 us
# under top-heavy routing; flat 0.1-3.6 us. issue="inline" stays off (+7..+18 us: its
# predicated issue takes bundles for every block of every tile); nb 3 and 4 measure
# within 3 us. Shapes the explicit-MXU body does not handle run v1's build.
STAGE_A = dict(nb=4, down="xmxu", tile_min=4096, flat=True, tall_rung_rows=32)

_NT = 256  # explicit-MXU N-tile width (MXU column size on v7x)


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
):
    assert down in ("dot", "xmxu") and issue in ("loop", "inline")
    assert not (issue == "inline" and down != "xmxu"), (
        "inline issue lives in the xmxu N-tile sequence"
    )
    inline = issue == "inline"
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
    # padded contraction of the down matmul (the explicit MXU takes K = 256); unpadded
    # under "dot"
    kp = _cdiv(i, _NT) * _NT if xmxu else i
    kp_s = _cdiv(i_s, _NT) * _NT if xmxu else i_s
    shared = experts
    slab_rows = routes + tokens + tile_rows
    scale_rows = _cdiv(slab_rows, _LANES) + 1
    token_rows = _cdiv(routes, _LANES) * _LANES + window
    n_blk_max = tile_rows // _GATHER_UNROLL
    # inline issue: block b of the next tile is issued after N-tile blocks_at.index(b)
    blocks_at = [
        [b for b in range(n_blk_max) if b * nt // n_blk_max == n] for n in range(nt)
    ]

    def kernel(*refs):
        refs = list(refs)
        visit_expert, visit_start, visit_rows, num_visits = refs[:4]
        k0 = 4
        row_token_sm = None
        if inline:
            row_token_sm = refs[4]
            k0 = 5
        x_hbm, row_scale_hbm = refs[k0], refs[k0 + 1]
        k0 += 2
        row_token_hbm = None
        if not inline:
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
            slab_hbm,
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
        if not inline:
            token_window = refs[k0]
            k0 += 1
        act_sems = refs[k0]
        k0 += 1
        if not inline:
            window_sems = refs[k0]
            k0 += 1
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
            return [
                pltpu.make_async_copy(s, d, ring_sems.at[slot])
                for s, d in zip(srcs, dsts)
            ]

        sw2_dst = shared_w2_vmem if kp_s == i_s else shared_w2_vmem.at[pl.ds(0, i_s)]
        shared_copies = [
            pltpu.make_async_copy(s, d, shared_sem)
            for s, d in zip(
                (
                    shared_w1_hbm.at[0],
                    shared_w2_hbm.at[0],
                    shared_w1_scale_hbm.at[0],
                    shared_w2_scale_hbm.at[0],
                ),
                (shared_w1_vmem, sw2_dst, shared_w1_scale_vmem, shared_w2_scale_vmem),
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

        # ---- token-number windows (loop issue only) ----
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
            if inline:
                return

            @pl.when(jnp.logical_and(v < n_visit, jnp.logical_not(is_shared(v))))
            def _():
                window_copy(tile_start(v, t), slot).start()

        def wait_window(v, t, slot):
            if inline:
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
                            if inline:
                                assert row_token_sm is not None
                                token = row_token_sm[ptr + u]
                            else:
                                assert token_window is not None
                                token = token_window[ptr + u]
                            row_copy(token, slot, r0 + u).start()
                        return ptr + _GATHER_UNROLL

                    ptr0 = start if inline else wslot * window + lax.rem(start, _LANES)
                    lax.fori_loop(
                        0, _cdiv(tile_live(v, t), _GATHER_UNROLL), block, ptr0
                    )

            scale_copy(start, slot).start()

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
        # zero the padded contraction rows of the down weights once (never DMA'd over)
        if kp != i:
            w2_ring[:, pl.ds(i, kp - i), :] = jnp.zeros((nb, kp - i, h), w2_ring.dtype)
        if kp_s != i_s:
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

        def tile_body(carry):
            v, t, g, n_out0, n_out1, vi, ti = carry
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

            def gateup_xmxu(rows, blocks, w1):
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
                    if k + 2 < kt:
                        push(k + 2)
                return pltpu.matmul_pop(accs[0]) + pltpu.matmul_pop(accs[1])

            down_accs = {}

            def compute(rows, w1, w2, w1_scale, w2_scale, inter):
                if xmxu:
                    # allocated here (inside this copy's run_scoped), shared by both
                    # parity branches
                    L = lag if lag is not None else _lag_for(rows, nt)
                    L = min(
                        L, 2 * (acc_entries_rows // rows - 1) - 1
                    )  # one tile of ACC stays with gate/up
                    S = L // 2 + 1
                    down_accs[rows] = (
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
                act_scale = _tile_row_scales(act_scale_vmem[aslot], start, rows)
                if xmxu:  # Mosaic refuses explicit MXU ops beside a dot in one program
                    acc1 = gateup_xmxu(rows, build_lhs(rows, as_blocks=True), w1)
                else:
                    acc1 = lax.dot_general(
                        build_lhs(rows), w1[...], DN, preferred_element_type=jnp.float32
                    )
                mid_q, mid_scale = _intermediate_rows(
                    acc1, act_scale, inter, w1s=w1_scale[...], act_fn="silu", w1b=None
                )
                assert mid_scale is not None
                half = lane_blocks // 2
                for par in range(2):

                    @pl.when(p == par)
                    def _(par=par):
                        out32 = (
                            out_vmem.at[par]
                            .bitcast(jnp.uint32)
                            .reshape(tile_rows * half, _LANES)
                            if out_by_pairs
                            else None
                        )
                        if xmxu:
                            down_xmxu(rows, mid_q, mid_scale, w2, w2_scale, out32, half)
                            return
                        acc2 = lax.dot_general(
                            mid_q, w2[...], DN, preferred_element_type=jnp.float32
                        )
                        out = acc2 * mid_scale * w2_scale[...]
                        if out_by_pairs:
                            assert out32 is not None
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
                            out = out.reshape(rows, lane_blocks, _LANES)
                            out_vmem[par, pl.ds(0, rows)] = out.astype(jnp.bfloat16)

            def down_xmxu(rows, mid_q, mid_scale, w2, w2_scale, out32, half):
                """out[:, n-tile] = pop(acc_n) * mid_scale * w2s[n-tile], one 256-column
                N-tile at a time on alternating MXUs; the pop of N-tile n waits `lag`
                N-tiles (>= 32 vmatmul slots)."""
                k_in = mid_q.shape[1]
                k_pad = w2.shape[0]
                lhs = (
                    mid_q
                    if k_in == k_pad
                    else jnp.concatenate(
                        [mid_q, jnp.zeros((rows, k_pad - k_in), mid_q.dtype)], axis=1
                    )
                )
                L, S, accs = down_accs[rows]

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
                for n in range(nt):
                    if n + 1 < nt:
                        push(n + 1)
                    pltpu.matmul_acc_lhs(acc_of(n), lhs=lhs, load_staged_rhs=0)
                    if n >= L:
                        epi(n - L)
                    issue_at(n)
                for n in range(nt - L, nt):
                    epi(n)

            def scoped(f):
                if xmxu:
                    pl.run_scoped(f)
                else:
                    f()

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
                        )
                    )

                @pl.when(jnp.logical_not(is_shared(v)))
                def _():
                    scoped(
                        lambda: compute(
                            rows,
                            w1_ring.at[slot],
                            w2_ring.at[slot],
                            w1_scale_ring.at[slot],
                            w2_scale_ring.at[slot],
                            i,
                        )
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
    n_in = 11 - (1 if inline else 0)
    scratch = [
        pltpu.VMEM((n_act, tile_rows, lane_blocks, _LANES), weight_dtype),
        pltpu.VMEM((n_act, scale_window, _LANES), jnp.float32),
        pltpu.VMEM((2, tile_rows, lane_blocks, _LANES), jnp.bfloat16),
        pltpu.VMEM((nb, h, 2 * i), weight_dtype),
        pltpu.VMEM((nb, kp, h), weight_dtype),
        pltpu.VMEM((nb, 1, 2 * i), jnp.float32),
        pltpu.VMEM((nb, 1, h), jnp.float32),
    ]
    if not inline:
        scratch.append(pltpu.SMEM((2 * window,), jnp.int32))
    scratch.append(pltpu.SemaphoreType.DMA((n_act,)))
    if not inline:
        scratch.append(pltpu.SemaphoreType.DMA((2,)))
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
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5 if inline else 4,
            in_specs=[hbm] * n_in,
            out_specs=hbm,
            scratch_shapes=tuple(scratch),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((slab_rows, lane_blocks, _LANES), jnp.bfloat16),
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


def combine(slab, slab_row, coeff, axis_name, flat=False):
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
):
    """fused_moe_prefill_kernel.fused_moe_prefill with the stage-A levers as arguments.
    x_q [T, h/128, 128] fp8 and x_scale [T] f32
    (fused_add_rmsnorm.fused_add_rms_norm_quant_moe) replace the op's own quantization
    when given; they must be _quantize_rows of the op's input, bit for bit."""
    if x_q is None:
        tokens, h = x.shape
    else:
        tokens, h = x_q.shape[0], x_q.shape[1] * x_q.shape[2]
    experts, _, two_i = w1.shape
    i_s = shared_w2.shape[0]
    routes = tokens * topk_ids.shape[1]
    weight_dtype = w1.dtype
    if down == "xmxu" and not xmxu_supported(h, two_i // 2, i_s):
        # Static shapes only: another TP degree (e.g. TP8, i = 256) gets v1's levers --
        # the incumbent's body, which takes x_q / x_scale like the rest of this
        # function.
        key = (h, two_i // 2, i_s)
        if key not in _FALLBACK_LOGGED:
            _FALLBACK_LOGGED.add(key)
            logger.warning(
                "FUSED_MOE_PREFILL_KERNEL_V2: h=%d i=%d i_s=%d is not a shape the "
                "stage-A body handles "
                "(gate|up must be one 256-column tile); running v1's build",
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
    )
    if x_q is None:
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
    visit_expert = visit_expert.at[routed_visits].set(experts)
    visit_start = visit_start.at[routed_visits].set(routes)
    visit_rows = visit_rows.at[routed_visits].set(tokens)
    row_scale = jnp.concatenate([sorted_scale, x_scale])
    row_scale = jnp.pad(row_scale, (0, scale_rows * _LANES - row_scale.shape[0]))
    row_token = jnp.pad(row_token, (0, token_rows - routes))
    args = [visit_expert, visit_start, visit_rows, (routed_visits + 1).reshape(1)]
    if issue == "inline":
        args += [row_token, x_q, row_scale.reshape(-1, _LANES)]
    else:
        args += [x_q, row_scale.reshape(-1, _LANES), row_token]
    slab = call(
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
    shared_row = routes + jnp.arange(tokens, dtype=jnp.int32)[:, None]
    slab_row = jnp.where(route_valid, slab_row, shared_row)
    coeff = jnp.where(route_valid, topk_weights.astype(jnp.float32), 0.0)
    if routed_scale != 1.0:
        coeff = coeff * routed_scale
    return combine(
        slab,
        jnp.concatenate([slab_row, shared_row], 1),
        jnp.concatenate([coeff, gate], 1),
        axis_name,
        flat=flat,
    )
