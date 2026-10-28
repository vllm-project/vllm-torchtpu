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
"""Serving layer of the push-dispatch fused EP MoE.

`fused_ep_moe_push` is v2's `fused_ep_moe_v2` (fused stage) with the fp8
activation all-gather replaced by push dispatch:

1. every shard sorts its own quantized rows by global expert while the
   routing exchange (an all-gather of the packed top-k, row scales and
   per-expert counts, as in v2) is in flight;
2. from the exchanged counts it builds the dispatch segment table -- one
   entry per (destination, expert) run of its rows, in each destination's
   expert visit order;
3. a SparseCore kernel pushes the runs of the first `sc_prepush` visit
   positions into every destination's expert slab before the TensorCore
   kernel starts, overlapped with the routing plan;
4. the TensorCore kernel pushes the remaining visit positions itself, one
   position ahead of its own compute, then computes and combines exactly
   like v2.

Only data movement changes; the plan, the FFN and the combine arithmetic are
v2's.
"""

import threading
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.push_fused_moe.host import (
    build_routing_tables_sharded,
    local_sort_activations,
    push_dispatch_segments,
    run_tables,
)
from vllm_torchtpu.kernels.experimental.push_fused_moe.kernel import (
    build_combine_fs_kernel,
    build_push_moe_kernel,
)
from vllm_torchtpu.kernels.experimental.push_fused_moe.sc_dispatch import (
    sc_dispatch_chunk_2core,
)
from vllm_torchtpu.kernels.experimental.push_fused_moe.tc_exchange import (
    build_tc_exchange_all_gather,
)
from vllm_torchtpu.kernels.fused_moe.v2.host import (
    HIDDEN_LANE_BLOCK,
    MAX_ROUTING_BLOCK,
    QB4,
    ROWBLK,
    U32_SUBLANE_TILE,
    GatheredPairs,
    RoutingTables,
    ShardRoutingTables,
    WeightFormat,
    act_scale_slab_rows,
    align_up,
    build_routing_tables,
    fp4_scale_rows,
    local_slab_rows,
    pair_block_hist,
    ragged_stride_bound,
    routing_block,
    shard_count_vector,
    shard_expert_slabs,
    shard_push_tables_in_rows,
    shard_transport_tables_in_blocks,
    weight_form,
    weight_format_of_dtype,
)
from vllm_torchtpu.kernels.fused_moe.v2.kernel import combine_step_tokens, rowquant_fp8
from vllm_torchtpu.kernels.fused_moe.v2.layer import (
    SCORING_FNS,
    _apply_scoring_fn,
    _combine_arrivals,
    _relabel_expert_ids_to_mesh_order,
    _row_routes,
    _unpack_routing_exchange,
)
from vllm_torchtpu.kernels.fused_moe.v2.router_ops import pallas_select
from vllm_torchtpu.kernels.fused_moe.v2.token_parallel import TokenReplicaLayout

# The routing payload is padded to whole lane blocks: an unaligned payload
# (e.g. 192 experts) is lowered as an all-reduce, ~3x slower than the
# all-gather an aligned one gets.
EXCHANGE_WIDTH_ALIGN = 128


def _exchange_layout(t_local, topk, e_total, quantized):
    """(scale offset, count offset, width) of the flat routing payload: v2's
    layout with the width padded to EXCHANGE_WIDTH_ALIGN."""
    idx = align_up(t_local * topk, HIDDEN_LANE_BLOCK)
    scale = align_up(t_local, HIDDEN_LANE_BLOCK) if quantized else 0
    return idx, idx + scale, align_up(idx + scale + e_total, EXCHANGE_WIDTH_ALIGN)


def _pack_routing_exchange(pairs, scale_bits, counts, layout):
    """The flat i32 payload: pairs [t*K], row-scale bits [t], counts [E],
    zero-padded to the layout's width."""
    scale_at, count_at, width = layout
    parts = [pairs, jnp.zeros((scale_at - pairs.size,), jnp.int32)]
    if scale_bits is not None:
        parts += [
            scale_bits,
            jnp.zeros((count_at - scale_at - scale_bits.size,), jnp.int32),
        ]
    parts.append(counts)
    parts.append(jnp.zeros((width - count_at - counts.size,), jnp.int32))
    return jnp.concatenate([p for p in parts if p.size])


def _combine_arrivals_fs(
    arrivals, arrival_scales, pos, mirror_pos, topk_weights, out_dtype
):
    """v2's `_combine_arrivals`, with the arrival-scale lookup done inside the
    combine kernel (see kernel._build_combine_fs_kernel) where that kernel
    applies; v2's XLA epilogue everywhere else."""
    t_local, topk = pos.shape
    recv_rows, lane_blocks, lanes = arrivals.shape
    mirror_sublanes, tile_m = arrival_scales.shape
    if not combine_step_tokens(t_local):
        return _combine_arrivals(
            arrivals, arrival_scales, pos, mirror_pos, topk_weights, out_dtype
        )
    # A DMA does not clamp an out-of-range index the way a gather does.
    # Neither position is ever negative.
    rows = jnp.minimum(pos, recv_rows - 1)
    scl = jnp.minimum(mirror_pos, mirror_sublanes * tile_m - 1)
    combine = build_combine_fs_kernel(
        t_local=t_local,
        topk=topk,
        hidden=lane_blocks * lanes,
        lane_blocks=lane_blocks,
        wire_dtype=arrivals.dtype,
        out_dtype=out_dtype,
    )
    return combine(arrivals, arrival_scales, scl, topk_weights, rows)


def _remap_topk_ids(topk_idx, table, src_rank):
    """Static EPLB: logical top-k ids [rows, k] of rows routed from EP rank
    `src_rank` -> physical ids, one gather through the replicated replica
    table [E_logical, ep] (static-EPLB PR #2475's `remap_topk_ids`). Ids
    outside [0, E_logical) -- the empty-row sentinel -- pass through; the
    router weight stays with its top-k slot."""
    e = table.shape[0]
    valid = (topk_idx >= 0) & (topk_idx < e)
    safe = jnp.where(valid, topk_idx, 0)
    return jnp.where(valid, table[safe, src_rank], topk_idx).astype(topk_idx.dtype)


_LAYER_SM_CACHE = {}
_LAYER_SM_CACHE_LOCK = threading.Lock()


def fused_ep_moe_push(
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    gating,
    w1_bias=None,
    w2_bias=None,
    *,
    topk,
    renormalize,
    mesh,
    capacity,
    block=None,
    ragged_stride=None,
    weight_format=WeightFormat.FP8,
    rhs_qb=None,
    act_fn="silu",
    rank=None,
    mesh_ep_ranks=None,
    sharded_plan=False,
    scoring_fn="softmax",
    score_bias=None,
    hash_topk_ids=None,
    routed_scaling_factor=1.0,
    token_replica_groups=None,
    logical_intermediate_size=None,
    preprocess_topology=None,
    sc_prepush=None,
    tc_exchange=None,
    eplb_table=None,
):
    """Run one MoE layer through the push-dispatch fused EP kernel.

    Same operands, contract and result as v2's `fused_ep_moe_v2` on its
    fused stage (see its docstring); `preprocess_topology` is accepted for
    call compatibility and unused, since no activation all-gather exists.

    `sc_prepush` (default envs.MOE_FUSED_EP_PUSH_SC_PREPUSH) is how many
    destination visit positions the SparseCore pushes before the TensorCore
    kernel, clamped to [1, experts per shard]. `tc_exchange` (default
    envs.MOE_FUSED_EP_PUSH_TC_EXCHANGE) runs the routing exchange as a
    TensorCore remote-DMA kernel instead of XLA's all-gather.

    `eplb_table` [E_logical, ep] i32 (replicated, optional) is the static-EPLB
    hook of PR #2475: the weights then carry E_phys = w1.shape[0] experts
    (each shard's block is its home experts followed by replica slots), the
    gate, score bias and hash ids stay in logical expert ids, and the
    selected ids are remapped to physical ids right after top-k. Without it
    the program is the plain one.
    """
    del preprocess_topology
    form = weight_form(weight_format)
    rhs_qb = QB4 if rhs_qb is None else int(rhs_qb)
    for name, w in (("w1", w1), ("w2", w2)):
        actual = weight_format_of_dtype(w.dtype)
        if actual != weight_format:
            raise ValueError(
                f"weight format {weight_format!r} takes "
                f"{jnp.dtype(form.weight_dtype).name} expert weights and "
                f"{name} is {jnp.dtype(w.dtype).name}"
            )
    if form.has_scales != (w1_scale is not None):
        raise ValueError(
            f"weight format {weight_format!r} carries "
            f"{form.scale_layout} weight scales and w1_scale is "
            f"{'present' if w1_scale is not None else 'absent'}"
        )
    if (w1_scale is None) != (w2_scale is None):
        raise ValueError("both weight scales are supplied together or neither is")
    T, hidden = x.shape
    e_total = w1.shape[0]
    has_eplb = eplb_table is not None
    # The router's width: logical experts. Equal to e_total unless static
    # EPLB adds replica slots to the weights.
    e_logical = int(eplb_table.shape[0]) if has_eplb else e_total
    if has_eplb and not 0 < e_logical <= e_total:
        raise ValueError(
            f"eplb_table covers {e_logical} logical experts, weights have {e_total}"
        )
    inter = w1.shape[2] // 2
    pad_scale_rows = (
        form.has_scales
        and form.scale_layout == "per_contraction_block"
        and w1_scale.ndim == 2
    )
    if pad_scale_rows:
        for name, s, k, n in (
            ("w1_scale", w1_scale, hidden, 2 * inter),
            ("w2_scale", w2_scale, inter, hidden),
        ):
            want = (e_total * fp4_scale_rows(k, rhs_qb, padded=True), n)
            if tuple(s.shape) != want:
                raise ValueError(
                    f"{name} layout {tuple(s.shape)} is not the row-padded "
                    f"flat per_contraction_block form {want} the kernel takes"
                )
    elif form.has_scales:
        ndim = 3 if form.scale_layout == "per_contraction_block" else 2
        for name, s, n in (
            ("w1_scale", w1_scale, 2 * inter),
            ("w2_scale", w2_scale, hidden),
        ):
            if s.ndim != ndim or s.shape[0] != e_total or s.shape[-1] != n:
                raise ValueError(
                    f"{name} layout {tuple(s.shape)} is not the "
                    f"{form.scale_layout} form the kernel takes"
                )
    (ax,) = mesh.axis_names
    ep = mesh.shape[ax]
    if mesh_ep_ranks is not None:
        mesh_ep_ranks = tuple(int(r) for r in mesh_ep_ranks)
        if tuple(sorted(mesh_ep_ranks)) != tuple(range(ep)):
            raise ValueError(
                f"mesh_ep_ranks must be a permutation of range({ep}); got "
                f"{mesh_ep_ranks}"
            )
    g_local = e_total // ep
    if T % ep or gating.shape != (T, e_logical):
        raise ValueError(
            "input and gating rows must agree and partition evenly over EP"
        )
    token_layout = TokenReplicaLayout.create(
        ep=ep,
        input_rows=T // ep,
        groups=token_replica_groups,
        row_alignment=U32_SUBLANE_TILE,
    )
    t_local = token_layout.local_rows
    T = ep * t_local
    P = jax.sharding.PartitionSpec
    stride_bound = ragged_stride_bound(T, topk, e_total, capacity)
    if ragged_stride is None:
        ragged_stride = stride_bound
    if block is None:
        block = routing_block(t_local, topk)
    if ragged_stride % capacity != 0 or ragged_stride < stride_bound:
        raise ValueError(
            f"ragged_stride {ragged_stride} must be a multiple of capacity "
            f"{capacity} and at least the no-drop bound {stride_bound}"
        )
    has_w1_bias = w1_bias is not None
    has_w2_bias = w2_bias is not None
    has_score_bias = score_bias is not None
    has_hash_ids = hash_topk_ids is not None
    if scoring_fn not in SCORING_FNS:
        raise ValueError(f"scoring_fn must be one of {SCORING_FNS}; got {scoring_fn!r}")
    if has_score_bias and score_bias.shape != (e_logical,):
        raise ValueError(
            f"score_bias must be one value per global expert, [{e_logical}]; "
            f"got {tuple(score_bias.shape)}"
        )
    if has_hash_ids and (
        hash_topk_ids.shape != (x.shape[0], topk)
        or not jnp.issubdtype(hash_topk_ids.dtype, jnp.integer)
    ):
        raise ValueError(
            f"hash_topk_ids must be integer expert ids [{x.shape[0]}, {topk}]; "
            f"got {hash_topk_ids.dtype}{tuple(hash_topk_ids.shape)}"
        )
    if sc_prepush is None:
        sc_prepush = envs.MOE_FUSED_EP_PUSH_SC_PREPUSH
    # Visit positions [0, n_pre) go out on the SparseCore before the kernel;
    # at least one, so the kernel always starts on rows that have landed.
    n_pre = max(1, min(int(sc_prepush), g_local))
    if tc_exchange is None:
        tc_exchange = envs.MOE_FUSED_EP_PUSH_TC_EXCHANGE
    tc_exchange = bool(tc_exchange)
    kfn = build_push_moe_kernel(
        g_local=g_local,
        capacity=capacity,
        hidden=hidden,
        inter=inter,
        ep=ep,
        ragged_rows_alloc=ragged_stride,
        weight_format=weight_format,
        rhs_qb=rhs_qb,
        act_fn=act_fn,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
        pad_scale_rows=pad_scale_rows,
        logical_intermediate_size=logical_intermediate_size,
        push_tokens=t_local,
        sc_prepush=n_pre,
    )

    layout = _exchange_layout(t_local, topk, e_total, form.quantized_activations)

    def exchange_all_gather(packed, rank_l):
        """The routing exchange all-gather of the flat i32 payload."""
        if tc_exchange:
            return build_tc_exchange_all_gather(ep=ep, width=packed.size)(
                packed, rank_l[0, 0]
            )
        return lax.all_gather(packed, ax, axis=0, tiled=True)

    def unpack_operands(scales_gating_biases):
        """(w1 scale, w2 scale, gate, hash ids, score bias, EPLB table, bias
        iterator)."""
        operands = iter(scales_gating_biases)
        if pad_scale_rows:
            # [G * rows, N] -> [G, rows, N]: whole sublane tiles per expert,
            # so this is a bitcast of the operand rather than a relayout.
            w1s_l = next(operands).reshape(
                g_local, fp4_scale_rows(hidden, rhs_qb, padded=True), 2 * inter
            )
            w2s_l = next(operands).reshape(
                g_local, fp4_scale_rows(inter, rhs_qb, padded=True), hidden
            )
        else:
            w1s_l = next(operands) if form.has_scales else None
            w2s_l = next(operands) if form.has_scales else None
        gate_l = next(operands)
        hash_l = next(operands) if has_hash_ids else None
        score_bias_l = next(operands) if has_score_bias else None
        eplb_l = next(operands) if has_eplb else None
        return w1s_l, w2s_l, gate_l, hash_l, score_bias_l, eplb_l, operands

    def quantize_local(x_l, rank_l, gate_l, hash_l):
        """v2's: split this shard's rows out of a replica group, quantize."""
        me = rank_l[0, 0]
        x_l = token_layout.split(x_l, me)
        gate_l = token_layout.split(gate_l, me, fill_value=-jnp.inf)
        if hash_l is not None:
            hash_l = token_layout.split(hash_l, me)
        if len(token_layout.groups[0]) > 1 and form.quantized_activations:
            # Slicing fused into rowquant changes BF16 rounding on TPU.
            # Keep this boundary only for quantized replicated inputs.
            x_l = lax.optimization_barrier(x_l)
        rows_bf16 = x_l.astype(jnp.bfloat16)
        if form.quantized_activations:
            q_l, row_scale_l = rowquant_fp8(rows_bf16)
        else:
            q_l, row_scale_l = rows_bf16, None
        q_l = q_l.reshape(t_local, hidden // HIDDEN_LANE_BLOCK, HIDDEN_LANE_BLOCK)
        return gate_l, hash_l, q_l, row_scale_l

    def select_local(gate_l, hash_l, score_bias_l, row_scale_l, eplb_l, rank_l):
        """v2's: top-k, router weights, and this shard's routing payload."""
        scores = _apply_scoring_fn(gate_l, scoring_fn)
        select_rows = MAX_ROUTING_BLOCK
        while t_local % select_rows:
            select_rows //= 2
        if hash_l is not None:
            topk_idx = hash_l.astype(jnp.int32)
            topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1)
        elif score_bias_l is None:
            topk_weights, topk_idx = pallas_select(
                scores, topk=topk, block_rows=select_rows
            )
        else:
            topk_weights, topk_idx = pallas_select(
                scores + score_bias_l.astype(jnp.float32)[None, :],
                topk=topk,
                block_rows=select_rows,
                weight_scores=scores,
            )
        if eplb_l is not None:
            # Logical -> physical (EP-rank order) before the mesh relabel,
            # which then moves whole (home + replica) blocks. `me` is this
            # shard's mesh index; the table is indexed by EP rank.
            me = rank_l[0, 0]
            src = me if mesh_ep_ranks is None else jnp.asarray(mesh_ep_ranks)[me]
            topk_idx = _remap_topk_ids(
                topk_idx.astype(jnp.int32), eplb_l, src.astype(jnp.int32)
            )
        topk_idx = _relabel_expert_ids_to_mesh_order(
            topk_idx, g_local=g_local, mesh_ep_ranks=mesh_ep_ranks
        )
        # A row carrying no real score routes nowhere (see v2).
        row_routes = _row_routes(scores, gate_l)
        if renormalize:
            denom = jnp.maximum(topk_weights.sum(axis=-1, keepdims=True), 1e-20)
            topk_weights = topk_weights / jnp.where(row_routes, denom, 1.0)
        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * jnp.float32(routed_scaling_factor)
        topk_weights = jnp.where(row_routes, topk_weights, 0.0)
        topk_idx = jnp.where(row_routes, topk_idx, jnp.int32(e_total))
        local_blocks = topk_idx.astype(jnp.int32).reshape(
            t_local * topk // block, block
        )
        block_hist = pair_block_hist(local_blocks, e_total)
        scale_bits_l = (
            None
            if row_scale_l is None
            else (
                lax.bitcast_convert_type(
                    row_scale_l[:, 0].astype(jnp.float32), jnp.int32
                )
            )
        )
        packed = _pack_routing_exchange(
            local_blocks.reshape(-1), scale_bits_l, block_hist.sum(axis=0), layout
        )
        return topk_weights, packed, local_blocks, block_hist, row_routes

    def compute_local(
        q_sorted,
        seg_start,
        exchange,
        local_blocks,
        block_hist,
        topk_weights,
        row_routes,
        rank_l,
        w1_l,
        w2_l,
        w1s_l,
        w2s_l,
        bias_it,
        out_dtype,
    ):
        """Dispatch tables, SparseCore pre-push, plan, kernel and combine."""
        me = rank_l[0, 0]
        pairs_g, scale_bits_g, rows_by_dest = _unpack_routing_exchange(
            exchange, layout, t_local, topk, e_total
        )
        with jax.named_scope("moe_dispatch_tables"):
            runs = run_tables(rows_by_dest)
            seg, dmeta, visit_all = push_dispatch_segments(
                runs[3], runs[2], rows_by_dest, me, seg_start, e_total=e_total, ep=ep
            )
            recv_by_e = dmeta[g_local + 1 : 2 * g_local + 1]
            S = ep * g_local
            seg_n = jnp.bitwise_and(seg[:S], 65535).reshape(g_local, ep)
            seg_remote = (jnp.arange(ep)[None, :] != me).astype(jnp.int32)
            visit = lax.dynamic_slice(visit_all, (me, 0), (1, g_local)).reshape(g_local)
            # Rows landing here at each pre-pushed visit position.
            vis_s = visit[:n_pre]
            hit_s = (
                vis_s[:, None] == jnp.arange(g_local, dtype=jnp.int32)[None, :]
            ) & (vis_s[:, None] >= 0)
            rows_at_v = jnp.sum(jnp.where(hit_s, recv_by_e[None, :], 0), axis=1)
            # The kernel waits only the remote rows IT sends: positions >= n_pre.
            n_rem_tc = jnp.sum(seg_n[n_pre:] * seg_remote).astype(jnp.int32).reshape(1)
            dmeta = jnp.concatenate(
                [dmeta[: 2 * g_local + 1], n_rem_tc, dmeta[2 * g_local + 2 :]]
            )
            # SparseCore core c pushes the positions v with v % 2 == c.
            counts_sc = jnp.stack(
                [
                    jnp.sum(rows_at_v[0:n_pre:2]),
                    jnp.sum(rows_at_v[1:n_pre:2]),
                    jnp.sum(seg_n[0:n_pre:2] * seg_remote),
                    jnp.sum(seg_n[1:n_pre:2] * seg_remote),
                ]
            ).astype(jnp.int32)
        # Compute the pre-push operands before the routing plan, so XLA
        # launches the SparseCore dispatch ahead of it and the two overlap.
        (
            q_sorted,
            seg,
            counts_sc,
            dmeta,
            visit,
            pairs_g,
            scale_bits_g,
            *runs,
        ) = lax.optimization_barrier(
            (q_sorted, seg, counts_sc, dmeta, visit, pairs_g, scale_bits_g, *runs)
        )
        slab = sc_dispatch_chunk_2core(
            q_sorted,
            seg,
            counts_sc,
            ep=ep,
            g_local=g_local,
            v0=0,
            v1=n_pre,
            slab_rows=ragged_stride,
            name=f"moe_sc_hybrid_2c_s{n_pre}",
            rank=me,
        )
        plan_kw = dict(
            e_total=e_total,
            ep=ep,
            t_local=t_local,
            block=block,
            tile_m=capacity,
            shard_stride=ragged_stride,
        )
        if sharded_plan:
            gathered = GatheredPairs(
                expert_blocks=pairs_g.reshape(T * topk // block, block),
                local_blocks=local_blocks,
                block_hist=block_hist,
                rows_by_dest=rows_by_dest,
                n_tokens=T,
                topk=topk,
            )
            routing = build_routing_tables_sharded(me, gathered, runs, **plan_kw)
        else:
            routing = build_routing_tables(pairs_g.reshape(T, topk), **plan_kw)
        # Both slab tables scatter onto this shard's own slab. The rows of
        # the other shards are dropped where they are computed rather than
        # built into a replicated slab and sliced away afterwards.
        slab_row = local_slab_rows(routing, me, shard_stride=ragged_stride)
        # Keep the scale-slab scatter (a SparseCore offload) behind the
        # pre-push on the SparseCore queue.
        slab_row, scale_bits_g, slab = lax.optimization_barrier(
            (slab_row, scale_bits_g, slab)
        )
        block_tables = shard_transport_tables_in_blocks(
            routing, me, e_total=e_total, ep=ep
        )
        row_tables = shard_push_tables_in_rows(routing, me, e_total=e_total, ep=ep)
        # The activation row scale, scattered onto the slab row each routed
        # pair computes on (v2's scale slab; see there).
        if form.quantized_activations:
            scale_bits = jnp.repeat(cast(jax.Array, scale_bits_g), topk)
            scale_rows = act_scale_slab_rows(ragged_stride)
            scale_slab = lax.bitcast_convert_type(
                jnp.zeros((scale_rows * HIDDEN_LANE_BLOCK,), jnp.int32)
                .at[slab_row]
                .add(scale_bits, mode="drop"),
                jnp.float32,
            ).reshape(scale_rows, HIDDEN_LANE_BLOCK)
        else:
            scale_slab = None
        expert_rows, slab_base = shard_expert_slabs(routing, me, e_total=e_total, ep=ep)
        recv_rows = align_up(t_local * topk + (ROWBLK - 1) * e_total, ROWBLK)
        w1b_k = next(bias_it).reshape(g_local, 2 * inter) if has_w1_bias else None
        w2b_k = next(bias_it).reshape(g_local, hidden) if has_w2_bias else None
        counts = shard_count_vector(routing, expert_rows, me, e_total=e_total, ep=ep)
        kernel_tables = (*block_tables[:3], block_tables[5], row_tables[0])
        arrivals, arrival_scales = kfn(
            kernel_tables,
            expert_rows,
            slab_base,
            q_sorted,
            scale_slab,
            w1_l,
            w2_l,
            w1s_l,
            w2s_l,
            w1b_k,
            w2b_k,
            recv_rows=recv_rows,
            visit=visit,
            counts=counts,
            rank=rank_l,
            dispatch=(seg, dmeta),
            slab=slab,
        )

        # The destination's own table: one arrival row per selection slot.
        pos = (
            cast(ShardRoutingTables, routing).pos
            if sharded_plan
            else lax.dynamic_slice(
                cast(RoutingTables, routing).arrival_row,
                (me * t_local, 0),
                (t_local, topk),
            )
        )
        mirror_pos = (
            cast(ShardRoutingTables, routing).mirror_pos
            if sharded_plan
            else lax.dynamic_slice(
                cast(RoutingTables, routing).mirror_row,
                (me * t_local, 0),
                (t_local, topk),
            )
        )
        combined = _combine_arrivals_fs(
            arrivals, arrival_scales, pos, mirror_pos, topk_weights, out_dtype
        )
        # A row that routed nowhere read arrival rows nobody wrote; its weight is
        # zero, but zero times an unwritten NaN is not.
        combined = jnp.where(row_routes, combined, jnp.zeros_like(combined))
        return token_layout.restore(combined, ax)

    def local_fn(x_l, rank_l, w1_l, w2_l, *scales_gating_biases):
        """One shard's half of the layer: route, dispatch, kernel, combine."""
        w1s_l, w2s_l, gate_l, hash_l, score_bias_l, eplb_l, bias_it = unpack_operands(
            scales_gating_biases
        )
        gate_l, hash_l, q_l, row_scale_l = quantize_local(x_l, rank_l, gate_l, hash_l)
        topk_weights, packed, local_blocks, block_hist, row_routes = select_local(
            gate_l, hash_l, score_bias_l, row_scale_l, eplb_l, rank_l
        )
        # Issue the routing exchange before the local sort, so the sort runs
        # on the TensorCore while the exchange is in flight.
        my_counts = block_hist.sum(axis=0)
        packed, q_l, local_blocks, my_counts = lax.optimization_barrier(
            (packed, q_l, local_blocks, my_counts)
        )
        exchange = exchange_all_gather(packed, rank_l).reshape(ep, layout[-1])
        with jax.named_scope("moe_early_qsort"):
            q_sorted, seg_start = local_sort_activations(
                local_blocks.reshape(-1), my_counts, q_l, e_total=e_total, topk=topk
            )
        exchange, q_sorted, seg_start = lax.optimization_barrier(
            (exchange, q_sorted, seg_start)
        )
        return compute_local(
            q_sorted,
            seg_start,
            exchange,
            local_blocks,
            block_hist,
            topk_weights,
            row_routes,
            rank_l,
            w1_l,
            w2_l,
            w1s_l,
            w2s_l,
            bias_it,
            x_l.dtype,
        )

    bias_args = tuple(b for b in (w1_bias, w2_bias) if b is not None)
    scale_args = (w1_scale, w2_scale) if form.has_scales else ()
    # `rank` rides the mesh axis so each shard reads its own index out of it;
    # see v2 for why a torch_tpu caller must pass it as a real operand.
    if rank is None:
        rank = jnp.arange(ep, dtype=jnp.int32).reshape(ep, 1)
    score_bias_args = (score_bias,) if has_score_bias else ()
    hash_args = (hash_topk_ids,) if has_hash_ids else ()
    # The EPLB table is indexed by logical expert id like the score bias, and
    # every shard remaps its own rows, so it is replicated too.
    eplb_args = (eplb_table,) if has_eplb else ()
    in_specs = (
        (P(ax),) * (5 + len(scale_args) + len(hash_args))
        + (P(),) * len(score_bias_args)
        + (P(),) * len(eplb_args)
        + (P(ax),) * len(bias_args)
    )
    args = (
        (x, rank, w1, w2)
        + scale_args
        + (gating,)
        + hash_args
        + score_bias_args
        + eplb_args
        + bias_args
    )
    NS = jax.sharding.NamedSharding
    args = tuple(jax.device_put(a, NS(mesh, sp)) for a, sp in zip(args, in_specs))
    key = (
        mesh,
        T,
        hidden,
        e_total,
        inter,
        topk,
        bool(renormalize),
        capacity,
        block,
        ragged_stride,
        weight_format,
        rhs_qb,
        act_fn,
        mesh_ep_ranks,
        token_layout,
        bool(sharded_plan),
        x.dtype,
        w1.dtype,
        w2.dtype,
        form.has_scales and w1_scale.dtype,
        pad_scale_rows,
        gating.dtype,
        has_w1_bias and w1_bias.dtype,
        has_w2_bias and w2_bias.dtype,
        scoring_fn,
        has_score_bias and cast(jax.Array, score_bias).dtype,
        has_hash_ids and hash_topk_ids.dtype,
        has_eplb and (tuple(eplb_table.shape), str(eplb_table.dtype)),
        float(routed_scaling_factor),
        logical_intermediate_size,
        n_pre,
        tc_exchange,
    )
    sm = _LAYER_SM_CACHE.get(key)
    if sm is None:
        with _LAYER_SM_CACHE_LOCK:
            sm = _LAYER_SM_CACHE.get(key)
            if sm is None:
                sm = jax.shard_map(
                    local_fn,
                    mesh=mesh,
                    in_specs=in_specs,
                    out_specs=P(ax),
                    check_vma=False,
                )
                _LAYER_SM_CACHE[key] = sm
    return sm(*args)
