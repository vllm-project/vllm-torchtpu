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

from vllm_torchtpu.kernels.experimental.push_fused_moe import repeats
from vllm_torchtpu.kernels.experimental.push_fused_moe.host import (
    build_routing_tables_sharded,
    local_sort_activations,
    local_sort_tokens,
    push_dispatch_segments,
    push_dispatch_tables,
    routing_pairs_from_tables,
    run_tables,
)
from vllm_torchtpu.kernels.experimental.push_fused_moe.kernel import (
    build_combine_fs_kernel,
    build_push_moe_kernel,
)
from vllm_torchtpu.kernels.experimental.push_fused_moe.plan_kernel import (
    build_exchange_rows_tables,
    build_push_tables,
)
from vllm_torchtpu.kernels.experimental.push_fused_moe.routing import prepare_routing
from vllm_torchtpu.kernels.experimental.push_fused_moe.sc_dispatch import (
    sc_dispatch_chunk_2core,
)
from vllm_torchtpu.kernels.experimental.push_fused_moe.tc_exchange import (
    build_tc_exchange_all_gather,
    build_tc_exchange_all_gather_rows,
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
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    _absmax_rows,
    _quant_apply,
    _row_scale,
    combine_step_tokens,
)
from vllm_torchtpu.kernels.fused_moe.v2.layer import (
    SCORING_FNS,
    _apply_scoring_fn,
    _combine_arrivals,
    _relabel_expert_ids_to_mesh_order,
    _row_routes,
    _unpack_routing_exchange,
    _unpermute_load_to_ep_rank_order,
)
from vllm_torchtpu.kernels.fused_moe.v2.router_ops import pallas_select
from vllm_torchtpu.kernels.fused_moe.v2.token_parallel import TokenReplicaLayout
from vllm_torchtpu.kernels.quantized_matmul.shared_expert import (
    fp8_swiglu_mlp,
    is_supported,
    silu_and_mul_with_clamp,
)
from vllm_torchtpu.kernels.quantized_matmul.util import (
    get_max_min,
    xla_quantized_matmul,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The routing payload is padded to whole lane blocks: an unaligned payload
# (e.g. 192 experts) is lowered as an all-reduce, ~3x slower than the
# all-gather an aligned one gets.
EXCHANGE_WIDTH_ALIGN = 128
# light_prepush applies from this many local slots up: the heavy experts it
# takes off the pre-push need visits to arrive in (at DP16 the 9-slot kernel
# lost ~20 us a call with it while the 25-slot kernel held its time).
LIGHT_PREPUSH_MIN_SLOTS = 16


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
    arrivals, arrival_scales, pos, mirror_pos, topk_weights, out_dtype, runs=None
):
    """v2's `_combine_arrivals`, with the arrival-scale lookup done inside the
    combine kernel (see kernel._build_combine_fs_kernel) where that kernel
    applies; v2's XLA epilogue everywhere else.

    `runs` = (run_start [t], row_routes [t, 1]) folds the zeroing of rows that
    routed nowhere and the copy of each run's first row onto its repeats into
    the kernel; only where the kernel applies (`combine_step_tokens`)."""
    t_local, topk = pos.shape
    recv_rows, lane_blocks, lanes = arrivals.shape
    mirror_sublanes, tile_m = arrival_scales.shape
    if not combine_step_tokens(t_local):
        if runs is not None:
            raise ValueError(f"runs need the combine kernel; t_local={t_local}")
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
        run_rows=runs is not None,
    )
    return combine(arrivals, arrival_scales, scl, topk_weights, rows, runs or ())


def _shared_expert(x, row_max, w13, s13, w2, s2, *, limit):
    """`fp8_swiglu_mlp`, reusing `row_max` = max|x| per row [t, 1] (the
    routed quantization's) where it runs unfused, so x is reduced once.

    The unfused branch is `xla_quantized_matmul`'s floating-point gate-up
    with `quantize_block`'s steps on the given maximum: the same values, bit
    for bit."""
    if (
        row_max is None
        or not jnp.issubdtype(w13.dtype, jnp.floating)
        or is_supported(x, w13, s13, w2, s2)
    ):
        return fp8_swiglu_mlp(x, w13, s13, w2, s2, limit=limit)
    fp8_max, fp8_min = get_max_min(w13.dtype)
    scale = row_max / fp8_max
    scale = jnp.where(scale == 0, 1.0, scale)
    x_q = (x / scale).clip(fp8_min, fp8_max).astype(w13.dtype)
    gate_up = lax.dot_general(
        x_q,
        w13,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    gate_up = gate_up * scale.astype(jnp.float32) * jnp.expand_dims(s13, 0)
    gate, up = jnp.split(gate_up.astype(x.dtype), 2, axis=-1)
    return xla_quantized_matmul(silu_and_mul_with_clamp(gate, up, limit), w2, s2)


def _to_slots(topk_idx, expert_slot, row_base, *, onehot=False):
    """Selected global ids [rows, K] -> slots under an [experts, R] table.

    Row r's selection of e takes replica (row_base + r) mod (replicas of e),
    so a replicated expert's rows split evenly over its copies within every
    step. Positions are kept, so the k combine terms keep their order.

    `onehot` reads the selected expert's table row through one one-hot
    matmul instead of two gathers: XLA offloads a small gather to the
    SparseCore and the TensorCore waits for it, while this is one small dot.
    Same values: each output is one table entry (< 2**24) times 1.0 at
    HIGHEST precision.
    """
    experts, replicas = expert_slot.shape
    n_rep = jnp.sum(expert_slot >= 0, axis=1, dtype=jnp.int32)
    e = jnp.clip(topk_idx, 0, experts - 1)
    token = row_base + lax.broadcasted_iota(jnp.int32, topk_idx.shape, 0)
    if not onehot:
        replica = lax.rem(token, jnp.take(n_rep, e, axis=0))
        flat = e * replicas + replica
        return jnp.take(expert_slot.reshape(-1), flat, axis=0)
    hit = (e.reshape(-1, 1) == jnp.arange(experts, dtype=jnp.int32)).astype(jnp.float32)
    rows = jnp.concatenate([n_rep[:, None], expert_slot], axis=1).astype(jnp.float32)
    picked = jnp.dot(hit, rows, precision=lax.Precision.HIGHEST).astype(jnp.int32)
    replica = lax.rem(token.reshape(-1), picked[:, 0])
    slot = jnp.sum(
        jnp.where(
            replica[:, None] == jnp.arange(replicas, dtype=jnp.int32), picked[:, 1:], 0
        ),
        axis=1,
    )
    return slot.reshape(topk_idx.shape).astype(jnp.int32)


def _repeat_bits(bits, repeats):
    """jnp.repeat(bits, repeats) for i32 `bits` [n] as one one-hot matmul
    over the four bytes of each word.

    XLA spells the repeat as a [n, repeats] broadcast (one word in each
    128-lane row) and then flattens it; this builds the lane-dense
    [n / 128, 128 * repeats] result directly. Exact for any bit pattern: each
    output is one byte (< 256, exact in bf16) times 1.0, summed with zeros in
    f32.
    """
    (n_in,) = bits.shape
    lanes = HIDDEN_LANE_BLOCK
    n = align_up(n_in, lanes)
    bits = jnp.pad(bits, (0, n - n_in))
    shifts = 8 * jnp.arange(4, dtype=jnp.int32)[:, None, None]
    planes = (
        jnp.right_shift(bits.reshape(1, n // lanes, lanes), shifts) & 255
    ).reshape(4 * n // lanes, lanes)
    spread = (
        jnp.arange(lanes * repeats, dtype=jnp.int32)[None, :] // repeats
        == jnp.arange(lanes, dtype=jnp.int32)[:, None]
    ).astype(jnp.bfloat16)
    out = jnp.dot(
        planes.astype(jnp.bfloat16), spread, preferred_element_type=jnp.float32
    ).astype(jnp.int32)
    out = out.reshape(4, n // lanes, lanes * repeats)
    word = out[0] | (out[1] << 8) | (out[2] << 16) | (out[3] << 24)
    return word.reshape(n * repeats)[: n_in * repeats]


def _repeated_tail_start(rows):
    """Index of the first of the trailing rows that equal the last row.

    `rows` are [t, ...] arrays read together: a row repeats the last one when
    it matches it bit for bit in every array. 0 when every row matches.
    """
    t = rows[0].shape[0]
    same = jnp.ones((t,), jnp.bool_)
    for a in rows:
        if jnp.issubdtype(a.dtype, jnp.floating):
            a = lax.bitcast_convert_type(
                a, jnp.dtype(f"int{jnp.dtype(a.dtype).itemsize * 8}")
            )
        a = a.reshape(t, -1)
        same &= jnp.all(a == a[-1:], axis=1)
    idx = lax.broadcasted_iota(jnp.int32, (t,), 0)
    return jnp.max(jnp.where(same, -1, idx)) + 1


def _repeated_run_starts(rows, repeats_prev=None):
    """Per row, the index of the first row of its run of equal rows.

    `rows` are [t, ...] arrays read together: a row repeats the one before it
    when it matches it bit for bit in every array, and in `repeats_prev` [t]
    (another array's comparison, already made) where given.
    """
    t = rows[0].shape[0]
    same = jnp.ones((t - 1,), jnp.bool_) if repeats_prev is None else repeats_prev[1:]
    for a in rows:
        if jnp.issubdtype(a.dtype, jnp.floating):
            a = lax.bitcast_convert_type(
                a, jnp.dtype(f"int{jnp.dtype(a.dtype).itemsize * 8}")
            )
        a = a.reshape(t, -1)
        same &= jnp.all(a[1:] == a[:-1], axis=1)
    repeats = jnp.concatenate([jnp.zeros((1,), jnp.bool_), same])
    idx = lax.broadcasted_iota(jnp.int32, (t,), 0)
    return lax.cummax(jnp.where(repeats, 0, idx), axis=0)


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
    sc_prepush=2,
    tc_exchange=False,
    height_rungs_override=0,
    tail_barrier=True,
    onehot_slots=False,
    hbm_staging=False,
    dispatch_ahead=1,
    matmul_scale_repeat=False,
    exchange_row_gather=False,
    bf16_sort_counts=False,
    slim_prepush_barrier=False,
    fused_dispatch_meta=False,
    table_kernel=False,
    exchange_tables=False,
    kernel_plan_tables=False,
    light_prepush=0,
    single_barrier=False,
    early_scale_scatter=0,
    drop_repeated_tail=False,
    drop_repeated_runs=False,
    fused_runs=False,
    expert_slot=None,
    expert_load=None,
    shared_expert=None,
    shared_limit=None,
):
    """Run one MoE layer through the push-dispatch fused EP kernel.

    Same operands, contract and result as v2's `fused_ep_moe_v2` on its
    fused stage (see its docstring); `preprocess_topology` is accepted for
    call compatibility and unused, since no activation all-gather exists.

    `sc_prepush` is how many destination visit positions the SparseCore
    pushes before the TensorCore kernel: at least 1 (a smaller value raises),
    and a value above the experts per shard is clamped to it with a warning.
    `tc_exchange` runs the routing exchange as a TensorCore remote-DMA kernel
    instead of XLA's all-gather. `height_rungs_override` is the tile-height
    ladder's rung count (0 = the default; see kernel.tile_height_rungs).
    `tail_barrier=False` drops the kernel's closing all-pairs barrier (see
    kernel._build_push_moe_kernel), `onehot_slots` maps the layout's
    slots without gathers (see `_to_slots`), `hbm_staging` ships result
    tiles through the HBM contribution slab and `dispatch_ahead` is how many
    visit positions the TensorCore's dispatch pushes run ahead of its compute
    (see the kernel), `matmul_scale_repeat` builds the scale slab's per-pair
    activation scales with `_repeat_bits` instead of jnp.repeat, and
    `exchange_row_gather` (with `tc_exchange`) has the routing-exchange kernel
    also copy the expert-sorted activation rows, while it waits on its peers,
    instead of a TensorCore gather after it, `bf16_sort_counts` finds the
    expert-sorted token order with two single-pass bf16 matmuls
    (host._sorted_tokens_by_count), and `slim_prepush_barrier` keeps the run
    tables out of the pre-push barrier, and `fused_dispatch_meta` builds the
    pre-push counts, this shard's visit list and the kernel's remote-row
    count with the dispatch segments (host.push_dispatch_tables) instead of
    decoding them back out of them, and `table_kernel`
    builds the dispatch and shard tables in one TensorCore kernel
    (plan_kernel.build_push_tables) -- inside the routing-exchange kernel,
    while its row copies finish, with `exchange_tables`; with
    `kernel_plan_tables` the plan's small tables come from that kernel too
    (host.routing_pairs_from_tables); `light_prepush` (1 or 2) gives the
    SparseCore pre-push the lightest experts and orders the rest
    (plan_kernel._tables); `single_barrier` opens the expert kernel and the
    routing exchange with one barrier phase, not two; `early_scale_scatter`
    lets the scale-slab scatter run under the in-op shared expert (1: not
    held behind it; 2: nor behind the SparseCore pre-push);
    `drop_repeated_tail` routes only the first of the trailing rows that repeat
    this shard's last row bit for bit in what routing reads (the quantized row
    and its scale where activations are quantized, else x; gating, hash ids and
    a per-row score bias) and copies its result onto the others, so a bucket's
    padding rows, which are all alike, cost one row instead of loading the
    experts they all select. Each row's FFN, wire quantization and combine are
    per row, and an expert's replicas hold the same weights, so the copied
    result is the one the row would have computed; the dropped rows are left out
    of `expert_load`. `drop_repeated_runs` does the same for every run of rows
    that repeat the row before them, not only the trailing one, so padding rows
    that differ from the last row but come in runs still cost one row per run.
    `fused_runs` (with `drop_repeated_runs`) has the combine kernel zero
    unrouted rows and copy each run's first result onto its repeats itself
    (kernel._build_combine_fs_kernel `run_rows`), instead of an XLA select and
    gather over [rows, hidden] after it; the same values, bit for bit. The
    caller (layers/adapter/fused_moe_ep.py) reads all of them from envs.

    `expert_slot` [experts, R] i32 and `expert_load` [e_total] i32 (both
    replicated, optional) are the expert-layout operands dynamic EPLB
    (#2523) passes. With `expert_slot`, the weights' expert axis is in
    physical slot order (e_total slots over the mesh) while gating and
    score_bias stay per logical expert: selection runs on logical ids, then
    each selected id e maps to slot expert_slot[e, (global row) mod
    (copies of e)] (`_to_slots`), so a replicated expert's rows split evenly
    over its copies. With `expert_load`, the layer returns (output,
    expert_load + this step's routed rows per physical slot); the counts are
    the routing exchange's, identical on every shard. Without either, the
    program is unchanged.

    `shared_expert` (w13, s13, w2, s2), replicated, is the layer's per-channel
    FP8 shared expert (`kernels/quantized_matmul/shared_expert.fp8_swiglu_mlp`
    with `shared_limit`). It is computed on this shard's rows after the
    SparseCore pre-push is issued and before the expert kernel, so it runs
    while the pre-push is in flight; the layer then also returns it, last.
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
    if drop_repeated_tail and drop_repeated_runs:
        raise ValueError("drop_repeated_runs covers drop_repeated_tail; set one")
    if fused_runs and not drop_repeated_runs:
        raise ValueError("fused_runs needs drop_repeated_runs")
    T, hidden = x.shape
    e_total = w1.shape[0]
    has_expert_slot = expert_slot is not None
    has_expert_load = expert_load is not None
    has_shared = shared_expert is not None
    if has_shared and (len(shared_expert) != 4 or shared_limit is None):
        raise ValueError("shared_expert is (w13, s13, w2, s2) with a shared_limit")
    # Global experts the router scores; without a layout, one slot each.
    e_logical = gating.shape[1] if has_expert_slot else e_total
    if has_expert_slot and (
        expert_slot.ndim != 2
        or expert_slot.shape[0] != e_logical
        or expert_slot.dtype != jnp.int32
        or e_total < e_logical
    ):
        raise ValueError(
            f"expert_slot must be i32 [{e_logical}, replicas] over {e_total} "
            f"slots; got {jnp.dtype(expert_slot.dtype).name}"
            f"{tuple(expert_slot.shape)}"
        )
    if has_expert_load and (
        expert_load.shape != (e_total,) or expert_load.dtype != jnp.int32
    ):
        raise ValueError(
            f"expert_load must be i32 [{e_total}], one count per physical slot; "
            f"got {jnp.dtype(expert_load.dtype).name}{tuple(expert_load.shape)}"
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
    if has_score_bias and score_bias.shape not in (
        (e_logical,),
        (gating.shape[0], e_logical),
    ):
        raise ValueError(
            f"score_bias must be one value per global expert, [{e_logical}], "
            f"or one such row per token; got {tuple(score_bias.shape)}"
        )
    row_bias = score_bias is not None and score_bias.ndim == 2
    if has_hash_ids and (
        hash_topk_ids.shape != (x.shape[0], topk)
        or not jnp.issubdtype(hash_topk_ids.dtype, jnp.integer)
    ):
        raise ValueError(
            f"hash_topk_ids must be integer expert ids [{x.shape[0]}, {topk}]; "
            f"got {hash_topk_ids.dtype}{tuple(hash_topk_ids.shape)}"
        )
    # Visit positions [0, n_pre) go out on the SparseCore before the kernel;
    # at least one, so the kernel always starts on rows that have landed.
    n_pre = int(sc_prepush)
    if n_pre < 1:
        raise ValueError(f"sc_prepush must be >= 1; got {sc_prepush}")
    if n_pre > g_local:
        logger.warning_once(
            "push fused EP MoE: sc_prepush %d exceeds the %d experts per shard; "
            "using %d",
            n_pre,
            g_local,
            g_local,
        )
        n_pre = g_local
    tc_exchange = bool(tc_exchange)
    exchange_row_gather = bool(exchange_row_gather)
    if kernel_plan_tables and not (table_kernel and sharded_plan):
        raise ValueError("kernel_plan_tables needs table_kernel and sharded_plan")
    if exchange_tables and not (table_kernel and exchange_row_gather):
        raise ValueError("exchange_tables needs table_kernel and exchange_row_gather")
    light_prepush = int(light_prepush)
    if light_prepush not in (0, 1, 2):
        raise ValueError(f"light_prepush is 0, 1 or 2; got {light_prepush}")
    if light_prepush and not table_kernel:
        raise ValueError("light_prepush orders the visits in the table kernel")
    if g_local < LIGHT_PREPUSH_MIN_SLOTS:
        light_prepush = 0
    if exchange_row_gather and not tc_exchange:
        raise ValueError(
            "exchange_row_gather needs the TensorCore exchange (tc_exchange)"
        )
    height_rungs_override = int(height_rungs_override)
    tail_barrier = bool(tail_barrier)
    hbm_staging = bool(hbm_staging)
    # Every window past the last visit position issues the same program.
    dispatch_ahead = min(int(dispatch_ahead), g_local)
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
        height_rungs_override=height_rungs_override,
        tail_barrier=tail_barrier,
        hbm_staging=hbm_staging,
        dispatch_ahead=dispatch_ahead,
        single_barrier=bool(single_barrier),
    )

    layout = _exchange_layout(t_local, topk, e_total, form.quantized_activations)

    def exchange_all_gather(packed, rank_l):
        """The routing exchange all-gather of the flat i32 payload."""
        if tc_exchange:
            return build_tc_exchange_all_gather(
                ep=ep, width=packed.size, single_barrier=bool(single_barrier)
            )(packed, rank_l[0, 0])
        return lax.all_gather(packed, ax, axis=0, tiled=True)

    def unpack_operands(scales_gating_biases):
        """(w1 scale, w2 scale, gate, hash ids, score bias, slot table, load
        accumulator, shared expert, bias iterator)."""
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
        expert_slot_l = next(operands) if has_expert_slot else None
        expert_load_l = next(operands) if has_expert_load else None
        shared_l = tuple(next(operands) for _ in range(4)) if has_shared else None
        return (
            w1s_l,
            w2s_l,
            gate_l,
            hash_l,
            score_bias_l,
            expert_slot_l,
            expert_load_l,
            shared_l,
            operands,
        )

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
            # v2's rowquant_fp8, keeping the row maximum for the shared expert.
            row_max = _absmax_rows(rows_bf16)
            row_scale_l, sinv, usable = _row_scale(row_max)
            q_l = _quant_apply(rows_bf16, sinv.astype(rows_bf16.dtype), usable)
        else:
            q_l, row_scale_l, row_max = rows_bf16, None, None
        q_l = q_l.reshape(t_local, hidden // HIDDEN_LANE_BLOCK, HIDDEN_LANE_BLOCK)
        return gate_l, hash_l, q_l, row_scale_l, row_max

    def select_local(
        gate_l, hash_l, score_bias_l, row_scale_l, expert_slot_l, rank_l, keep=None
    ):
        """v2's: top-k, router weights, and this shard's routing payload;
        rows where `keep` [rows, 1] is False route nowhere."""
        me = rank_l[0, 0]
        if row_bias:
            score_bias_l = token_layout.split(score_bias_l, me)
        scores = _apply_scoring_fn(gate_l, scoring_fn)
        select_rows = MAX_ROUTING_BLOCK
        while t_local % select_rows:
            select_rows //= 2
        if hash_l is not None:
            topk_idx = hash_l.astype(jnp.int32)
            topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1)
            if expert_slot_l is not None:
                topk_idx = _to_slots(
                    topk_idx, expert_slot_l, me * t_local, onehot=onehot_slots
                )
        elif score_bias_l is None:
            topk_weights, topk_idx = pallas_select(
                scores, topk=topk, block_rows=select_rows
            )
        else:
            topk_weights, topk_idx = pallas_select(
                scores + score_bias_l.astype(jnp.float32),
                topk=topk,
                block_rows=select_rows,
                weight_scores=scores,
            )
        if expert_slot_l is not None and hash_l is None:
            topk_idx = _to_slots(
                topk_idx, expert_slot_l, me * t_local, onehot=onehot_slots
            )
        topk_idx = _relabel_expert_ids_to_mesh_order(
            topk_idx, g_local=g_local, mesh_ep_ranks=mesh_ep_ranks
        )
        # A row carrying no real score routes nowhere (see v2).
        row_routes = _row_routes(scores, gate_l)
        if keep is not None:
            row_routes &= keep
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
        expert_load_l=None,
        pre_tables=None,
        x_l=None,
        shared_l=None,
        tail_start=None,
        run_start=None,
        shared_row_max=None,
    ):
        """Dispatch tables, SparseCore pre-push, plan, kernel and combine.

        `pre_tables`, where the exchange kernel already built them, are
        build_push_tables' outputs; `shared_l`, the shared expert computed on
        `x_l` between the pre-push and the kernel, reusing `shared_row_max`
        (x_l's row maxima, or None) to quantize x_l."""
        me = rank_l[0, 0]
        pairs_g, scale_bits_g, rows_by_dest = _unpack_routing_exchange(
            exchange, layout, t_local, topk, e_total
        )
        # The step's rows per physical slot, summed over every source rank:
        # identical on every shard, so the accumulator stays replicated.
        updated_load = (
            None
            if expert_load_l is None
            else expert_load_l
            + _unpermute_load_to_ep_rank_order(
                rows_by_dest.sum(axis=0).astype(jnp.int32),
                g_local=g_local,
                mesh_ep_ranks=mesh_ep_ranks,
            )
        )
        with jax.named_scope("moe_dispatch_tables"):
            runs = run_tables(rows_by_dest)
            shard_tables = None
            if pre_tables is not None:
                seg, dmeta, visit, counts_sc, *shard_tables = pre_tables
            elif table_kernel:
                seg, dmeta, visit, counts_sc, *shard_tables = build_push_tables(
                    rows_by_dest,
                    seg_start,
                    me,
                    ep=ep,
                    n_pre=n_pre,
                    tile_m=capacity,
                    light_prepush=light_prepush,
                )
            elif fused_dispatch_meta:
                seg, dmeta, _, visit, counts_sc = push_dispatch_tables(
                    runs[3],
                    runs[2],
                    rows_by_dest,
                    me,
                    seg_start,
                    e_total=e_total,
                    ep=ep,
                    n_pre=n_pre,
                )
            else:
                seg, dmeta, visit_all = push_dispatch_segments(
                    runs[3],
                    runs[2],
                    rows_by_dest,
                    me,
                    seg_start,
                    e_total=e_total,
                    ep=ep,
                )
                recv_by_e = dmeta[g_local + 1 : 2 * g_local + 1]
                S = ep * g_local
                seg_n = jnp.bitwise_and(seg[:S], 65535).reshape(g_local, ep)
                seg_remote = (jnp.arange(ep)[None, :] != me).astype(jnp.int32)
                visit = lax.dynamic_slice(visit_all, (me, 0), (1, g_local)).reshape(
                    g_local
                )
                # Rows landing here at each pre-pushed visit position.
                vis_s = visit[:n_pre]
                hit_s = (
                    vis_s[:, None] == jnp.arange(g_local, dtype=jnp.int32)[None, :]
                ) & (vis_s[:, None] >= 0)
                rows_at_v = jnp.sum(jnp.where(hit_s, recv_by_e[None, :], 0), axis=1)
                # The kernel waits only the remote rows IT sends: positions >= n_pre.
                n_rem_tc = (
                    jnp.sum(seg_n[n_pre:] * seg_remote).astype(jnp.int32).reshape(1)
                )
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
        # The plan reads pairs_g, so it follows the barrier either way; the
        # run tables only add a layout copy at it (`slim_prepush_barrier`).
        barriered_runs = () if slim_prepush_barrier else tuple(runs)
        # The shared expert's rows come out of the same barrier as the
        # pre-push operands, so it cannot be scheduled ahead of them.
        shared_x = (x_l, shared_row_max) if shared_l is not None else ()
        (
            q_sorted,
            seg,
            counts_sc,
            dmeta,
            visit,
            pairs_g,
            scale_bits_g,
            *rest,
        ) = lax.optimization_barrier(
            (q_sorted, seg, counts_sc, dmeta, visit, pairs_g, scale_bits_g)
            + shared_x
            + barriered_runs
        )
        shared_x, barriered_runs = rest[: len(shared_x)], rest[len(shared_x) :]
        if not slim_prepush_barrier:
            runs = barriered_runs
        slab = sc_dispatch_chunk_2core(
            q_sorted,
            seg,
            counts_sc,
            ep=ep,
            g_local=g_local,
            v0=0,
            v1=n_pre,
            slab_rows=ragged_stride,
            name=f"moe_sc_prepush_2c_s{n_pre}",
            rank=me,
        )
        shared_out = ()
        if shared_l is not None:
            assert shared_limit is not None
            with jax.named_scope("moe_shared_expert"):
                shared_out = (_shared_expert(*shared_x, *shared_l, limit=shared_limit),)
        plan_kw = dict(
            e_total=e_total,
            ep=ep,
            t_local=t_local,
            block=block,
            tile_m=capacity,
            shard_stride=ragged_stride,
        )
        gathered = None
        if sharded_plan:
            gathered = GatheredPairs(
                expert_blocks=pairs_g.reshape(T * topk // block, block),
                local_blocks=local_blocks,
                block_hist=block_hist,
                rows_by_dest=rows_by_dest,
                n_tokens=T,
                topk=topk,
            )
        pair_tables = None
        routing_inputs = None
        routing = None
        if kernel_plan_tables:
            # The table kernel built the plan's small tables; only the
            # per-pair lookups are left.
            assert shard_tables is not None
            pair_tables = routing_pairs_from_tables(
                me,
                gathered,
                *shard_tables[4],
                e_total=e_total,
                ep=ep,
                t_local=t_local,
                block=block,
                shard_stride=ragged_stride,
            )
            slab_row = pair_tables[2]
        elif sharded_plan and shard_tables is None:
            # The sharded plan's consumer inputs, through the fused routing
            # metadata (routing.prepare_routing) where it admits the shape.
            assert gathered is not None
            routing_inputs = prepare_routing(me, gathered, tuple(runs), **plan_kw)
            slab_row = routing_inputs.slab_row
        else:
            if sharded_plan:
                routing = build_routing_tables_sharded(me, gathered, runs, **plan_kw)
            else:
                routing = build_routing_tables(pairs_g.reshape(T, topk), **plan_kw)
            # Both slab tables scatter onto this shard's own slab. The rows of
            # the other shards are dropped where they are computed rather than
            # built into a replicated slab and sliced away afterwards.
            slab_row = local_slab_rows(routing, me, shard_stride=ragged_stride)
        # Keep the scale-slab scatter (a SparseCore offload) behind the
        # pre-push on the SparseCore queue. The shared expert joins here, so
        # the expert kernel, which reads the slab, cannot start before it.
        if early_scale_scatter == 2:
            # Nor behind the pre-push: the scatter may take the SparseCore
            # first; the light pre-push still lands under the shared expert.
            slab_row, scale_bits_g = lax.optimization_barrier((slab_row, scale_bits_g))
        elif early_scale_scatter:
            # The shared expert joins after the scatter instead (below), so
            # the scatter's SparseCore work can run under it.
            slab_row, scale_bits_g, slab = lax.optimization_barrier(
                (slab_row, scale_bits_g, slab)
            )
        else:
            slab_row, scale_bits_g, slab, *shared_out = lax.optimization_barrier(
                (slab_row, scale_bits_g, slab) + shared_out
            )
        # The activation row scale, scattered onto the slab row each routed
        # pair computes on (v2's scale slab; see there).
        if form.quantized_activations:
            scale_bits = (
                _repeat_bits(cast(jax.Array, scale_bits_g), topk)
                if matmul_scale_repeat
                else jnp.repeat(cast(jax.Array, scale_bits_g), topk)
            )
            scale_rows = act_scale_slab_rows(ragged_stride)
            scale_slab = lax.bitcast_convert_type(
                jnp.zeros((scale_rows * HIDDEN_LANE_BLOCK,), jnp.int32)
                .at[slab_row]
                .add(scale_bits, mode="drop"),
                jnp.float32,
            ).reshape(scale_rows, HIDDEN_LANE_BLOCK)
        else:
            scale_slab = None
        if early_scale_scatter and shared_out and scale_slab is not None:
            scale_slab, slab, *shared_out = lax.optimization_barrier(
                (scale_slab, slab) + tuple(shared_out)
            )
        if shard_tables is not None:
            kernel_tables, expert_rows, slab_base, counts, _ = shard_tables
        elif routing_inputs is not None:
            kernel_tables = routing_inputs.kernel_tables
            expert_rows = routing_inputs.expert_rows
            slab_base = routing_inputs.slab_base
            counts = routing_inputs.counts
        else:
            assert routing is not None
            block_tables = shard_transport_tables_in_blocks(
                routing, me, e_total=e_total, ep=ep
            )
            row_tables = shard_push_tables_in_rows(routing, me, e_total=e_total, ep=ep)
            expert_rows, slab_base = shard_expert_slabs(
                routing, me, e_total=e_total, ep=ep
            )
            counts = shard_count_vector(
                routing, expert_rows, me, e_total=e_total, ep=ep
            )
            kernel_tables = (*block_tables[:3], block_tables[5], row_tables[0])
        recv_rows = align_up(t_local * topk + (ROWBLK - 1) * e_total, ROWBLK)
        w1b_k = next(bias_it).reshape(g_local, 2 * inter) if has_w1_bias else None
        w2b_k = next(bias_it).reshape(g_local, hidden) if has_w2_bias else None
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
        if pair_tables is not None:
            pos, mirror_pos = pair_tables[0], pair_tables[1]
        elif routing_inputs is not None:
            pos, mirror_pos = routing_inputs.pos, routing_inputs.mirror_pos
        elif sharded_plan:
            pos = cast(ShardRoutingTables, routing).pos
            mirror_pos = cast(ShardRoutingTables, routing).mirror_pos
        else:
            pos = lax.dynamic_slice(
                cast(RoutingTables, routing).arrival_row,
                (me * t_local, 0),
                (t_local, topk),
            )
            mirror_pos = lax.dynamic_slice(
                cast(RoutingTables, routing).mirror_row,
                (me * t_local, 0),
                (t_local, topk),
            )
        # With fused_runs the kernel zeroes unrouted rows and copies each run's
        # first result onto its repeats; below the combine kernel's token
        # minimum the XLA select and gather stay.
        fold_runs = (
            run_start is not None and fused_runs and bool(combine_step_tokens(t_local))
        )
        combined = _combine_arrivals_fs(
            arrivals,
            arrival_scales,
            pos,
            mirror_pos,
            topk_weights,
            out_dtype,
            runs=(run_start, row_routes) if fold_runs else None,
        )
        if not fold_runs:
            # A row that routed nowhere read arrival rows nobody wrote; its
            # weight is zero, but zero times an unwritten NaN is not.
            combined = jnp.where(row_routes, combined, jnp.zeros_like(combined))
        if tail_start is not None:
            # The dropped repeats of the tail's first row take its result.
            first = lax.dynamic_slice_in_dim(combined, tail_start, 1, axis=0)
            rows = lax.broadcasted_iota(jnp.int32, (t_local, 1), 0)
            combined = jnp.where(rows > tail_start, first, combined)
        if run_start is not None and not fold_runs:
            # The dropped repeats of each run take its first row's result.
            combined = jnp.take(combined, run_start, axis=0, mode="clip")
        out = token_layout.restore(combined, ax)
        outs = (out,) + (() if updated_load is None else (updated_load,))
        outs += tuple(shared_out)
        return outs if len(outs) > 1 else out

    def local_fn(x_l, rank_l, w1_l, w2_l, *scales_gating_biases):
        """One shard's half of the layer: route, dispatch, kernel, combine."""
        (
            w1s_l,
            w2s_l,
            gate_l,
            hash_l,
            score_bias_l,
            expert_slot_l,
            expert_load_l,
            shared_l,
            bias_it,
        ) = unpack_operands(scales_gating_biases)
        gate_l, hash_l, q_l, row_scale_l, row_max = quantize_local(
            x_l, rank_l, gate_l, hash_l
        )
        # The routed rows' maxima are x_l's own where the rows are not split
        # out of a replica group and need no cast.
        own_rows = len(token_layout.groups[0]) == 1 and x_l.dtype == jnp.bfloat16
        tail_start = run_start = keep = None
        if drop_repeated_tail or drop_repeated_runs:
            me = rank_l[0, 0]
            q_repeats = None
            # Shifted compares of q_l lower badly in XLA (a 1.44 ms slice at
            # 2048 rows), so runs read q_l only through repeats.py.
            if form.quantized_activations and (
                drop_repeated_tail or repeats.supports(q_l)
            ):
                # The routed experts read a row's FP8 values and scale, so rows
                # equal there route to equal results. q_l keeps its rows on the
                # major axis, where repeats.py compares them in place.
                row_inputs = [row_scale_l, gate_l]
                if drop_repeated_runs:
                    q_repeats = repeats.repeats_previous_row(q_l)
                else:
                    row_inputs.insert(0, q_l.reshape(t_local, -1))
            else:
                row_inputs = [token_layout.split(x_l, me), gate_l]
            if hash_l is not None:
                row_inputs.append(hash_l)
            if row_bias:
                row_inputs.append(token_layout.split(score_bias_l, me))
            rows = lax.broadcasted_iota(jnp.int32, (t_local, 1), 0)
            if drop_repeated_runs:
                run_start = _repeated_run_starts(row_inputs, q_repeats)
                keep = rows == run_start[:, None]
            else:
                tail_start = _repeated_tail_start(row_inputs)
                keep = rows <= tail_start
        topk_weights, packed, local_blocks, block_hist, row_routes = select_local(
            gate_l, hash_l, score_bias_l, row_scale_l, expert_slot_l, rank_l, keep
        )
        pre_tables = None
        my_counts = block_hist.sum(axis=0)
        if exchange_row_gather:
            # The sorted rows are copied by the exchange kernel itself.
            with jax.named_scope("moe_early_qsort"):
                tok, seg_start = local_sort_tokens(
                    local_blocks.reshape(-1),
                    my_counts,
                    e_total=e_total,
                    topk=topk,
                    bf16_counts=bf16_sort_counts,
                )
            if exchange_tables:
                exchange, q_sorted, *pre_tables = build_exchange_rows_tables(
                    ep=ep,
                    width=layout[-1],
                    n_rows=tok.shape[0],
                    count_at=layout[1],
                    e_total=e_total,
                    n_pre=n_pre,
                    tile_m=capacity,
                    light_prepush=light_prepush,
                    single_barrier=bool(single_barrier),
                )(packed, rank_l[0, 0], q_l, tok, seg_start)
            else:
                exchange, q_sorted = build_tc_exchange_all_gather_rows(
                    ep=ep,
                    width=layout[-1],
                    n_rows=tok.shape[0],
                    single_barrier=bool(single_barrier),
                )(packed, rank_l[0, 0], q_l, tok)
            exchange = exchange.reshape(ep, layout[-1])
        else:
            # Issue the routing exchange before the local sort, so the sort
            # runs on the TensorCore while the exchange is in flight.
            packed, q_l, local_blocks, my_counts = lax.optimization_barrier(
                (packed, q_l, local_blocks, my_counts)
            )
            exchange = exchange_all_gather(packed, rank_l).reshape(ep, layout[-1])
            with jax.named_scope("moe_early_qsort"):
                q_sorted, seg_start = local_sort_activations(
                    local_blocks.reshape(-1),
                    my_counts,
                    q_l,
                    e_total=e_total,
                    topk=topk,
                    bf16_counts=bf16_sort_counts,
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
            expert_load_l,
            pre_tables,
            x_l,
            shared_l,
            tail_start,
            run_start,
            row_max if own_rows else None,
        )

    bias_args = tuple(b for b in (w1_bias, w2_bias) if b is not None)
    scale_args = (w1_scale, w2_scale) if form.has_scales else ()
    # `rank` rides the mesh axis so each shard reads its own index out of it;
    # see v2 for why a torch_tpu caller must pass it as a real operand.
    if rank is None:
        rank = jnp.arange(ep, dtype=jnp.int32).reshape(ep, 1)
    score_bias_args = (score_bias,) if has_score_bias else ()
    hash_args = (hash_topk_ids,) if has_hash_ids else ()
    # Replicated like the score bias: indexed by global expert / slot id.
    expert_slot_args = (expert_slot,) if has_expert_slot else ()
    expert_load_args = (expert_load,) if has_expert_load else ()
    # Replicated: every shard holds the whole shared expert.
    shared_args = () if shared_expert is None else tuple(shared_expert)
    in_specs = (
        (P(ax),) * (5 + len(scale_args) + len(hash_args))
        # A per-token score bias rides with the gating rows.
        + (P(ax) if row_bias else P(),) * len(score_bias_args)
        + (P(),) * (len(expert_slot_args) + len(expert_load_args) + len(shared_args))
        + (P(ax),) * len(bias_args)
    )
    args = (
        (x, rank, w1, w2)
        + scale_args
        + (gating,)
        + hash_args
        + score_bias_args
        + expert_slot_args
        + expert_load_args
        + shared_args
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
        row_bias,
        has_hash_ids and cast(jax.Array, hash_topk_ids).dtype,
        has_expert_slot and tuple(cast(jax.Array, expert_slot).shape),
        has_expert_load,
        float(routed_scaling_factor),
        logical_intermediate_size,
        n_pre,
        tc_exchange,
        height_rungs_override,
        tail_barrier,
        bool(onehot_slots),
        hbm_staging,
        dispatch_ahead,
        bool(matmul_scale_repeat),
        exchange_row_gather,
        bool(bf16_sort_counts),
        bool(slim_prepush_barrier),
        bool(fused_dispatch_meta),
        bool(table_kernel),
        bool(exchange_tables),
        bool(kernel_plan_tables),
        has_shared and tuple((a.shape, a.dtype) for a in shared_args),
        float(shared_limit) if has_shared and shared_limit is not None else False,
        int(light_prepush),
        bool(single_barrier),
        int(early_scale_scatter),
        bool(drop_repeated_tail),
        bool(drop_repeated_runs),
        bool(fused_runs),
    )
    out_specs = (P(ax),) + ((P(),) if has_expert_load else ()) + ((P(ax),) * has_shared)
    out_specs = out_specs if len(out_specs) > 1 else out_specs[0]
    sm = _LAYER_SM_CACHE.get(key)
    if sm is None:
        with _LAYER_SM_CACHE_LOCK:
            sm = _LAYER_SM_CACHE.get(key)
            if sm is None:
                sm = jax.shard_map(
                    local_fn,
                    mesh=mesh,
                    in_specs=in_specs,
                    out_specs=out_specs,
                    check_vma=False,
                )
                _LAYER_SM_CACHE[key] = sm
    return sm(*args)
