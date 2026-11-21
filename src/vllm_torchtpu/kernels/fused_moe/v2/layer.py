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
"""Serving layer for the fused expert-parallel MoE kernel.

fused_ep_moe_v2 wraps the kernel builder in a shard_map over the
expert-parallel mesh axis: quantize, dispatch, kernel, combine.
"""

import threading

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.fused_moe.v2.host import (
    HIDDEN_LANE_BLOCK,
    MAX_ROUTING_BLOCK,
    QB4,
    ROWBLK,
    U32_SUBLANE_TILE,
    GatheredPairs,
    WeightFormat,
    act_scale_slab_rows,
    align_up,
    build_routing_tables,
    build_routing_tables_sharded,
    expert_visit_list,
    local_slab_rows,
    pair_block_hist,
    pow2_shift,
    ragged_stride_bound,
    routing_block,
    shard_count_vector,
    shard_expert_slabs,
    shard_push_tables_in_rows,
    shard_token_gather,
    shard_transport_tables_in_blocks,
    token_gather_window_rows,
    weight_form,
    weight_format_of_dtype,
)
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    build_combine_kernel,
    build_fused_ep_moe_kernel,
    combine_step_tokens,
    rowquant_fp8,
)
from vllm_torchtpu.kernels.fused_moe.v2.router_ops import pallas_select
from vllm_torchtpu.kernels.fused_moe.v2.token_parallel import TokenReplicaLayout

# The scoring functions the router can apply to the gate logits. These
# mirror `layers/adapter/moe_routing._apply_scoring_fn` exactly.
SCORING_FNS = ("softmax", "sigmoid", "sqrtsoftplus")


def _apply_scoring_fn(logits, scoring_fn):
    """Gate logits -> f32 routing scores."""
    scores = logits.astype(jnp.float32)
    if scoring_fn == "softmax":
        return jax.nn.softmax(scores, axis=-1)
    if scoring_fn == "sigmoid":
        return jax.nn.sigmoid(scores)
    if scoring_fn == "sqrtsoftplus":
        return jnp.sqrt(jax.nn.softplus(scores))
    raise NotImplementedError(
        f"the fused EP MoE router scores with {SCORING_FNS}; got {scoring_fn!r}"
    )


def _row_routes(scores, gate_l):
    """Per-token `[rows, 1]` mask: True iff the row has a finite score and logit.

    `gate_l` is checked too: under sigmoid/sqrtsoftplus a `-inf` logit scores a
    finite `0`, so replica padding (`token_layout.split(..., fill_value=-jnp.inf)`)
    would otherwise pass the score check.
    """
    return jnp.any(jnp.isfinite(scores), axis=-1, keepdims=True) & jnp.any(
        jnp.isfinite(gate_l), axis=-1, keepdims=True
    )


def _combine_arrivals(
    arrivals,
    arrival_scales,
    pos,
    mirror_pos,
    topk_weights,
    out_dtype,
    *,
    topk_weights_1d=None,
):
    """Combine arrivals [rows, lane blocks, 128] into [t_local, hidden].

    Two tables index two different objects, so each owns its own clamp.
    `pos` is the arrival ROW: a row of `arrivals`, and on the kernel path
    below it is a DMA address. `mirror_pos` is a position in the wire's scale
    mirror, which is [sublanes, tile_m] because a transport can only address
    the mirror a sublane at a time and a sublane holds tile_m f32 whatever is
    written in it -- so a whole source TILE's scales are one sublane and the
    mirror position is NOT the arrival row.

    Where the kernel path applies (kernel.combine_step_tokens) the row gather
    and the weighted sum are one Mosaic call: XLA answers `arrivals[rows]`
    with a SparseCore gather whose output is in HBM at any size, so the rows
    are written out and read straight back before the sum sees them.
    Everywhere else the gather is not offloaded and there is no round trip to
    remove, so the fused XLA epilogue below stays.

    Both paths fold the arrival row's scale and the router weight into one
    coefficient over K * t_local values, so each of the K * t_local * hidden
    elements is multiplied once rather than twice.
    """
    t_local, topk = pos.shape
    recv_rows, lane_blocks, lanes = arrivals.shape
    mirror_sublanes, tile_m = arrival_scales.shape
    hidden = lane_blocks * lanes
    # A gather clamps an out-of-range index to the last one; a DMA does not,
    # and a sublane index clamped on its own would pair the last sublane with
    # the wrong lane. Neither position is ever negative.
    shift = pow2_shift(tile_m, "the mirror's lane count")
    if combine_step_tokens(t_local):
        combine = build_combine_kernel(
            t_local=t_local,
            topk=topk,
            hidden=hidden,
            lane_blocks=lane_blocks,
            wire_dtype=arrivals.dtype,
            out_dtype=out_dtype,
        )
        # Avoid 2D [t_local, topk] clamp and scale multiply on the kernel path:
        # topk < 128 pads the minor axis to a full lane block and adds a retile.
        rows_1d = jnp.minimum(pos.reshape(-1), recv_rows - 1)
        scl_1d = jnp.minimum(mirror_pos.reshape(-1), mirror_sublanes * tile_m - 1)
        row_scales_1d = arrival_scales[
            jnp.right_shift(scl_1d, shift), scl_1d & (tile_m - 1)
        ]
        tw_1d = (
            topk_weights_1d
            if topk_weights_1d is not None
            else topk_weights.astype(jnp.float32).reshape(-1)
        )
        return combine(arrivals, row_scales_1d * tw_1d, rows_1d)
    scl = jnp.minimum(mirror_pos, mirror_sublanes * tile_m - 1)
    row_scales = arrival_scales[jnp.right_shift(scl, shift), scl & (tile_m - 1)]
    coef = row_scales * topk_weights.astype(jnp.float32)
    rows_fp8 = arrivals[pos.T.reshape(-1)]  # f8 [K*t, lane blocks, 128]
    coef_k = coef.T.reshape(-1)  # k-major [K*t_local]
    terms = []
    for k in range(topk):
        slot_rows = lax.slice(
            rows_fp8, (k * t_local, 0, 0), ((k + 1) * t_local, lane_blocks, lanes)
        )
        slot_coef = lax.slice(coef_k, (k * t_local,), ((k + 1) * t_local,))
        terms.append(slot_rows.astype(jnp.float32) * slot_coef[:, None, None])
    return sum(terms).astype(out_dtype).reshape(t_local, hidden)


def _exchange_layout(t_local, topk, e_total, quantized):
    """(scale offset, count offset, width) of the flat routing payload.

    ONE all-gather carries everything the other shards need to know about
    this shard's tokens: the selections, the row scales the wire quantized
    them by, and the per-expert pair counts the routing plan exchanges. The
    plan's own count collective disappears into this one, and the selections
    arrive already flat -- the pair grid the plan reads is a reshape of the
    gathered payload rather than a [T, topk] array whose 10 live lanes sit in
    a 128-lane tile.

    Section starts are lane-block aligned so every unpack slice is a view.
    """
    idx = align_up(t_local * topk, HIDDEN_LANE_BLOCK)
    scale = align_up(t_local, HIDDEN_LANE_BLOCK) if quantized else 0
    return idx, idx + scale, idx + scale + e_total


def _pack_routing_exchange(pairs, scale_bits, counts, layout):
    """The flat i32 payload: pairs [t*K], row-scale bits [t], counts [E].

    The router weight stays local: the destination combine applies it in f32
    from its own copy, so the wire never carries it.
    """
    scale_at, count_at, _ = layout
    parts = [pairs, jnp.zeros((scale_at - pairs.size,), jnp.int32)]
    if scale_bits is not None:
        parts += [
            scale_bits,
            jnp.zeros((count_at - scale_at - scale_bits.size,), jnp.int32),
        ]
    parts.append(counts)
    return jnp.concatenate([p for p in parts if p.size])


def _unpack_routing_exchange(exchange, layout, t_local, topk, e_total):
    """Inverse of _pack_routing_exchange over the gathered [ep, width] rows.

    Returns the gathered pairs [ep, t*K], the row-scale BIT PATTERNS [T] (the
    scale slab wants them as int32, so nothing is bitcast back) and the
    [ep, e_total] count table. The bits are None where the payload has none.
    """
    scale_at, count_at, _ = layout
    ep = exchange.shape[0]
    pairs = lax.slice(exchange, (0, 0), (ep, t_local * topk))
    scale_bits = (
        None
        if scale_at == count_at
        else lax.slice(exchange, (0, scale_at), (ep, scale_at + t_local)).reshape(-1)
    )
    counts = lax.slice(exchange, (0, count_at), (ep, count_at + e_total))
    return pairs, scale_bits, counts


def _relabel_expert_ids_to_mesh_order(topk_idx, *, g_local, mesh_ep_ranks):
    """Map selected experts from EP-rank order to device-mesh order.

    vLLM numbers contiguous expert blocks by EP rank, while the Mosaic remote
    DMA kernel addresses peers by their position in the device-id-ordered JAX
    mesh. ``mesh_ep_ranks[i]`` names the EP rank at mesh index ``i``.

    Apply the permutation after top-k selection. Reordering the full router
    logits before selection materializes a ``[tokens, experts]`` gather at the
    Torch/JAX boundary, and can also change which expert wins an exact tie.
    Here only the ``[tokens, topk]`` integer IDs move; the selected weights keep
    the arithmetic and tie behavior of the unpermuted router.
    """
    if mesh_ep_ranks is None:
        return topk_idx
    ep = len(mesh_ep_ranks)
    if tuple(sorted(mesh_ep_ranks)) != tuple(range(ep)):
        raise ValueError(
            f"mesh_ep_ranks must be a permutation of range({ep}); got {mesh_ep_ranks}"
        )
    owning_rank = topk_idx // g_local
    mesh_block = jnp.zeros_like(topk_idx)
    for mesh_index, ep_rank in enumerate(mesh_ep_ranks):
        mesh_block = jnp.where(
            owning_rank == ep_rank, jnp.int32(mesh_index), mesh_block
        )
    return mesh_block * g_local + topk_idx % g_local


# Local tokens at or above which the activation all-gather is enqueued ahead
# of the slab scatters. Measured, not chosen: +2.80% at t_local 2048 and
# -0.01 / -7.81 / -1.92 / -1.94% at 1024 / 256 / 64 / 32, because the scatters
# lose ~90 us of cover to buy a ~129 us hoist and only the largest shape's
# plan is long enough to pay for it.
AGQ_MIN_LOCAL_TOKENS = 1024

# Per-config cache of the shard_map'd MoE callable; a fresh shard_map
# body would re-trace the whole kernel for every hidden layer.
_LAYER_SM_CACHE = {}
_LAYER_SM_CACHE_LOCK = threading.Lock()


TP_PAIR_MAX_ROW_SPLITS = 4
TP_PAIR_VISIT_FLOOR_TILES = 2.4
TP_PAIR_Z_CLONE_MIN_TOTAL = 2048
TP_PAIR_Z_CLONE_MIN_REMOTE = 1024
# Avoid scheduling the cloned visit early (delay <= 4): waiting on the Z-link
# weight DMA during peak early-visit combine traffic regresses skewed layers
# by +160..+240 us/L; appending after local visits hides the weight transfer
# behind cold experts.
TP_PAIR_Z_CLONE_DELAY_STEPS = 1024


def _ep_tp_pair_flip_mask(counts_2):
    """Return alternating-parity swap mask [n_chips, g_local] for paired experts."""
    c0, c1 = counts_2[:, 0, :], counts_2[:, 1, :]
    hi = jnp.maximum(c0, c1)
    lo = jnp.minimum(c0, c1)
    # Sort by 2*max - min so asymmetric heavy pairs alternate across cores while
    # balanced (H, H) pairs do not skew the alternating parity.
    pair_key = (2 * hi) - lo
    pair_rank = jnp.argsort(jnp.argsort(-pair_key, axis=1, stable=True), axis=1).astype(
        jnp.int32
    )
    return jnp.where((pair_rank & 1) == 0, c1 > c0, c0 > c1)


def _pack_flip_words(flip):
    n_pairs = flip.size
    n_words = (n_pairs + 31) // 32
    flip_pad = jnp.pad(
        flip.reshape(-1).astype(jnp.uint32), (0, n_words * 32 - n_pairs)
    ).reshape(n_words, 32)
    bit_weights = jnp.left_shift(jnp.uint32(1), jnp.arange(32, dtype=jnp.uint32))
    return jnp.sum(flip_pad * bit_weights[None, :], axis=1, dtype=jnp.uint32)


def _select_by_chip(chip_idx, per_chip_vals, n_chips):
    # Avoid 1D dynamic LUT gather over T*topk pairs: XLA offloads it to
    # SparseCore, whereas a small unrolled chip select stays on the VPU.
    val = per_chip_vals[0]
    for ci in range(1, n_chips):
        val = jnp.where(chip_idx == jnp.int32(ci), per_chip_vals[ci], val)
    return val


def _choose_ep_tp_row_splits(counts_chip, *, g_local, k_split, capacity):
    """Pick per-chip split count ks in [0, k_split] minimizing lockstep tiles."""
    n_chips = counts_chip.shape[0]
    top_order = jnp.argsort(-counts_chip, axis=1, stable=True).astype(jnp.int32)
    split_phys = top_order[:, :k_split]
    split_rows = jnp.take_along_axis(counts_chip, split_phys, axis=1)
    phys_idx = jnp.arange(2 * g_local, dtype=jnp.int32)[None, :]
    min_split_rows = jnp.int32(2 * capacity)
    k_arange = jnp.arange(k_split, dtype=jnp.int32)[None, :]
    floor_tiles = jnp.float32(TP_PAIR_VISIT_FLOOR_TILES)

    def _tile_cost(rows):
        tiles = ((rows + jnp.int32(capacity - 1)) // jnp.int32(capacity)).astype(
            jnp.float32
        )
        return jnp.where(rows > 0, jnp.maximum(tiles, floor_tiles), jnp.float32(0.0))

    costs = []
    for ks in range(k_split + 1):
        act = (k_arange < jnp.int32(ks)) & (split_rows >= min_split_rows)
        is_sp = jnp.any(
            (phys_idx[:, :, None] == split_phys[:, None, :]) & act[:, None, :],
            axis=-1,
        )
        rem_2 = jnp.where(is_sp, jnp.int32(0), counts_chip).reshape(n_chips, 2, g_local)
        fl = _ep_tp_pair_flip_mask(rem_2)
        c0_rem = jnp.where(fl, rem_2[:, 1, :], rem_2[:, 0, :])
        c1_rem = jnp.where(fl, rem_2[:, 0, :], rem_2[:, 1, :])
        h0 = jnp.where(act, (split_rows + jnp.int32(1)) // jnp.int32(2), jnp.int32(0))
        h1 = jnp.where(act, split_rows // jnp.int32(2), jnp.int32(0))
        v0 = jnp.sort(jnp.concatenate([c0_rem, h0], axis=1), axis=1)[:, ::-1]
        v1 = jnp.sort(jnp.concatenate([c1_rem, h1], axis=1), axis=1)[:, ::-1]
        # Tiny tie-break favors fewer splits when lockstep cost is identical.
        step_cost = jnp.sum(
            jnp.maximum(_tile_cost(v0), _tile_cost(v1)), axis=1
        ) + jnp.float32(1e-4 * ks)
        costs.append(step_cost)

    best_ks = jnp.argmin(jnp.stack(costs, axis=0), axis=0).astype(jnp.int32)
    split_active = (k_arange < best_ks[:, None]) & (split_rows >= min_split_rows)
    is_split_phys = jnp.any(
        (phys_idx[:, :, None] == split_phys[:, None, :]) & split_active[:, None, :],
        axis=-1,
    )
    rem_2 = jnp.where(is_split_phys, jnp.int32(0), counts_chip).reshape(
        n_chips, 2, g_local
    )
    flip = _ep_tp_pair_flip_mask(rem_2)
    return split_phys, split_active, is_split_phys, flip


def _resolve_ep_tp_row_splits(
    *,
    ep_tp_size,
    has_bias,
    g_local,
    n_tokens,
    topk,
    ep,
    capacity,
    user_ragged_stride,
    ep_tp_row_splits,
):
    if ep_tp_size != 2 or has_bias or g_local <= 0 or (g_local & (g_local - 1)) != 0:
        return 0
    if ep_tp_row_splits is None:
        if n_tokens < 2 * capacity:
            return 0
        candidate = min(TP_PAIR_MAX_ROW_SPLITS, g_local)
    else:
        candidate = min(max(int(ep_tp_row_splits), 0), g_local)
    if candidate <= 0:
        return 0
    split_stride = ragged_stride_bound(
        n_tokens, topk, ep * (g_local + candidate), capacity
    )
    if user_ragged_stride is not None and user_ragged_stride < split_stride:
        return 0
    return candidate


def _resolve_ep_tp_z_clone(
    *,
    ep_tp_size,
    has_bias,
    weight_format,
    g_local,
    k_split,
    n_tokens,
    topk,
    ep,
    capacity,
    user_ragged_stride,
    ep_tp_z_clone,
):
    if (
        ep_tp_size != 2
        or has_bias
        or weight_format != WeightFormat.FP8
        or g_local <= 0
        or (g_local & (g_local - 1)) != 0
        or ep < 4
        or ep % 4 != 0
    ):
        return False
    if ep_tp_z_clone is None:
        if ep != 16 or n_tokens < TP_PAIR_Z_CLONE_MIN_TOTAL:
            return False
        want = True
    else:
        want = bool(ep_tp_z_clone)
    if not want:
        return False
    clone_stride = ragged_stride_bound(
        n_tokens, topk, ep * (g_local + k_split + 1), capacity
    )
    return user_ragged_stride is None or user_ragged_stride >= clone_stride


def _select_z_clone_plan(
    rows_by_dest,
    counts_chip,
    *,
    ep: int,
    n_chips: int,
    g_local: int,
    capacity: int,
    z_min_total: int,
    z_min_remote: int,
    z_waterfill: bool,
):
    half_ep = ep // 2
    half_chips = n_chips // 2
    counts_h0 = rows_by_dest[:half_ep].sum(axis=0).reshape(n_chips, 2 * g_local)
    counts_h1 = rows_by_dest[half_ep:].sum(axis=0).reshape(n_chips, 2 * g_local)
    chip_host = jnp.arange(n_chips, dtype=jnp.int32) >= jnp.int32(half_chips)
    remote_counts = jnp.where(chip_host[:, None], counts_h0, counts_h1)
    z_hot_phys = jnp.argmax(remote_counts, axis=1).astype(jnp.int32)
    z_hot_tot = jnp.take_along_axis(counts_chip, z_hot_phys[:, None], axis=1)[:, 0]
    chip_base_1d = jnp.arange(n_chips, dtype=jnp.int32) * jnp.int32(2 * g_local)
    hot_global = chip_base_1d + z_hot_phys
    per_rank_hot = jnp.take(rows_by_dest, hot_global, axis=1)

    # Avoid activating z_clone when 0 remote rows move: z_partner would skip an
    # empty slot_z visit and leave z_recv_sems unwaited.
    min_rem = max(int(z_min_remote), 1)
    if z_waterfill:
        # Avoid offloading all remote-host ranks blindly: bidirectional Z-clones
        # or pushes onto a heavier Z-partner thrash the inter-host link and raise
        # lockstep tiles, whereas water-filling remote ranks minimizes max(c, c^4).
        ordered_rem = jnp.where(
            chip_host[:, None],
            per_rank_hot[:half_ep, :].T,
            per_rank_hot[half_ep:, :].T,
        )
        opts = jnp.concatenate(
            [jnp.zeros((n_chips, 1), dtype=jnp.int32), jnp.cumsum(ordered_rem, axis=1)],
            axis=1,
        )
        chip_tot = counts_chip.sum(axis=1)
        z_partner_idx = lax.bitwise_xor(
            jnp.arange(n_chips, dtype=jnp.int32), jnp.int32(half_chips)
        )
        partner_tot = jnp.take(chip_tot, z_partner_idx, axis=0)
        gap = chip_tot - partner_tot
        visit_pen = int(round(TP_PAIR_VISIT_FLOOR_TILES * capacity))
        k_idx = jnp.arange(half_ep + 1, dtype=jnp.int32)[None, :]
        pen = jnp.where(k_idx == 0, jnp.int32(0), jnp.int32(visit_pen))
        valid_opt = (k_idx == 0) | (
            (opts >= jnp.int32(min_rem))
            & (z_hot_tot[:, None] >= jnp.int32(z_min_total))
            & (gap[:, None] >= jnp.int32(z_min_total))
        )
        obj = jnp.where(
            valid_opt,
            jnp.maximum(chip_tot[:, None] - opts, partner_tot[:, None] + opts + pen),
            jnp.int32(1 << 29),
        )
        best_k = jnp.argmin(obj, axis=1).astype(jnp.int32)
    else:
        z_hot_rem = jnp.take_along_axis(remote_counts, z_hot_phys[:, None], axis=1)[
            :, 0
        ]
        legacy_active = (z_hot_tot >= jnp.int32(z_min_total)) & (
            z_hot_rem >= jnp.int32(min_rem)
        )
        best_k = jnp.where(legacy_active, jnp.int32(half_ep), jnp.int32(0))

    z_active = best_k > jnp.int32(0)
    z_push_phys_by_chip = jnp.where(z_active, z_hot_phys, jnp.int32(-1))
    phys_idx_z = jnp.arange(2 * g_local, dtype=jnp.int32)[None, :]
    is_z_hot = (phys_idx_z == z_hot_phys[:, None]) & z_active[:, None]

    src_ranks = jnp.arange(ep, dtype=jnp.int32)
    src_is_h1 = src_ranks >= jnp.int32(half_ep)
    if (half_ep & (half_ep - 1)) == 0:
        src_rem_ord = jnp.bitwise_and(src_ranks, jnp.int32(half_ep - 1))
    else:
        src_rem_ord = src_ranks % jnp.int32(half_ep)
    take_rc = (src_is_h1[:, None] != chip_host[None, :]) & (
        src_rem_ord[:, None] < best_k[None, :]
    )
    z_moved_tot = jnp.sum(per_rank_hot * take_rc.astype(jnp.int32), axis=0)
    counts_for_split = jnp.where(
        is_z_hot, counts_chip - z_moved_tot[:, None], counts_chip
    )
    return (
        chip_host,
        z_active,
        z_hot_phys,
        z_push_phys_by_chip,
        is_z_hot,
        best_k,
        src_is_h1,
        src_rem_ord,
        take_rc,
        counts_for_split,
    )


def _row_hist_by_core_slot(ids_2d, *, n_cores, g_plan):
    """Per-row histogram of planned ids -> [rows, n_cores * g_plan].

    An id counts in bin `id` when it lies in [0, n_cores * g_plan).
    """
    core = jnp.floor_divide(ids_2d, jnp.int32(g_plan))
    slot = ids_2d - core * jnp.int32(g_plan)
    # Avoid a [rows, e_plan, pairs] one-hot compare-and-reduce: build the two
    # small one-hots class-major (pairs on lanes) and contract them on the MXU.
    core_oh = (
        core[:, None, :] == jnp.arange(n_cores, dtype=jnp.int32)[None, :, None]
    ).astype(jnp.bfloat16)
    slot_oh = (
        slot[:, None, :] == jnp.arange(g_plan, dtype=jnp.int32)[None, :, None]
    ).astype(jnp.bfloat16)
    counts = jnp.einsum(
        "rce,rse->rcs", core_oh, slot_oh, preferred_element_type=jnp.float32
    )
    return counts.astype(jnp.int32).reshape(ids_2d.shape[0], n_cores * g_plan)


def rebalance_ep_tp_pair_experts(
    pairs_g,
    local_blocks,
    block_hist,
    rows_by_dest,
    *,
    ep,
    ep_tp_size,
    g_local,
    e_total,
    k_split=0,
    capacity=128,
    me=None,
    t_local=None,
    topk=None,
    z_clone=False,
    z_min_total=TP_PAIR_Z_CLONE_MIN_TOTAL,
    z_min_remote=TP_PAIR_Z_CLONE_MIN_REMOTE,
    z_waterfill=True,
):
    """Balance paired experts and optionally split or Z-clone hot experts."""
    g_shift = g_local.bit_length() - 1
    chip_shift = g_shift + 1
    n_chips = ep // ep_tp_size
    counts_chip = rows_by_dest.sum(axis=0).reshape(n_chips, 2 * g_local)

    if k_split <= 0 and not z_clone:
        flip = _ep_tp_pair_flip_mask(counts_chip.reshape(n_chips, 2, g_local))
        words = _pack_flip_words(flip)
        n_words = words.shape[0]

        def _remap_ids(ids):
            # Avoid 1D dynamic LUT gather over T*topk pairs: XLA offloads it to
            # SparseCore and adds ~438us/layer overhead on uniform routing.
            pair_idx = jnp.bitwise_or(
                jnp.left_shift(jnp.right_shift(ids, chip_shift), g_shift),
                jnp.bitwise_and(ids, jnp.int32(g_local - 1)),
            )
            word_idx = jnp.right_shift(pair_idx, 5)
            bit_idx = jnp.bitwise_and(pair_idx, jnp.int32(31)).astype(jnp.uint32)
            w = words[0]
            for wi in range(1, n_words):
                w = jnp.where(word_idx == jnp.int32(wi), words[wi], w)
            bit = jnp.bitwise_and(jnp.right_shift(w, bit_idx), jnp.uint32(1))
            do_flip = jnp.where(
                (ids >= jnp.int32(0)) & (ids < jnp.int32(e_total)),
                bit.astype(jnp.int32),
                jnp.int32(0),
            )
            return jnp.bitwise_xor(ids, jnp.left_shift(do_flip, g_shift))

        def _swap_hist(h):
            h4 = h.reshape(h.shape[0], n_chips, 2, g_local)
            return jnp.where(flip[None, :, None, :], h4[:, :, ::-1, :], h4).reshape(
                h.shape
            )

        base_phys = jnp.broadcast_to(
            jnp.arange(2 * g_local, dtype=jnp.int32).reshape(1, 2, g_local),
            (n_chips, 2, g_local),
        )
        tp_slot_to_local_phys = jnp.where(
            flip[:, None, :], base_phys[:, ::-1, :], base_phys
        ).reshape(n_chips, 2 * g_local)
        return (
            _remap_ids(pairs_g),
            _remap_ids(local_blocks),
            _swap_hist(block_hist),
            _swap_hist(rows_by_dest),
            tp_slot_to_local_phys,
        )

    assert me is not None and t_local is not None and topk is not None
    half_chips = n_chips // 2
    g_plan = g_local + k_split + int(z_clone)
    slot_z = g_local + k_split
    e_plan = ep * g_plan
    z_phys_id = 2 * g_local

    if z_clone:
        (
            chip_host,
            z_active,
            z_hot_phys,
            z_push_phys_by_chip,
            _,
            best_k,
            src_is_h1,
            src_rem_ord,
            take_rc,
            counts_for_split,
        ) = _select_z_clone_plan(
            rows_by_dest,
            counts_chip,
            ep=ep,
            n_chips=n_chips,
            g_local=g_local,
            capacity=capacity,
            z_min_total=z_min_total,
            z_min_remote=z_min_remote,
            z_waterfill=z_waterfill,
        )
    else:
        chip_host = None
        z_active = None
        z_hot_phys = None
        z_push_phys_by_chip = None
        best_k = None
        src_is_h1 = None
        src_rem_ord = None
        take_rc = None
        counts_for_split = counts_chip

    if k_split > 0:
        split_phys, split_active, _, flip = _choose_ep_tp_row_splits(
            counts_for_split, g_local=g_local, k_split=k_split, capacity=capacity
        )
    else:
        flip = _ep_tp_pair_flip_mask(counts_for_split.reshape(n_chips, 2, g_local))
        split_phys = jnp.zeros((n_chips, 0), dtype=jnp.int32)
        split_active = jnp.zeros((n_chips, 0), dtype=jnp.bool_)

    words = _pack_flip_words(flip)
    n_words = words.shape[0]
    chip_base_1d = jnp.arange(n_chips, dtype=jnp.int32) * jnp.int32(2 * g_local)
    split_match = jnp.where(
        split_active, chip_base_1d[:, None] + split_phys, jnp.int32(-1)
    )
    use_rank_take_mask = bool(z_clone and n_chips <= 32)
    if z_clone:
        assert (
            chip_host is not None
            and z_active is not None
            and z_hot_phys is not None
            and best_k is not None
            and take_rc is not None
        )
        z_match = jnp.where(z_active, chip_base_1d + z_hot_phys, jnp.int32(-1))
        if use_rank_take_mask:
            # Avoid broadcasting [ep, t_local * topk] host and rank-ordinal
            # tables: packing take_rc into one uint32 bitmask per rank tests
            # chip membership in place.
            take_mask_by_rank = jnp.sum(
                jnp.left_shift(
                    take_rc.astype(jnp.uint32),
                    jnp.arange(n_chips, dtype=jnp.uint32)[None, :],
                ),
                axis=1,
                keepdims=True,
                dtype=jnp.uint32,
            )
        else:
            take_mask_by_rank = None
    else:
        z_match = None
        take_mask_by_rank = None

    def _remap_with_splits(ids, tok_par, tok_host, tok_rem_ord):
        valid = (ids >= jnp.int32(0)) & (ids < jnp.int32(e_total))
        pair_idx = jnp.bitwise_or(
            jnp.left_shift(jnp.right_shift(ids, chip_shift), g_shift),
            jnp.bitwise_and(ids, jnp.int32(g_local - 1)),
        )
        word_idx = jnp.right_shift(pair_idx, 5)
        bit_idx = jnp.bitwise_and(pair_idx, jnp.int32(31)).astype(jnp.uint32)
        w = words[0]
        for wi in range(1, n_words):
            w = jnp.where(word_idx == jnp.int32(wi), words[wi], w)
        bit = jnp.bitwise_and(jnp.right_shift(w, bit_idx), jnp.uint32(1))
        do_flip = jnp.where(valid, bit.astype(jnp.int32), jnp.int32(0))
        swapped = jnp.bitwise_xor(ids, jnp.left_shift(do_flip, g_shift))
        core = jnp.right_shift(swapped, g_shift)
        slot = jnp.bitwise_and(swapped, jnp.int32(g_local - 1))
        log_id = core * jnp.int32(g_plan) + slot
        ch = jnp.right_shift(ids, chip_shift)
        if z_clone:
            z_m = _select_by_chip(ch, z_match, n_chips)
            if take_mask_by_rank is not None:
                ch_2d = jnp.clip(ch, 0, n_chips - 1).reshape(ep, t_local * topk)
                take_z = (
                    jnp.bitwise_and(
                        jnp.right_shift(take_mask_by_rank, ch_2d.astype(jnp.uint32)),
                        jnp.uint32(1),
                    )
                    != jnp.uint32(0)
                ).reshape(ch.shape)
                hit_z = valid & (ids == z_m) & take_z
            else:
                ch_host = ch >= jnp.int32(half_chips)
                k_take = _select_by_chip(ch, best_k, n_chips)
                hit_z = (
                    valid
                    & (ids == z_m)
                    & (tok_host != ch_host)
                    & (tok_rem_ord < k_take)
                )
        else:
            hit_z = None
        for k in range(k_split):
            s_k = _select_by_chip(ch, split_match[:, k], n_chips)
            hit_k = valid & (ids == s_k)
            if hit_z is not None:
                hit_k = hit_k & (~hit_z)
            dst_k = (jnp.left_shift(ch, 1) + tok_par) * jnp.int32(g_plan) + jnp.int32(
                g_local + k
            )
            log_id = jnp.where(hit_k, dst_k, log_id)
        if hit_z is not None:
            ch_z = lax.bitwise_xor(ch, jnp.int32(half_chips))
            dst_z = (jnp.left_shift(ch_z, 1) + tok_par) * jnp.int32(g_plan) + jnp.int32(
                slot_z
            )
            log_id = jnp.where(hit_z, dst_z, log_id)
        # Avoid leaving masked rows at e_total: e_total is now a live virtual
        # expert slot, so shift non-negative out-of-range sentinels to e_plan.
        sentinel = jnp.where(ids >= jnp.int32(e_total), jnp.int32(e_plan), ids)
        return jnp.where(valid, log_id, sentinel)

    n_tokens = pairs_g.size // topk
    global_par = jnp.broadcast_to(
        (jnp.arange(n_tokens, dtype=jnp.int32) & jnp.int32(1))[:, None],
        (n_tokens, topk),
    ).reshape(pairs_g.shape)
    if z_clone and not use_rank_take_mask:
        assert src_is_h1 is not None and src_rem_ord is not None
        global_host = jnp.broadcast_to(
            src_is_h1[:, None], (ep, t_local * topk)
        ).reshape(pairs_g.shape)
        global_rem_ord = jnp.broadcast_to(
            src_rem_ord[:, None], (ep, t_local * topk)
        ).reshape(pairs_g.shape)
    else:
        global_host = None
        global_rem_ord = None
    new_pairs = _remap_with_splits(pairs_g, global_par, global_host, global_rem_ord)
    # NOTE(cross-boundary-contract): fused_ep_moe_v2.local_fn all-gathers
    # local_blocks.reshape(-1) in rank order, so pairs_g[me] is this shard's
    # pairs and its remap is new_local. Avoid remapping local_blocks a second
    # time: it repeats the per-pair select chains (~30 us/layer on TPU).
    new_local = lax.dynamic_index_in_dim(
        new_pairs.reshape(ep, t_local * topk), me, axis=0, keepdims=False
    ).reshape(local_blocks.shape)

    # Avoid re-deriving the histograms from block_hist / rows_by_dest plus
    # per-chip masked counts of the moved pairs: each (class, chip, parity)
    # count is a full masked reduction, and on [t_local, topk]-shaped masks the
    # topk-wide minor axis pads to full lane tiles. Every valid pair lands in
    # exactly one planned id and masked pairs stay outside [0, e_plan), so
    # direct histograms of the remapped ids are the same counts.
    new_rbd = _row_hist_by_core_slot(
        new_pairs.reshape(ep, t_local * topk), n_cores=ep, g_plan=g_plan
    )
    new_bh = pair_block_hist(new_local, e_plan)

    base_phys = jnp.broadcast_to(
        jnp.arange(2 * g_local, dtype=jnp.int32).reshape(1, 2, g_local),
        (n_chips, 2, g_local),
    )
    orig_phys = jnp.where(flip[:, None, :], base_phys[:, ::-1, :], base_phys)
    phys_parts = [orig_phys]
    if k_split > 0:
        phys_parts.append(
            jnp.broadcast_to(split_phys[:, None, :], (n_chips, 2, k_split))
        )
    if z_clone:
        phys_parts.append(jnp.full((n_chips, 2, 1), z_phys_id, dtype=jnp.int32))
    tp_slot_to_local_phys = jnp.concatenate(phys_parts, axis=2).reshape(
        n_chips, 2 * g_plan
    )
    if z_clone:
        return (
            new_pairs,
            new_local,
            new_bh,
            new_rbd,
            tp_slot_to_local_phys,
            z_push_phys_by_chip,
        )
    return new_pairs, new_local, new_bh, new_rbd, tp_slot_to_local_phys


def fused_ep_moe_v2(
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
    wire_dtype=None,
    ep_tp_size=1,
    ep_tp_row_splits=None,
    ep_tp_z_clone=None,
    ep_tp_z_clone_delay=TP_PAIR_Z_CLONE_DELAY_STEPS,
    ep_tp_z_clone_min_total=TP_PAIR_Z_CLONE_MIN_TOTAL,
    ep_tp_z_clone_min_remote=TP_PAIR_Z_CLONE_MIN_REMOTE,
):
    """Run one MoE layer through the fused expert-parallel kernel.

    x [tokens, hidden], w1 [experts, hidden, 2 * inter], w2 [experts,
    inter, hidden] and gating [tokens, experts], all sharded over the mesh
    axis. Returns [tokens, hidden], sharded like x.

    weight_format names one of the kernel's accepted weight forms and
    decides everything that follows from it. w1_scale and w2_scale are per
    output channel for the per-channel forms (fp8 e4m3, int8), [experts,
    2 * inter] and [experts, hidden], and per contraction block for fp4
    e2m1, [experts, blocks, 2 * inter] and [experts, blocks, hidden]; on an
    unquantized weight (bf16) there are no scales and both must be None.
    Both are taken in the kernel's own operand layout, so a table shaped
    for a loader rather than for the kernel is refused. The token rows are
    quantized to fp8 only where the format's matmuls take fp8 rows: an
    unquantized weight gets unquantized activations, with no quantize and
    dequantize pair inserted around a model that never asked for one.

    w1_bias [experts, 1, 2 * inter] and w2_bias [experts, 1, hidden] are
    optional and independent. The gate and up halves are added to the first
    matmul's post-scale accumulator before the activation; the down bias is
    added to the second matmul's post-scale row before it is quantized for
    the wire. Under expert parallelism the second matmul is sharded on the
    expert axis rather than on its contraction, so a routed row's whole down
    projection is produced on one shard and the bias enters it exactly once;
    the combine then weights it by that row's router weight, which is the
    per-selected-expert weighting the sum wants.

    block and ragged_stride default to the values this function derives;
    an explicitly passed one still has to divide tokens // ep * topk and to
    reach the no-drop bound respectively. act_fn selects the fused FFN
    activation from the kernel's ACT_FNS and is a build-time constant.

    sharded_plan computes the same kernel operands from the two complementary
    1/ep slices their consumers read, plus one small expert-count all-gather.
    It changes routing-plan construction only, never the core kernel.

    token_replica_groups lists ordered groups of device-mesh indices whose
    x and gating rows are identical (for example, a TP group). Each member
    routes and computes a disjoint slice, then gathers the combined outputs
    within its group to restore the input row order. Expert sharding stays
    unchanged. None means singleton groups with no token replication.

    wire_dtype=jnp.bfloat16 carries the computed rows back to their token
    owners in bf16 instead of the format's wire (row-quantized fp8 for fp8
    weights); None keeps the format's. See `host.weight_form`.

    mesh_ep_ranks names the EP rank at every device-mesh index. A non-identity
    order relabels only the selected expert ids after top-k, keeping the full
    router-logit tensor in its original order.

    hash_topk_ids [tokens, topk] i32 replaces the router's selection with
    expert ids the caller already chose -- DeepSeek-V4's bottom layers look
    theirs up in a static table indexed by token id rather than scoring the
    gate. The gate logits are still required and still scored: the selection
    comes from the table but the WEIGHTS are the scores at the selected ids,
    which is what the reference hash router does. Everything downstream of
    selection -- renormalization, the routed scale, the empty-row mask,
    dispatch, the matmuls and the combine -- is the top-k path unchanged.
    """
    form = weight_form(weight_format)
    rhs_qb = QB4 if rhs_qb is None else int(rhs_qb)
    # The weights themselves, against the format they were named as. This
    # validated only that the SCALES matched, which was largely
    # self-limiting while fp8 and fp4 were the whole table: four-bit weights
    # stream as packed u32 words, so a wrong format name died on the
    # ref-level bitcast or on a shape mismatch soon after. int8 and bf16
    # declare IDENTICAL slab shapes and differ only in element size, one
    # byte against two, so naming one and passing the other is a half-slab
    # read with correct-looking shapes all the way down. This is the
    # documented public entry, so the check belongs where the operands come
    # in rather than only in the serving adapter above it.
    for name, w in (("w1", w1), ("w2", w2)):
        actual = weight_format_of_dtype(w.dtype)
        if actual != weight_format:
            raise ValueError(
                f"weight format {weight_format!r} takes "
                f"{jnp.dtype(form.weight_dtype).name} expert weights and "
                f"{name} is {jnp.dtype(w.dtype).name}"
                f"{f', which is the {actual!r} form' if actual else ''}"
            )
    if form.has_scales != (w1_scale is not None):
        raise ValueError(
            f"weight format {weight_format!r} carries "
            f"{form.scale_layout} weight scales and w1_scale is "
            f"{'present' if w1_scale is not None else 'absent'}"
        )
    if (w1_scale is None) != (w2_scale is None):
        raise ValueError("both weight scales are supplied together or neither is")
    ep_tp_size = int(ep_tp_size)
    if ep_tp_size < 1:
        raise ValueError(f"ep_tp_size must be >= 1; got {ep_tp_size}")
    if ep_tp_size > 1 and weight_format != WeightFormat.FP8:
        raise NotImplementedError(
            f"ep_tp_size > 1 supports FP8 weights only; got {weight_format!r}"
        )
    T, hidden = x.shape
    w1_experts = w1.shape[0]
    if w1_experts % ep_tp_size:
        raise ValueError(
            f"w1 expert axis {w1_experts} is not divisible by ep_tp_size={ep_tp_size}"
        )
    e_total = w1_experts // ep_tp_size
    inter_slice = w1.shape[2] // 2
    inter = inter_slice * ep_tp_size
    if ep_tp_size > 1 and w2.shape != (w1_experts, inter_slice, hidden):
        raise ValueError(
            f"w2 layout {tuple(w2.shape)} is not {(w1_experts, inter_slice, hidden)}"
        )
    # Where the weight scales' layout is fixed: [E, N] per output channel,
    # [E, blocks, N] per contraction block, which is what the kernel's own
    # operands are. The scales are constants of the weights, so whatever
    # prepares the weights gives them that shape once; a reshape here would
    # instead stand between the parameter and the kernel call on every call,
    # and it is a layout change rather than free. A table carrying a
    # loader's singleton axes is refused rather than reshaped.
    if form.has_scales:
        ndim = 3 if form.scale_layout == "per_contraction_block" else 2
        for name, s, n in (
            ("w1_scale", w1_scale, 2 * inter_slice),
            ("w2_scale", w2_scale, hidden),
        ):
            if s.ndim != ndim or s.shape[0] != w1_experts or s.shape[-1] != n:
                want = (w1_experts, "blocks", n) if ndim == 3 else (w1_experts, n)
                raise ValueError(
                    f"{name} layout {tuple(s.shape)} is not the "
                    f"{form.scale_layout} form {want} the kernel "
                    f"takes"
                )
    (ax,) = mesh.axis_names
    ep = mesh.shape[ax]
    if ep % ep_tp_size:
        raise ValueError(f"mesh size {ep} is not divisible by ep_tp_size={ep_tp_size}")
    if ep_tp_size > 1 and mesh_ep_ranks is not None:
        raise ValueError("ep_tp_size > 1 requires a rank-ordered EP mesh")
    if mesh_ep_ranks is not None:
        mesh_ep_ranks = tuple(int(rank) for rank in mesh_ep_ranks)
        if tuple(sorted(mesh_ep_ranks)) != tuple(range(ep)):
            raise ValueError(
                "mesh_ep_ranks must be a permutation of "
                f"range({ep}); got {mesh_ep_ranks}"
            )
    g_local = e_total // ep
    if T % ep or gating.shape != (T, e_total):
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
    has_w1_bias = w1_bias is not None
    has_w2_bias = w2_bias is not None
    if ep_tp_size > 1 and (has_w1_bias or has_w2_bias):
        raise NotImplementedError("ep_tp_size > 1 does not support expert biases")
    k_split = _resolve_ep_tp_row_splits(
        ep_tp_size=ep_tp_size,
        has_bias=has_w1_bias or has_w2_bias,
        g_local=g_local,
        n_tokens=T,
        topk=topk,
        ep=ep,
        capacity=capacity,
        user_ragged_stride=ragged_stride,
        ep_tp_row_splits=ep_tp_row_splits,
    )
    use_z_clone = _resolve_ep_tp_z_clone(
        ep_tp_size=ep_tp_size,
        has_bias=has_w1_bias or has_w2_bias,
        weight_format=weight_format,
        g_local=g_local,
        k_split=k_split,
        n_tokens=T,
        topk=topk,
        ep=ep,
        capacity=capacity,
        user_ragged_stride=ragged_stride,
        ep_tp_z_clone=ep_tp_z_clone,
    )
    z_clone_delay = max(int(ep_tp_z_clone_delay), 0)
    z_min_total = max(int(ep_tp_z_clone_min_total), 1)
    z_min_remote = max(int(ep_tp_z_clone_min_remote), 1)
    g_plan = g_local + k_split + int(use_z_clone)
    e_plan = ep * g_plan
    stride_bound = ragged_stride_bound(T, topk, e_plan, capacity)
    if ragged_stride is None:
        ragged_stride = stride_bound
    if block is None:
        block = routing_block(t_local, topk)
    if ragged_stride % capacity != 0:
        raise ValueError(
            f"ragged_stride must be a multiple of capacity {capacity}; got "
            f"{ragged_stride}. Pass {stride_bound}."
        )
    if ragged_stride < stride_bound:
        raise ValueError(
            f"ragged_stride {ragged_stride} is below the no-drop bound "
            f"{stride_bound}, so one shard's slab could bleed into the "
            f"next. Pass {stride_bound} or more."
        )
    has_score_bias = score_bias is not None
    has_hash_ids = hash_topk_ids is not None
    if scoring_fn not in SCORING_FNS:
        raise ValueError(f"scoring_fn must be one of {SCORING_FNS}; got {scoring_fn!r}")
    if has_score_bias and score_bias.shape != (e_total,):
        raise ValueError(
            f"score_bias must be one value per global expert, [{e_total}]; "
            f"got {tuple(score_bias.shape)}"
        )
    if has_hash_ids:
        if hash_topk_ids.shape != (x.shape[0], topk):
            raise ValueError(
                f"hash_topk_ids must be one expert id per token slot, "
                f"[{x.shape[0]}, {topk}]; got {tuple(hash_topk_ids.shape)}"
            )
        if not jnp.issubdtype(hash_topk_ids.dtype, jnp.integer):
            raise ValueError(
                "hash_topk_ids are expert ids and must be an integer dtype; "
                f"got {jnp.dtype(hash_topk_ids.dtype).name}"
            )
    kfn = build_fused_ep_moe_kernel(
        g_local=g_plan,
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
        **({} if wire_dtype is None else {"wire_dtype": jnp.dtype(wire_dtype)}),
        **({} if ep_tp_size == 1 else {"ep_tp_size": ep_tp_size}),
        **({"ep_tp_z_clone": True} if use_z_clone else {}),
    )

    def local_fn(x_l, rank_l, w1_l, w2_l, *scales_gating_biases):
        """One shard's half of the layer: route, dispatch, kernel, combine."""
        # The scales come before the gating logits and the biases after
        # them, which is the operand order the scaled formats have always
        # had: a format that supplies no scales drops them out of the middle
        # rather than moving anything that stayed.
        operands = iter(scales_gating_biases)
        w1s_l = next(operands) if form.has_scales else None
        w2s_l = next(operands) if form.has_scales else None
        gate_l = next(operands)
        hash_l = next(operands) if has_hash_ids else None
        score_bias_l = next(operands) if has_score_bias else None
        # This shard's index arrives as data, not from `lax.axis_index`: that
        # lowers to `partition-id`, which the SPMD partitioner rejects when
        # torch_tpu recompiles this module through its jax.export bridge.
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
            # The rows go on the wire as they arrived. There is no scale to
            # carry, so nothing downstream builds or ships one.
            q_l, row_scale_l = rows_bf16, None
        q_g = lax.all_gather(
            q_l.reshape(t_local, hidden // HIDDEN_LANE_BLOCK, HIDDEN_LANE_BLOCK),
            ax,
            axis=0,
            tiled=True,
        )

        scores = _apply_scoring_fn(gate_l, scoring_fn)
        select_rows = MAX_ROUTING_BLOCK
        while t_local % select_rows:
            select_rows //= 2
        if hash_l is not None:
            # The selection was made by the caller's table, so the selector
            # does not run at all; the weights are still the scores at the
            # selected ids, which is the only thing the selector would have
            # contributed here.
            topk_idx = hash_l.astype(jnp.int32)
            topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1)
        elif score_bias_l is None:
            topk_weights, topk_idx = pallas_select(
                scores, topk=topk, block_rows=select_rows
            )
        else:
            # Selection is based on (scores + score_bias_l); the weights come
            # from the scores.
            topk_weights, topk_idx = pallas_select(
                scores + score_bias_l.astype(jnp.float32)[None, :],
                topk=topk,
                block_rows=select_rows,
                weight_scores=scores,
            )
        topk_idx = _relabel_expert_ids_to_mesh_order(
            topk_idx, g_local=g_local, mesh_ep_ranks=mesh_ep_ranks
        )
        # A row of scores carrying no real value routes nowhere. It was
        # dropped by arithmetic accident: the selector gives every slot of
        # such a row a large negative sentinel, and the renormalization below
        # divided by their sum, which OVERFLOWED float32 to -inf and so
        # returned exactly zero. That outcome is a property of the
        # accumulation dtype and the association order, not of the design --
        # a wider accumulator, a reassociating compiler pass, or a different
        # renormalization would silently turn such a row from dropped into
        # routed at full weight to expert zero. Mask it explicitly instead.
        # gate_l is checked too: under sigmoid/sqrtsoftplus a -inf logit scores
        # a finite 0, so replica padding would otherwise pass the score check.
        row_routes = _row_routes(scores, gate_l)
        if renormalize:
            denom = jnp.maximum(topk_weights.sum(axis=-1, keepdims=True), 1e-20)
            topk_weights = topk_weights / jnp.where(row_routes, denom, 1.0)
        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * jnp.float32(routed_scaling_factor)
        topk_weights = jnp.where(row_routes, topk_weights, 0.0)
        topk_idx = jnp.where(row_routes, topk_idx, jnp.int32(e_total))
        # Avoid deferring topk_weights 2D->1D relayout to the combine epilogue:
        # flattening in the prologue hides the retile under the dispatch gather.
        topk_weights_1d = (
            lax.optimization_barrier(topk_weights.astype(jnp.float32).reshape(-1))
            if combine_step_tokens(t_local)
            else None
        )
        # A count of the masked rows would be the other half of this: an
        # incident today is degraded output against a completely clean log.
        # It is not added here because the only in-trace channel for it,
        # jax.debug.print, makes the program refuse to lower for TPU from a
        # host, which is how this repository verifies that a change served
        # the same program. A counter belongs in a metrics channel outside
        # the trace.
        # One all-gather carries the indices, the row scales where the rows
        # were quantized, and the per-expert pair counts the routing plan
        # would otherwise exchange in a collective of its own.
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
        layout = _exchange_layout(t_local, topk, e_total, form.quantized_activations)
        exchange = lax.all_gather(
            _pack_routing_exchange(
                local_blocks.reshape(-1), scale_bits_l, block_hist.sum(axis=0), layout
            ),
            ax,
            axis=0,
            tiled=True,
        ).reshape(ep, layout[-1])
        pairs_g, scale_bits_g, rows_by_dest = _unpack_routing_exchange(
            exchange, layout, t_local, topk, e_total
        )
        tp_slot_to_local_phys = None
        z_push_phys_by_chip = None
        if (
            ep_tp_size == 2
            and not has_w1_bias
            and not has_w2_bias
            and g_local > 0
            and (g_local & (g_local - 1)) == 0
        ):
            rebal = rebalance_ep_tp_pair_experts(
                pairs_g,
                local_blocks,
                block_hist,
                rows_by_dest,
                ep=ep,
                ep_tp_size=ep_tp_size,
                g_local=g_local,
                e_total=e_total,
                k_split=k_split,
                capacity=capacity,
                me=me,
                t_local=t_local,
                topk=topk,
                z_clone=use_z_clone,
                z_min_total=z_min_total,
                z_min_remote=z_min_remote,
            )
            (
                pairs_g,
                local_blocks,
                block_hist,
                rows_by_dest,
                tp_slot_to_local_phys,
            ) = rebal[:5]
            if len(rebal) == 6:
                z_push_phys_by_chip = rebal[5]
        gathered = GatheredPairs(
            expert_blocks=pairs_g.reshape(T * topk // block, block),
            local_blocks=local_blocks,
            block_hist=block_hist,
            rows_by_dest=rows_by_dest,
            n_tokens=T,
            topk=topk,
        )

        plan_kw = dict(
            e_total=e_plan,
            ep=ep,
            t_local=t_local,
            block=block,
            tile_m=capacity,
            shard_stride=ragged_stride,
        )
        if sharded_plan:
            routing = build_routing_tables_sharded(
                None, me, gathered=gathered, **plan_kw
            )
        else:
            routing = build_routing_tables(pairs_g.reshape(T, topk), **plan_kw)
        block_tables = shard_transport_tables_in_blocks(
            routing, me, e_total=e_plan, ep=ep
        )
        row_tables = shard_push_tables_in_rows(routing, me, e_total=e_plan, ep=ep)
        # Both slab tables scatter onto this shard's own slab. The rows of
        # the other shards are dropped where they are computed rather than
        # built into a replicated slab and sliced away afterwards.
        slab_row = local_slab_rows(routing, me, shard_stride=ragged_stride)
        # The two slab scatters and the activation all-gather are offloaded to
        # the SAME SparseCore. Ordering the scatters behind the collective
        # enqueues it 129 us earlier and costs them the cover they had; that
        # trade measures +2.80% at 2048 local tokens and negative at every
        # smaller shape, so it is taken only where the routing plan in front
        # of the collective is longer than the scatters behind it. Same bits
        # either way; only the queue moves.
        if t_local >= AGQ_MIN_LOCAL_TOKENS:
            slab_row, q_g = lax.optimization_barrier((slab_row, q_g))
        token_gather = shard_token_gather(
            routing, me, shard_stride=ragged_stride, rows=slab_row
        )
        # The kernel DMA-aligns each logical expert window down by as much as
        # 127 int32 rows and always copies a full fixed window. Pad after the
        # scatter (rather than enlarging its destination) so the off-shard
        # sentinel remains out of range and cannot write a live padding row.
        token_gather = jnp.pad(token_gather, (0, token_gather_window_rows(capacity)))
        # The activation row scale, scattered onto the slab row each routed
        # pair computes on. It exists only where the rows were quantized.
        #
        # The slab is built flat and handed over in the dense lane-block
        # view, which is the same bytes in the same order and so costs
        # nothing. A column would cost a copy of the whole slab: [rows, 1]
        # is padded out to a full lane block on the way to the kernel, and
        # that copy is the largest single piece of glue above a thousand
        # rows. The kernel rebuilds the column a tile at a time instead.
        #
        # The scatter runs the length of the view rather than of the slab.
        # Those extra elements are past the slab's last row, so a pair the
        # rebase sent past the end lands on one of them instead of being
        # dropped; nothing reads them, and the slab's own rows are the same
        # either way.
        if form.quantized_activations:
            scale_bits = jnp.repeat(scale_bits_g, topk)
            scale_rows = act_scale_slab_rows(ragged_stride)
            scale_slab = lax.bitcast_convert_type(
                jnp.zeros((scale_rows * HIDDEN_LANE_BLOCK,), jnp.int32)
                .at[slab_row]
                .add(scale_bits, mode="drop"),
                jnp.float32,
            ).reshape(scale_rows, HIDDEN_LANE_BLOCK)
        else:
            scale_slab = None
        expert_rows, slab_base = shard_expert_slabs(routing, me, e_total=e_plan, ep=ep)
        recv_rows = align_up(t_local * topk + (ROWBLK - 1) * e_plan, ROWBLK)
        # The bias tables are per expert and per output channel on every
        # weight format, so they take one layout: [G, N].
        bias_it = operands
        w1b_k = next(bias_it).reshape(g_local, 2 * inter) if has_w1_bias else None
        w2b_k = next(bias_it).reshape(g_local, hidden) if has_w2_bias else None
        # The empty experts drop out of the visit list, so an empty
        # expert's weight slab never streams. How MANY are visited is a
        # row of the count table rather than its own reduction, so this
        # builder's second value is not read; it leaves the program with
        # the rest of the dead code, and its own test still sees it.
        rank_for_kernel = rank_l
        if ep_tp_size > 1:
            tp_grp = lax.div(me, jnp.int32(ep_tp_size))
            grp_rows = lax.dynamic_slice(
                routing.expert_rows_aligned,
                (tp_grp * ep_tp_size * g_plan,),
                (ep_tp_size * g_plan,),
            ).reshape(ep_tp_size, g_plan)
            if use_z_clone:
                assert z_push_phys_by_chip is not None
                slot_z = g_plan - 1
                n_chips = ep // ep_tp_size
                half_chips = n_chips // 2
                grp_rows_loc = grp_rows[:, :slot_z]
                tp_order_loc = jnp.argsort(-grp_rows_loc, axis=1, stable=True).astype(
                    jnp.int32
                )
                sorted_loc = jnp.take_along_axis(grp_rows_loc, tp_order_loc, axis=1)
                n_loc = (sorted_loc.max(axis=0) > 0).sum().astype(jnp.int32)
                z_partner_chip = lax.bitwise_xor(tp_grp, jnp.int32(half_chips))
                z_recv_active = z_push_phys_by_chip[z_partner_chip] >= jnp.int32(0)
                z_pos = jnp.minimum(jnp.int32(z_clone_delay), n_loc)
                step_idx = jnp.arange(g_plan, dtype=jnp.int32)[None, :]
                loc_pad = jnp.concatenate(
                    [tp_order_loc, jnp.full((ep_tp_size, 1), slot_z, dtype=jnp.int32)],
                    axis=1,
                )
                shifted = jnp.take_along_axis(
                    loc_pad, jnp.maximum(step_idx - jnp.int32(1), jnp.int32(0)), axis=1
                )
                tp_order_with_z = jnp.where(
                    step_idx < z_pos,
                    loc_pad,
                    jnp.where(step_idx == z_pos, jnp.int32(slot_z), shifted),
                )
                tp_order = jnp.where(z_recv_active, tp_order_with_z, loc_pad)
                z_push_phys = z_push_phys_by_chip[tp_grp]
                z_recv_step = jnp.where(z_recv_active, z_pos, jnp.int32(-1))
                z_peer_rank = lax.bitwise_xor(me, jnp.int32(ep // 2))
                rank_for_kernel = jnp.stack(
                    [me, z_push_phys, z_recv_step, z_peer_rank], axis=0
                )[None, :]
            else:
                tp_order = jnp.argsort(-grp_rows, axis=1, stable=True).astype(jnp.int32)
            sorted_rows = jnp.take_along_axis(grp_rows, tp_order, axis=1)
            active_step = (sorted_rows.max(axis=0, keepdims=True) > 0).astype(jnp.int32)
            if tp_slot_to_local_phys is not None:
                chip_phys = tp_slot_to_local_phys[tp_grp].reshape(ep_tp_size, g_plan)
                tp_phys_order = jnp.take_along_axis(chip_phys, tp_order, axis=1)
                visit = jnp.concatenate([tp_order, active_step, tp_phys_order], axis=0)
            else:
                visit = jnp.concatenate([tp_order, active_step], axis=0)
        else:
            visit_1d, _ = expert_visit_list(expert_rows, g_local)
            visit = visit_1d[None, :]
        # One pass builds every count the kernel needs, still spread over
        # the expert-parallel axis; the kernel's scalar core closes them.
        counts = shard_count_vector(routing, expert_rows, me, e_total=e_plan, ep=ep)

        # The kernel takes four of the seven transport tables and one of the
        # three push tables. The two totals are rows of `counts`, and three
        # more are the same array as one already here -- push_src is the
        # commit offset, push_len is the commit length, and the receive row
        # offset is the push destination in rows -- so each goes over once and
        # the kernel derives the rest. Every table shipped is another blocking
        # HBM->SMEM DMA at the head of the kernel, and the prefetch also has to
        # fit in 1 MB of SMEM.
        kernel_tables = (*block_tables[:3], block_tables[5], row_tables[0])
        arrivals, arrival_scales = kfn(
            kernel_tables,
            token_gather,
            expert_rows,
            slab_base,
            q_g,
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
            rank=rank_for_kernel,
        )

        # The destination's own table: one arrival row per selection slot.
        pos = (
            routing.pos
            if sharded_plan
            else lax.dynamic_slice(
                routing.arrival_row, (me * t_local, 0), (t_local, topk)
            )
        )
        mirror_pos = (
            routing.mirror_pos
            if sharded_plan
            else lax.dynamic_slice(
                routing.mirror_row, (me * t_local, 0), (t_local, topk)
            )
        )
        combined = _combine_arrivals(
            arrivals,
            arrival_scales,
            pos,
            mirror_pos,
            topk_weights,
            x_l.dtype,
            topk_weights_1d=topk_weights_1d,
        )
        # A row that routed nowhere read arrival rows nobody wrote; its weight is
        # zero, but zero times an unwritten NaN is not.
        combined = jnp.where(row_routes, combined, jnp.zeros_like(combined))
        return token_layout.restore(combined, ax)

    # The biases ride the same expert-axis sharding as the weights they
    # belong to, so no shard ever holds a bias for an expert it does not own.
    bias_args = tuple(b for b in (w1_bias, w2_bias) if b is not None)
    scale_args = (w1_scale, w2_scale) if form.has_scales else ()
    # `rank` rides the mesh axis so each shard reads its own index out of it.
    # A caller that is going through torch_tpu MUST pass it as a real operand:
    # an arange built here is a constant of the traced function, and sharding a
    # jit-internal constant compiles to `dynamic-slice(constant, partition-id)`
    # -- which XLA then refuses when the module is recompiled for one device
    # ("partitionId instruction is not supported for SPMD partitioning").
    # Under plain `jax.jit` with a mesh the constant form is fine, so the
    # default keeps the device tests and any JAX-native caller working.
    if rank is None:
        rank = jnp.arange(ep, dtype=jnp.int32).reshape(ep, 1)
    # The score bias is the one operand that is NOT sharded on the mesh axis:
    # it is indexed by GLOBAL expert id and every shard scores the full
    # expert set, so `P()` hands each shard the whole [e_total] vector.
    # Sharding it would give each shard a 1/ep slice of the experts and
    # silently bias the wrong ones.
    score_bias_args = (score_bias,) if has_score_bias else ()
    hash_args = (hash_topk_ids,) if has_hash_ids else ()
    in_specs = (
        (P(ax),) * (5 + len(scale_args) + len(hash_args))
        + (P(),) * len(score_bias_args)
        + (P(ax),) * len(bias_args)
    )
    args = (
        (x, rank, w1, w2)
        + scale_args
        + (gating,)
        + hash_args
        + score_bias_args
        + bias_args
    )
    NS = jax.sharding.NamedSharding
    args = tuple(jax.device_put(a, NS(mesh, sp)) for a, sp in zip(args, in_specs))
    # local_fn closes over config-static values only -- the per-call data
    # are the shard_map arguments -- so this key is exact.
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
        gating.dtype,
        has_w1_bias and w1_bias.dtype,
        has_w2_bias and w2_bias.dtype,
        scoring_fn,
        has_score_bias and score_bias.dtype,
        has_hash_ids and hash_topk_ids.dtype,
        float(routed_scaling_factor),
        None if wire_dtype is None else jnp.dtype(wire_dtype),
        ep_tp_size,
        k_split,
        use_z_clone,
        z_clone_delay if use_z_clone else 0,
        z_min_total if use_z_clone else 0,
        z_min_remote if use_z_clone else 0,
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
