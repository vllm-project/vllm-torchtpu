# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EP16/TP2 BF16-reduction MoE and prepared-route overflow consumers.

The 8K metadata fast path fits local routes into 256 KiB of SMEM. Overflow
streams bounded metadata tiles instead, with a uniform decision on all ranks.
"""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from ...config import active as _config
from ..prefill_megakernel import (
    MOE_FAMILIES,
    _moe_sp_shared_rows,
    _moe_sp_tail,
    expert_family_slot,
    gmm_v2,
    ring_tables,
    sp_token_index,
    window_stripe,
)
from .indexed_ragged_gmm import indexed_gate_up
from .moe_down_valu import down_valu, down_valu_compact
from .routing_count_kernel import counting_positions
from .routing_topk_kernel import select_top16
from .shared_ring_kernel import make_shared_ring


def shared_ring_partial(latent, hidden, weights, tables, axis_name):
    fused = make_shared_ring(
        hidden.shape[0],
        budget=1 if hidden.shape[0] == 128 else 2,
    )
    return fused(
        tables,
        weights["expert_layer"],
        latent.reshape(latent.shape[0], 2, 1792),
        _moe_sp_shared_rows(hidden, axis_name),
        weights["shared_gu"],
        weights["shared_down"],
    )


def routed_experts(
    latent,
    topk_ids,
    topk_probabilities,
    expert_gate_up,
    expert_gate_up_scales,
    expert_down,
    expert_down_scales,
    *,
    first_expert,
    reduce_tables,
    expert_layer=None,
):
    """Sort local routes, consume packed rows, and reduce weighted products.

    Resident MXFP4 families use the runtime layer index. Inactive routes sort
    beyond the counted expert prefix; no routing capacity or token dropping
    is imposed. The output contains this rank's reduced token rows, in the
    ring's dtype (``moe_reduce_dtype``).
    """
    tokens, _, half = latent.shape
    input_width = 2 * half
    local_experts = expert_gate_up.shape[-3]
    if topk_ids.shape != topk_probabilities.shape or topk_ids.shape[0] != tokens:
        raise ValueError("topk ids and probabilities must have shape [T, topk]")
    if expert_gate_up.shape[-2] * 8 != input_width:
        raise ValueError("latent width must match the packed gate/up contraction")
    if expert_gate_up.shape[-1] != 2 * expert_down.shape[-2] * 8:
        raise ValueError("gate/up and down intermediate widths disagree")

    topk = topk_ids.shape[1]
    local_ids = topk_ids.astype(jnp.int32) - jnp.asarray(first_expert, jnp.int32)
    valid = (local_ids >= 0) & (local_ids < local_experts) & (topk_probabilities != 0)
    keys = jnp.where(valid, local_ids, local_experts).reshape(-1)
    pair_bits = (tokens * topk - 1).bit_length()
    if (local_experts + 1).bit_length() + pair_bits >= 31:
        raise ValueError("expert and token ids do not fit the packed sort key")
    # Sorting one packed integer gives both an expert-major order and its
    # source token without gathering a second large permutation vector.
    global_route_max = None
    if tokens == 8192:
        # Identical gathered routes give a rank-uniform fast/overflow decision
        # without another collective. All ranks must execute the same ring.
        global_keys = jnp.where(topk_probabilities != 0, topk_ids, 896).reshape(-1)
        _, global_counts = counting_positions(global_keys, 897)
        counts = jax.lax.dynamic_slice(global_counts, (first_expert,), (local_experts,))
        global_route_max = global_counts[:896].reshape(16, 56).sum(axis=1).max()
    else:
        _, all_counts = counting_positions(keys, local_experts + 1)
        counts = all_counts[:local_experts]
    packed = (keys << pair_bits) | jnp.arange(tokens * topk, dtype=jnp.int32)
    sorted_pairs = jnp.sort(packed, stable=False) & ((1 << pair_bits) - 1)
    sorted_tokens = sorted_pairs // topk
    sorted_probs = topk_probabilities.reshape(-1)[sorted_pairs]
    return routed_experts_prepared(
        latent,
        sorted_tokens,
        sorted_probs,
        counts,
        expert_gate_up,
        expert_gate_up_scales,
        expert_down,
        expert_down_scales,
        reduce_tables=reduce_tables,
        expert_layer=expert_layer,
        global_route_max=global_route_max,
    )


def routed_experts_prepared(
    latent,
    sorted_tokens,
    sorted_probs,
    counts,
    expert_gate_up,
    expert_gate_up_scales,
    expert_down,
    expert_down_scales,
    *,
    reduce_tables,
    expert_layer=None,
    global_route_max=None,
):
    """Consume prepared expert-major metadata with the existing GU/down path."""
    if expert_layer is not None:
        expert_layer = expert_family_slot(expert_gate_up.shape[0], expert_layer)
    tokens = latent.shape[0]
    # The E8M0 exponent bytes go to the GMM as they are (it converts each
    # tile in VMEM): an FP32 [E, blocks, 1, N] copy pads 8x in HBM.
    gu_scale = expert_gate_up_scales.astype(jnp.uint8)
    down_scale = expert_down_scales.astype(jnp.uint8)
    # gmm_v2 splits [gate, up] and interleaves lanes for the fused SiTU
    # activation (expert_activation).
    middle = indexed_gate_up(
        latent, sorted_tokens, expert_gate_up, counts, gu_scale, expert_layer
    )
    options = dict(
        tokens=tokens,
        layer=expert_layer,
        ring_dtype=jnp.dtype(_config().moe_reduce_dtype),
    )
    if tokens == 8192:
        return down_valu_compact(
            middle,
            expert_down,
            down_scale,
            counts,
            sorted_tokens,
            sorted_probs,
            reduce_tables,
            tile=256,
            columns=896,
            unroll=32,
            metadata_prefix_rows=32768,
            slice_input=False,
            global_route_max=global_route_max,
            **options,
        )
    return down_valu(
        middle,
        expert_down,
        down_scale,
        counts,
        sorted_tokens,
        sorted_probs,
        reduce_tables,
        tile=128,
        columns=1792,
        unroll=8,
        **options,
    )


def moe_local_routed(
    hidden_local, length, weights, *, axis_name, gather_tables, reduce_tables
):
    """Sequence-parallel routing and packed dispatch with shared-expert overlap."""
    rank = jax.lax.axis_index(axis_name)
    local_tokens, hidden_width = hidden_local.shape
    if hidden_width != 7168:
        raise ValueError("K3 sequence-parallel MoE expects [T/32,7168]")
    if weights["latent_down"].shape[-2:] != (7168, 4096):
        raise ValueError("latent projection weights must use replicated SP layout")
    local_experts = weights["expert_gu"].shape[-3]
    if local_experts != 56:
        raise ValueError("Packed prefill MoE requires EP16/TP2 expert weights")
    expert_tp = 32 * local_experts // 896
    tokens = local_tokens * 32
    logits = jax.lax.dot_general(
        hidden_local,
        weights["router"],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    raw = jax.nn.sigmoid(logits)
    local_ids, local_probs = select_top16(raw, weights["router_bias"])
    valid_rows = sp_token_index(
        rank, jnp.arange(local_tokens), window_stripe(local_tokens)
    ) < length.reshape(())
    local_probs = jnp.where(valid_rows[:, None], local_probs, 0)
    if weights["latent_down"].ndim == 3:
        latent_local = gmm_v2(
            hidden_local,
            weights["latent_down"][:, None],
            jnp.full((1,), local_tokens, jnp.int32),
            rhs_layer=weights["expert_layer"],
            maybe_quantize_lhs=False,
            zero_initialize=False,
        )
    else:
        latent_local = hidden_local @ weights["latent_down"]
    latent_all, shared = shared_ring_partial(
        latent_local[:, :3584], hidden_local, weights, gather_tables, axis_name
    )
    routes = jnp.concatenate(
        (
            local_ids.astype(jnp.int32),
            jax.lax.bitcast_convert_type(local_probs, jnp.int32),
        ),
        axis=1,
    )
    routes = jax.lax.all_gather(routes, axis_name).reshape(tokens, 32)
    ids_all = routes[:, :16]
    probs_all = jax.lax.bitcast_convert_type(routes[:, 16:], jnp.float32)
    routed = routed_experts(
        latent_all,
        ids_all,
        probs_all,
        weights["expert_gu"],
        weights["expert_gus"],
        weights["expert_down"],
        weights["expert_ds"],
        reduce_tables=reduce_tables,
        first_expert=rank // expert_tp * local_experts,
        expert_layer=weights.get("expert_layer"),
    )
    return (routed, shared)


def make_moe_fused_sp_repacked(
    mesh, *, seq_len, eps=1e-5, collective_id=42, token_ranks=None
):
    """Build the TP32 MoE layer for repacked projection weights: the local
    routed partials, their reduce-scatter ring (``moe_reduce_dtype``), then
    the latent-up and shared-expert tail on each rank's rows."""
    if (
        tuple(mesh.axis_names) != ("k3_sp",)
        or seq_len not in (4096, 8192)
        or mesh.size != 32
    ):
        raise ValueError(
            "Packed MoE requires TP32 k3_sp and T in {4096,8192} (a multiple of 2048)"
        )
    axis_name = mesh.axis_names[0]
    if token_ranks is None:
        token_ranks = tuple(range(32))
    tables = ring_tables(list(mesh.devices.flat), token_ranks)
    gather_tables = ring_tables(list(mesh.devices.flat), tuple(range(32)))

    def local_partial(hidden, prefix, weights, length):
        local_weights = jax.tree.map(lambda value: value[0], weights)
        routed, shared = moe_local_routed(
            hidden[0],
            length,
            local_weights,
            axis_name=axis_name,
            gather_tables=gather_tables,
            reduce_tables=tables,
        )
        return routed[None], shared[None]

    partial_op = jax.jit(
        jax.shard_map(
            local_partial,
            mesh=mesh,
            in_specs=(P(axis_name), P(axis_name), P(axis_name), P()),
            out_specs=(P(axis_name), P(axis_name)),
            check_vma=False,
        )
    )

    def local_tail(reduced, shared, prefix, weights, length):
        local_weights = jax.tree.map(lambda value: value[0], weights)
        return _moe_sp_tail(
            reduced[0],
            shared[0],
            prefix[0],
            length,
            local_weights,
            axis_name=axis_name,
            eps=eps,
        )[None]

    tail_op = jax.jit(
        jax.shard_map(
            local_tail,
            mesh=mesh,
            in_specs=(P(axis_name), P(axis_name), P(axis_name), P(axis_name), P()),
            out_specs=P(axis_name),
            check_vma=False,
        )
    )

    def run_ring(hidden, prefix, weights, length):
        partials, shared = partial_op(hidden, prefix, weights, length)
        reduced = partials
        return tail_op(reduced, shared, prefix, weights, length)

    return jax.jit(run_ring)


def make_indexed_moe_sp_ring(mesh, *, seq_len=8192):
    """Select a MoE layer, keeping packed expert families resident for tile DMA."""
    base = make_moe_fused_sp_repacked(mesh, seq_len=seq_len)

    def run(hidden, prefix, families, length, slot):
        selected = {
            name: (
                families[name]
                if name
                in (
                    "expert_gu",
                    "expert_down",
                    "expert_gus",
                    "expert_ds",
                    "latent_down",
                    "shared_gu",
                    "shared_down",
                )
                else jax.lax.dynamic_index_in_dim(
                    families[name],
                    slot[0],
                    axis=1,
                    keepdims=False,
                )
            )
            for name in MOE_FAMILIES
        }
        selected["expert_layer"] = jnp.broadcast_to(slot[None], (mesh.size, 1))
        return base(hidden, prefix, selected, length)

    return jax.jit(run)
