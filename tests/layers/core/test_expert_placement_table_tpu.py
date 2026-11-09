# SPDX-License-Identifier: Apache-2.0
"""Ragged all-to-all EP MoE under a static expert placement: the experts' weights
moved to their slots and the routed ids replaced by the slots give the linear
placement's result, bitwise where every member's partial sums keep their order.
Replicas of experts in redundant slots give the result of the same placement
without them, bitwise."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.layers.core import expert_placement as core
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    quantize_to_native_fp4_kmajor,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.layers.core.ragged_a2a_moe import (
    DP_AXIS,
    TP_AXIS,
    _capacity_overflow,
    exchange_capacity,
    ragged_a2a_moe,
    serving_slots,
)

HIDDEN, INTER, EXPERTS, RANKS = 2048, 512, 64, 8


def _devices():
    devices = jax.devices()
    if devices[0].platform != "tpu" or len(devices) < RANKS:
        pytest.skip("needs 8 TPU devices")
    return devices[:RANKS]


def _inputs(tp_size, num_tokens, topk, skew):
    kx, kl, k1, k2 = jax.random.split(jax.random.key(num_tokens + topk), 4)
    engines = RANKS // tp_size
    x = (jax.random.normal(kx, (engines, 1, num_tokens, HIDDEN)) / 4).astype(
        jnp.bfloat16
    )
    x = jnp.broadcast_to(x, (engines, tp_size, num_tokens, HIDDEN)).reshape(-1, HIDDEN)
    logits = jax.random.normal(kl, (engines, 1, num_tokens, EXPERTS))
    if skew:
        # Most routes go to the experts of linear EP rank 0.
        logits = logits + jnp.where(jnp.arange(EXPERTS) < EXPERTS // RANKS, 3.0, 0.0)
    logits = jnp.broadcast_to(logits, (engines, tp_size, num_tokens, EXPERTS))
    weights, ids = jax.lax.top_k(jax.nn.softmax(logits.reshape(-1, EXPERTS)), topk)
    w1 = jax.random.normal(k1, (EXPERTS, HIDDEN, 2 * INTER)) / 8
    w2 = jax.random.normal(k2, (EXPERTS, INTER, HIDDEN)) / 8
    w1, s1 = quantize_to_native_fp4_kmajor(w1, 256, pack=False)
    w2, s2 = quantize_to_native_fp4_kmajor(w2, 256, pack=False)
    es = jnp.arange(RANKS, dtype=jnp.int32) * (EXPERTS // RANKS)
    return x, w1, w2, s1, s2, weights.astype(jnp.bfloat16), ids.astype(jnp.int32), es


def _placed(args, slots):
    """Weights and scales moved to their slots, routed ids replaced by slots."""
    x, w1, w2, s1, s2, weights, ids, es = args
    logical = jnp.asarray(np.argsort(slots))
    w1, w2, s1, s2 = (a[logical] for a in (w1, w2, s1, s2))
    return x, w1, w2, s1, s2, weights, jnp.asarray(slots)[ids], es


def _run(args, dp_size, tp_size, topk, num_tokens, **knobs):
    mesh = Mesh(np.asarray(_devices()).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    stacked = tuple(a.reshape(dp_size, tp_size, -1, *a.shape[1:]) for a in args)
    run = functools.partial(
        ragged_a2a_moe,
        mesh=mesh,
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(
            dp_size * num_tokens * topk
        ),
        **knobs,
    )
    return np.asarray(jax.jit(run)(*stacked))


KNOBS = {
    "full": {},
    "capacity": dict(capacity_factor=2.5, route_factor=3.0),
    "streamed": dict(capacity_factor=2.5, route_factor=3.0, streamed_combine=True),
}


@pytest.mark.multichip
@pytest.mark.parametrize("knobs", list(KNOBS))
@pytest.mark.parametrize("dp_size,tp_size", [(8, 1), (2, 4)])
def test_reordering_experts_within_ranks_is_bitwise(dp_size, tp_size, knobs):
    """Top-2: no member sums three routes of a token, so with every expert on
    its linear rank the sums keep their values in any expert order."""
    num_tokens, topk = 24, 2
    args = _inputs(tp_size, num_tokens, topk, skew=True)
    local = EXPERTS // RANKS
    slots = (np.arange(EXPERTS) // local) * local + (
        local - 1 - np.arange(EXPERTS) % local
    )
    base = _run(args, dp_size, tp_size, topk, num_tokens, **KNOBS[knobs])
    got = _run(_placed(args, slots), dp_size, tp_size, topk, num_tokens, **KNOBS[knobs])
    assert np.abs(base.astype(np.float32)).max() > 0
    np.testing.assert_array_equal(got.view(np.uint16), base.view(np.uint16))


@pytest.mark.multichip
@pytest.mark.parametrize("knobs", list(KNOBS))
@pytest.mark.parametrize("dp_size,tp_size", [(8, 1), (4, 2), (2, 4)])
def test_balanced_placement_matches_linear_up_to_sum_order(dp_size, tp_size, knobs):
    """Top-10 under skewed routing, experts moved across ranks by ``balance``:
    each token's routes regroup into other members' partial sums, so only the
    rounding of the sums may differ. Each rank returns its sum before the TP
    all-reduce, and an expert's new rank may sit at another TP position, so
    the engines' sums over their TP ranks are compared."""
    num_tokens, topk = 24, 10
    args = _inputs(tp_size, num_tokens, topk, skew=True)
    counts = np.bincount(np.asarray(args[6]).ravel(), minlength=EXPERTS)
    slots = core.balance(counts, RANKS)
    local = EXPERTS // RANKS
    assert (slots // local != np.arange(EXPERTS) // local).mean() > 0.5
    base = _run(args, dp_size, tp_size, topk, num_tokens, **KNOBS[knobs])
    got = _run(_placed(args, slots), dp_size, tp_size, topk, num_tokens, **KNOBS[knobs])
    b, g = (a.astype(np.float32).sum(axis=1) for a in (base, got))  # [dp, S, H]
    assert np.abs(b).max() > 0 and np.isfinite(g).all()
    differ = (g != b).mean()
    print(f"dp{dp_size} tp{tp_size} {knobs}: {differ:.4%} of engine sums differ")
    np.testing.assert_allclose(g, b, rtol=2e-2, atol=2e-2)


def _greedy_replicas(counts, slots, redundant_per_rank):
    """Replicas from the builder's packing, every redundant slot asked for (the
    replication greedy without the guard)."""
    replicas = np.ones(EXPERTS, np.int64)
    for _ in range(RANKS * redundant_per_rank):
        replicas[int(np.argmax(counts / replicas))] += 1
    return core.place_replicas(counts, slots, RANKS, redundant_per_rank, replicas)


def _spread_replicas(counts, slots):
    """One redundant slot per rank: the busiest expert on every rank but its
    own, which replicates the next busiest expert it does not hold. Replicas
    then serve routes on the token's owner rank and on every TP column."""
    local = EXPERTS // RANKS
    busiest = np.argsort(-counts, kind="stable")
    redundant = np.full(RANKS, busiest[0], np.int32)
    own = slots[busiest[0]] // local
    redundant[own] = next(e for e in busiest[1:] if slots[e] // local != own)
    return redundant


def _with_replicas(args, slots, redundant):
    """The placement's weights with each rank's redundant slots after its primaries
    (an empty slot holds expert 0's weights: no route reaches it), the routed
    ids as physical primaries, and the replica table on every rank."""
    x, w1, w2, s1, s2, weights, ids, _ = args
    local = EXPERTS // RANKS
    redundant_per_rank = redundant.shape[0] // RANKS
    held = np.concatenate(
        [
            np.argsort(slots).reshape(RANKS, local),
            redundant.reshape(RANKS, redundant_per_rank),
        ],
        axis=1,
    ).reshape(-1)
    pick = jnp.asarray(np.maximum(held, 0))
    w1, w2, s1, s2 = (a[pick] for a in (w1, w2, s1, s2))
    primary = core.physical_slots(slots, RANKS, redundant_per_rank)
    ids = jnp.asarray(primary.astype(np.int32))[ids]
    es = jnp.arange(RANKS, dtype=jnp.int32) * (local + redundant_per_rank)
    table = core.replica_table(slots, redundant, RANKS)
    return (x, w1, w2, s1, s2, weights, ids, es), table


def _run_owner(
    args,
    dp_size,
    tp_size,
    topk,
    num_tokens,
    *,
    shared,
    threshold,
    replicas=None,
    **knobs,
):
    mesh = Mesh(np.asarray(_devices()).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    stacked = tuple(a.reshape(dp_size, tp_size, -1, *a.shape[1:]) for a in args)
    kwargs = {"shared": shared}
    if replicas is not None:
        tiled = np.tile(replicas, (RANKS, 1))
        kwargs["replicas"] = jnp.asarray(tiled).reshape(
            dp_size, tp_size, *replicas.shape
        )
    run = functools.partial(
        ragged_a2a_moe,
        mesh=mesh,
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=threshold,
        owner_combine=True,
        sorted_plan=True,
        **knobs,
    )
    return np.asarray(jax.jit(run)(*stacked, **kwargs))


# (rows, routes) factors: a step that fits (replicas serve routes), one that
# overflows on every rank (the full-size exchange serves the primaries), and
# none (the full-size exchange only). "overflow_sc": the full-size exchange
# runs the SparseCore permute and returns partials to the senders, as it does
# at 32 chips, because the one-hot threshold sits between the capacity
# exchange's routes and the full-size exchange's.
REPLICA_CAPACITY = {
    "fits": dict(capacity_factor=1.15, route_factor=1.5),
    "overflow": dict(capacity_factor=0.5, route_factor=0.5),
    "overflow_sc": dict(capacity_factor=0.5, route_factor=0.5),
    "full": dict(capacity_factor=0.0),
}
REPLICA_OPTIONS = {
    "served": dict(row_gather=True, row_scatter=True, routing_dma=True),
    "served_tail": dict(
        row_gather=True, row_scatter=True, routing_dma=True, owner_tail=True
    ),
    "plain": {},
}
REPLICA_CASES = (
    [
        (layout, table, capacity, options, False)
        for layout in ((2, 4, 112), (4, 2, 56))
        for table in ("spread", "greedy")
        for capacity in ("fits", "overflow")
        for options in ("served", "served_tail")
    ]
    + [
        (layout, "greedy2", "fits", options, False)
        for layout in ((2, 4, 112), (4, 2, 56))
        for options in ("served", "served_tail")
    ]
    + [
        (layout, "spread", "overflow_sc", "served", False)
        for layout in ((2, 4, 112), (4, 2, 56))
    ]
    + [((2, 4, 112), "spread", "fits", "plain", False)]
    + [((4, 2, 56), "greedy", "fits", "served_tail", True)]
    + [((2, 4, 112), "spread", "full", "served", False)]
)
_REPLICA_BASE = {}


@pytest.mark.multichip
@pytest.mark.parametrize("layout,table,capacity,options,padded", REPLICA_CASES)
def test_replicas_keep_every_output_bit(layout, table, capacity, options, padded):
    """The owner-combine exchange as served (sorted plan, owner combine with the
    shared partial; row copies, routing DMA and the owner tail by case) with
    replicas in redundant slots gives, bit for bit, the same placement's result
    without them: on a step that fits, where replicas serve routes, and on a
    step that overflows, where the full-size exchange serves primaries."""
    dp_size, tp_size, num_tokens = layout
    topk = 10
    args = _inputs(tp_size, num_tokens, topk, skew=False)
    if padded:
        weights = args[5].reshape(RANKS, num_tokens, topk).at[:, -5:].set(0)
        args = args[:5] + (weights.reshape(-1, topk),) + args[6:]
    counts = np.bincount(np.asarray(args[6]).ravel(), minlength=EXPERTS)
    slots = core.balance(counts, RANKS)
    redundant = {
        "spread": lambda: _spread_replicas(counts, slots),
        "greedy": lambda: _greedy_replicas(counts, slots, 1),
        "greedy2": lambda: _greedy_replicas(counts, slots, 2),
    }[table]()
    core.validate(slots[None], ["layer"], RANKS, redundant[None])
    replicated, table = _with_replicas(args, slots, redundant)
    knobs = dict(REPLICA_CAPACITY[capacity], **REPLICA_OPTIONS[options])
    knobs["skip_padded_tokens"] = padded
    threshold = resolve_onehot_permute_threshold(dp_size * num_tokens * topk)
    if capacity == "overflow_sc":
        threshold = 1024
        assert dp_size * num_tokens * topk > threshold
    rng = np.random.default_rng(3)
    shared = jnp.asarray(
        rng.standard_normal((dp_size, tp_size, num_tokens, HIDDEN)) / 8, jnp.bfloat16
    )

    # The branch the run with replicas takes, from its serving slots.
    local = EXPERTS // RANKS + redundant.shape[0] // RANKS
    engine_ids = np.asarray(replicated[6]).reshape(dp_size, tp_size, num_tokens, topk)
    engine_ids = jnp.asarray(engine_ids[:, 0])
    serving = jax.vmap(serving_slots, in_axes=(0, None))(engine_ids, jnp.asarray(table))
    if capacity != "full":
        rows, routes = exchange_capacity(
            dp_size=dp_size,
            tp_size=tp_size,
            num_tokens=num_tokens,
            topk=topk,
            local_num_experts=EXPERTS // RANKS,
            factor=knobs["capacity_factor"],
            onehot_moe_permute_threshold=threshold,
            use_sparse_core=True,
            route_factor=knobs["route_factor"],
        )
        assert rows < dp_size * num_tokens and routes <= threshold
        overflow = _capacity_overflow(
            serving, tp_size=tp_size, local_num_experts=local, rows=rows, routes=routes
        )
        assert bool(overflow) == capacity.startswith("overflow")
    assert (np.asarray(serving) != np.asarray(engine_ids)).mean() > 0.01

    key = (layout, capacity, options, padded)  # the same slots for every table
    if key not in _REPLICA_BASE:
        _REPLICA_BASE[key] = _run_owner(
            _placed(args, slots),
            dp_size,
            tp_size,
            topk,
            num_tokens,
            shared=shared,
            threshold=threshold,
            **knobs,
        )
    base = _REPLICA_BASE[key]
    got = _run_owner(
        replicated,
        dp_size,
        tp_size,
        topk,
        num_tokens,
        shared=shared,
        threshold=threshold,
        replicas=table,
        **knobs,
    )
    assert np.abs(base.astype(np.float32)).max() > 0
    np.testing.assert_array_equal(got.view(np.uint16), base.view(np.uint16))
