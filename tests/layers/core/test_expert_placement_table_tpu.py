# SPDX-License-Identifier: Apache-2.0
"""Ragged all-to-all EP MoE under a static expert placement: the experts' weights
moved to their slots and the routed ids replaced by the slots give the linear
placement's result, bitwise where every member's partial sums keep their order."""

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
from vllm_torchtpu.layers.core.ragged_a2a_moe import DP_AXIS, TP_AXIS, ragged_a2a_moe

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
