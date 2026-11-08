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
"""The ragged all-to-all EP exchange with unscaled (bf16) experts, as an MTP
draft layer runs it: against the all-gather path it replaces, and the
capacity exchange against the full-size one."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.layers.core.ragged_a2a_moe import (
    DP_AXIS,
    TP_AXIS,
    exchange_capacity,
    ragged_a2a_moe,
)

HIDDEN = 2048
INTER = 512
TOPK = 10


def _devices():
    devices = jax.devices()
    if devices[0].platform != "tpu" or len(devices) < 8:
        pytest.skip("needs 8 TPU devices")
    return devices[:8]


def _inputs(tp_size, num_tokens, num_experts, *, skew, topk=TOPK):
    key = jax.random.key(1000 + 10 * tp_size + num_tokens + skew)
    kx, kl, k1, k2, ks = jax.random.split(key, 5)
    engines = 8 // tp_size
    x = (jax.random.normal(kx, (engines, 1, num_tokens, HIDDEN)) / 4).astype(
        jnp.bfloat16
    )
    x = jnp.broadcast_to(x, (engines, tp_size, num_tokens, HIDDEN)).reshape(-1, HIDDEN)
    logits = jax.random.normal(kl, (engines, 1, num_tokens, num_experts))
    if skew:
        local = num_experts // 8
        owner_dp = (jnp.arange(num_experts) // local) // tp_size
        logits = logits + jnp.where(owner_dp == 0, 4.0, 0.0)
    logits = jnp.broadcast_to(logits, (engines, tp_size, num_tokens, num_experts))
    weights, ids = jax.lax.top_k(
        jax.nn.softmax(logits.reshape(-1, num_experts), axis=-1), topk
    )
    w1 = (jax.random.normal(k1, (num_experts, HIDDEN, 2 * INTER)) / 32).astype(
        jnp.bfloat16
    )
    w2 = (jax.random.normal(k2, (num_experts, INTER, HIDDEN)) / 16).astype(jnp.bfloat16)
    es = jnp.arange(8, dtype=jnp.int32) * (num_experts // 8)
    shared = (jax.random.normal(ks, (8 * num_tokens, HIDDEN)) / 8).astype(jnp.bfloat16)
    return (
        x,
        w1,
        w2,
        weights.astype(jnp.bfloat16),
        ids.astype(jnp.int32),
        es,
        shared,
    )


def _allgather_local(x, w1, w2, topk_weights, topk_ids, es, **kw):
    """The default unscaled path: DP all-gather, local experts, DP
    reduce-scatter (the TP sum stays outside)."""
    gathered = [
        jax.lax.all_gather(a, DP_AXIS, tiled=True) for a in (x, topk_weights, topk_ids)
    ]
    out = fused_moe_func(
        gathered[0],
        w1,
        w2,
        None,
        None,
        None,
        None,
        gathered[1],
        gathered[2],
        experts_start=es[0],
        topk=TOPK,
        use_ep=True,
        **kw,
    )
    return jax.lax.psum_scatter(out, DP_AXIS, scatter_dimension=0, tiled=True)


def _run(mesh, dp_size, tp_size, args, **kw):
    x, w1, w2, weights, ids, es, shared = args
    stacked = [
        a.reshape(dp_size, tp_size, -1, *a.shape[1:]) for a in (x, w1, w2, weights, ids)
    ]
    es = es.reshape(dp_size, tp_size, 1)
    kw.setdefault("topk", TOPK)
    call = functools.partial(ragged_a2a_moe, mesh=mesh, **kw)
    if kw.get("owner_combine"):
        shared = shared.reshape(dp_size, tp_size, -1, HIDDEN)

        def owner_call(x, w1, w2, w, i, e, s):
            return call(x, w1, w2, None, None, w, i, e, shared=s)

        out = jax.jit(owner_call)(*stacked, es, shared)
    else:

        def plain_call(x, w1, w2, w, i, e):
            return call(x, w1, w2, None, None, w, i, e)

        out = jax.jit(plain_call)(*stacked, es)
    return np.asarray(out).reshape(-1, HIDDEN)


def _threshold(dp_size, num_tokens, topk=TOPK):
    return resolve_onehot_permute_threshold(dp_size * num_tokens * topk)


@pytest.mark.multichip
@pytest.mark.parametrize(
    "dp_size,tp_size,num_tokens,skew",
    [
        (2, 4, 24, False),  # proxy draft pass, 24 tokens per engine
        (2, 4, 96, False),  # proxy first draft pass, 96 tokens per engine
        (2, 4, 24, True),
        (4, 2, 24, False),
        (8, 1, 24, False),
    ],
)
def test_unscaled_matches_the_allgather_path(dp_size, tp_size, num_tokens, skew):
    devices = _devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    args = _inputs(tp_size, num_tokens, 64, skew=skew)
    kw = dict(
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=_threshold(dp_size, num_tokens),
    )
    spec = P((DP_AXIS, TP_AXIS))
    reference = jax.jit(
        jax.shard_map(
            functools.partial(_allgather_local, **kw),
            mesh=mesh,
            in_specs=(spec,) * 6,
            out_specs=spec,
            check_vma=False,
        )
    )(*args[:6])
    reference = np.asarray(reference, np.float32)
    assert np.abs(reference).max() > 0
    for capacity in (0.0, 2.0):
        got = _run(mesh, dp_size, tp_size, args, capacity_factor=capacity, **kw).astype(
            np.float32
        )
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.multichip
@pytest.mark.parametrize("owner", [False, True])
@pytest.mark.parametrize(
    "dp_size,tp_size,num_tokens,topk,skew",
    [
        (2, 4, 96, 2, False),  # ~24% occupancy: capacity rows and routes
        (2, 4, 96, 2, True),  # skewed onto DP position 0: overflow, full rerun
        (8, 1, 24, 2, False),
        (8, 1, 24, 10, False),  # top-10: the capacity is the full size
    ],
)
def test_unscaled_capacity_matches_the_full_exchange(
    dp_size, tp_size, num_tokens, topk, skew, owner
):
    """Bitwise the full-size exchange, with and without the owner combine."""
    devices = _devices()
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    args = _inputs(tp_size, num_tokens, 64, skew=skew, topk=topk)
    kw = dict(
        topk=topk,
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=_threshold(dp_size, num_tokens, topk),
        owner_combine=owner,
        streamed_combine=owner,
    )
    full = _run(mesh, dp_size, tp_size, args, **kw)
    assert np.abs(full.astype(np.float32)).max() > 0
    rows, _ = exchange_capacity(
        dp_size=dp_size,
        tp_size=tp_size,
        num_tokens=num_tokens,
        topk=topk,
        local_num_experts=64 // (dp_size * tp_size),
        factor=2.0,
        onehot_moe_permute_threshold=kw["onehot_moe_permute_threshold"],
        use_sparse_core=True,
        route_factor=2.0,
    )
    assert (rows < dp_size * num_tokens) == (topk == 2)
    got = _run(
        mesh, dp_size, tp_size, args, capacity_factor=2.0, route_factor=2.0, **kw
    )
    np.testing.assert_array_equal(got.view(np.uint16), full.view(np.uint16))


@pytest.mark.multichip
def test_unscaled_owner_combine_sums_over_tp_with_the_shared_partial():
    devices = _devices()
    dp_size, tp_size, num_tokens = 2, 4, 24
    mesh = Mesh(np.asarray(devices).reshape(dp_size, tp_size), (DP_AXIS, TP_AXIS))
    args = _inputs(tp_size, num_tokens, 64, skew=False)
    kw = dict(
        activation="silu",
        use_sparse_core=True,
        onehot_moe_permute_threshold=_threshold(dp_size, num_tokens),
        capacity_factor=2.0,
    )
    partial = _run(mesh, dp_size, tp_size, args, **kw).astype(np.float32)
    owner = _run(
        mesh, dp_size, tp_size, args, owner_combine=True, streamed_combine=True, **kw
    ).astype(np.float32)
    shared = np.asarray(args[6], np.float32)
    # Rank-major rows: [dp, tp, S, H]; the owner result is the TP sum of the
    # partials plus every TP rank's shared partial, on every TP rank.
    want = (partial + shared).reshape(dp_size, tp_size, num_tokens, HIDDEN).sum(axis=1)
    got = owner.reshape(dp_size, tp_size, num_tokens, HIDDEN)
    for r in range(tp_size):
        np.testing.assert_allclose(got[:, r], want, rtol=2e-2, atol=2e-2)
