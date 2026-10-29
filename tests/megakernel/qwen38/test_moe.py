# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The routed experts' route split and combine (``megakernel.qwen38.moe``) on
the host."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.megakernel.qwen38.moe import (  # noqa: E402
    EXPERT_ROUTES,
    PLAN_TARGETS,
    SLAB,
    combine,
    core_routes,
    held_experts,
    sorted_routes,
    top_routes,
    window_routes,
)
from vllm_torchtpu.megakernel.qwen38.reference import Dims  # noqa: E402

CORES = 8


def _cost(routes):
    """A core's load: its routes plus a fixed cost per expert it runs."""
    return routes + EXPERT_ROUTES * (routes > 0)


def _busiest_reachable(counts, copies):
    """The smallest busiest load the copies allow (bisection on the cycle)."""
    local = counts.shape[1]
    load = _cost(counts).sum(axis=1)
    cap = _cost(counts[:, :: local // copies]).sum(axis=1)

    def reachable(target):
        passed = 0
        for _ in range(3):
            for c in range(CORES):
                passed = max(0, load[c] + passed - target)
                if passed > cap[c]:
                    return False
        return True

    low, high = -(-load.sum() // CORES), load.max()
    while low < high:
        mid = (low + high) // 2
        low, high = (low, mid) if reachable(mid) else (mid + 1, high)
    return low


@pytest.mark.parametrize("seed", [0, 1])
def test_core_routes_take_every_route_once_and_balance_the_cores(seed):
    d = Dims(k_heads=4, v_heads=8)
    local = d.experts // CORES
    held = [np.asarray(held_experts(d, CORES, c)) for c in range(CORES)]
    groups = len(held[0])
    rng = np.random.default_rng(seed)
    hot = rng.random(d.experts) ** 6  # skewed: a few experts take most routes
    rows = 128
    ids = np.stack(
        [
            rng.choice(d.experts, d.top_k, replace=False, p=hot / hot.sum())
            for _ in range(CORES * rows)
        ]
    ).astype(np.int32)
    weights = rng.random(ids.shape).astype(np.float32)
    scale = rng.random(CORES * rows).astype(np.float32)
    per_core = rows * d.top_k
    # Each core sorts its own routes; every core gets all of them.
    parts = [
        sorted_routes(
            jnp.asarray(ids[c * rows : (c + 1) * rows]),
            jnp.asarray(weights[c * rows : (c + 1) * rows]),
            jnp.asarray(scale[c * rows : (c + 1) * rows]),
            d.experts,
        )
        for c in range(CORES)
    ]
    route_all, weight_all, scale_all, counts = (
        np.concatenate([np.asarray(p[i]) for p in parts]) for i in range(4)
    )
    routes = route_all.size
    split = jax.jit(core_routes, static_argnums=(1, 2, 4))

    taken = np.zeros(routes, np.int32)
    loads, costs, busiests = [], [], []
    for me in range(CORES):
        runs, sizes, busiest = jax.tree.map(
            np.asarray,
            split(jnp.asarray(counts.reshape(CORES, -1)), local, groups, me, per_core),
        )
        total = int(sizes.sum())
        at = np.asarray(window_routes(runs, jnp.int32(0), total, routes))
        route = at // per_core * per_core + route_all[at]
        taken[route] += 1
        # Group g holds routes to the expert the weights hold at g, in token
        # order; each route carries its weight and its row's scale.
        group = np.repeat(np.arange(groups), sizes)
        np.testing.assert_array_equal(ids.reshape(-1)[route], held[me][group])
        assert np.all((np.diff(route) > 0) | (np.diff(group) > 0))
        np.testing.assert_array_equal(weight_all[at], weights.reshape(-1)[route])
        np.testing.assert_array_equal(scale_all[at], scale[route // d.top_k])
        # A later window reads the same positions.
        start = total // 3
        later = window_routes(runs, jnp.int32(start), total - start, routes)
        np.testing.assert_array_equal(np.asarray(later), at[start:])
        loads.append(total)
        costs.append(_cost(sizes).sum())
        busiests.append(int(busiest))
    np.testing.assert_array_equal(taken, 1)
    assert busiests == [max(loads)] * CORES

    counts = np.bincount(ids.reshape(-1), minlength=d.experts).reshape(CORES, local)
    best = _busiest_reachable(counts, groups - local)
    load = _cost(counts).sum(1)
    resolution = (load.max() - -(-load.sum() // CORES)) // (PLAN_TARGETS - 1) + 1
    # A split expert's copy pays its fixed cost again.
    assert best <= max(costs) <= best + resolution + EXPERT_ROUTES
    assert max(costs) < load.max()


def test_top_routes_match_softmax_top_k():
    rng = np.random.default_rng(4)
    logits = jnp.asarray(rng.standard_normal((512, 512)) * 2, jnp.float32)
    logits = logits.at[3, 7].set(logits[3, 9])  # equal logits: the lower index first
    ids, weights = top_routes(logits, 10, interpret=True)
    expected, want_ids = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), 10)
    np.testing.assert_array_equal(ids, want_ids)
    np.testing.assert_allclose(
        weights, expected / expected.sum(-1, keepdims=True), atol=1e-6
    )


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("start", [0, 24])
def test_combine_adds_each_windowed_route_to_its_token(start, first):
    rng = np.random.default_rng(0)
    tokens, top_k, hidden, span = 64, 10, 2048, 40
    products = rng.standard_normal((span, hidden)).astype(np.float32)
    # A third of the routes are this core's, their sorted positions a permutation.
    local = rng.random((tokens, top_k)) < 0.3
    position = np.full((tokens, top_k), -1, np.int32)
    position[local] = rng.permutation(int(local.sum()))
    weights = rng.random((tokens, top_k)).astype(np.float32)
    partial = rng.standard_normal((tokens, hidden)).astype(np.float32)
    if first:  # the first window: the sums start from zero
        partial = np.zeros_like(partial)

    out = combine(
        jnp.asarray(products, jnp.bfloat16).reshape(span, SLAB, hidden // SLAB),
        jnp.asarray(position),
        jnp.asarray(weights),
        None if first else jnp.asarray(partial).reshape(tokens, SLAB, hidden // SLAB),
        jnp.int32(start),
        interpret=True,
    ).reshape(tokens, hidden)

    rows = np.asarray(jnp.asarray(products, jnp.bfloat16).astype(jnp.float32))

    want = partial.copy()
    for t in range(tokens):
        for k in range(top_k):
            p = position[t, k] - start
            if position[t, k] >= 0 and 0 <= p < span:
                want[t] += weights[t, k] * rows[p]
    np.testing.assert_allclose(np.asarray(out), want, rtol=1e-6, atol=1e-5)
