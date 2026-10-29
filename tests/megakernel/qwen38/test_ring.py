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
"""The ring order (``megakernel.qwen38.ring``) and the routed experts' ring
reduce-scatter (``moe.combine_scatter``) on the host.

The kernel runs on an eight-device CPU mesh with
XLA_FLAGS=--xla_force_host_platform_device_count=8; the interpreted kernel
blocks host threads in semaphore waits, so the host needs more cores than
devices.
"""

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel.qwen38.moe import (  # noqa: E402
    SLAB,
    chunk_routes,
    combine_scatter,
)
from vllm_torchtpu.megakernel.qwen38.ring import ring_order  # noqa: E402

AXIS, CORES = "pcp", 8
RING = (0, 1, 2, 3, 6, 7, 4, 5)


def _per_core(f, *args):
    """``f`` on each core's slice ``[i]`` of ``args`` (``[cores, ...]``)."""
    if len(jax.devices()) != CORES:
        pytest.skip("requires eight devices; CPU virtual devices are sufficient")
    mesh = Mesh(np.asarray(jax.devices()), (AXIS,))
    run = jax.shard_map(
        lambda *a: f(*(x[0] for x in a))[None],
        mesh=mesh,
        in_specs=(P(AXIS),) * len(args),
        out_specs=P(AXIS),
        check_vma=False,
    )
    placed = [
        jax.device_put(jnp.asarray(a), NamedSharding(mesh, P(AXIS))) for a in args
    ]
    return np.asarray(jax.jit(run)(*placed), np.float32)


def test_ring_order_walks_the_chip_cycle_with_both_cores_of_each_chip():
    # Mesh order: chips (0,0) (1,0) (0,1) (1,1), two cores each.
    devices = [
        SimpleNamespace(coords=(x, y, 0), core_on_chip=c)
        for y in (0, 1)
        for x in (0, 1)
        for c in (0, 1)
    ]
    assert ring_order(devices) == (0, 1, 2, 3, 6, 7, 4, 5)
    with pytest.raises(ValueError):
        ring_order(devices[:6])


@pytest.mark.parametrize("later_windows", [False, True])
def test_combine_scatter_sums_each_tokens_routes_over_the_cores(later_windows):
    rng = np.random.default_rng(1)
    rows, top_k, columns, window = 64, 2, 128, 80  # one group of four 8-row chunks
    tokens = CORES * rows
    # Each route on one core, at a place in its sorted routes; places past
    # the window belong to later windows.
    core = rng.integers(0, CORES, (tokens, top_k))
    position = np.full((CORES, tokens, top_k), -1, np.int32)
    for c in range(CORES):
        mine = core == c
        position[c][mine] = rng.permutation(int(mine.sum()))
    # Small integers and power-of-two weights: every BF16 sum is exact.
    products = rng.integers(-3, 4, (CORES, window, SLAB, columns)).astype(np.float32)
    weights = 2.0 ** rng.integers(-2, 1, (tokens, top_k)).astype(np.float32)
    extra = rng.integers(-3, 4, (CORES, tokens, SLAB, columns)).astype(np.float32)

    def scatter(p, pos, w, e):
        # Each sorted place's token and weight, as moe.routed_experts has them.
        placed = pos.reshape(-1)
        mine = placed >= 0
        token = jnp.repeat(jnp.arange(tokens, dtype=jnp.int32), top_k)
        at = jnp.where(mine, placed, placed.size)
        by_place = [
            jnp.zeros(placed.size + 1, v.dtype).at[at].set(v)[:-1]
            for v in (token, w.reshape(-1))
        ]
        routes = chunk_routes(*by_place, jnp.sum(mine), window, tokens, 8, top_k)
        return combine_scatter(
            p.astype(jnp.bfloat16),
            routes,
            e if later_windows else None,
            RING,
            AXIS,
            top_k=top_k,
            chunk=8,
            interpret=True,
        )

    out = _per_core(
        scatter,
        products,
        position,
        np.broadcast_to(weights, (CORES, tokens, top_k)),
        extra,
    )

    want = extra.sum(0) if later_windows else np.zeros_like(extra[0])
    for c in range(CORES):
        for t, k in zip(*np.nonzero((position[c] >= 0) & (position[c] < window))):
            want[t] += weights[t, k] * products[c, position[c, t, k]]
    np.testing.assert_array_equal(out.reshape(tokens, SLAB, columns), want)
