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
"""The activation-scale slab the lookup kernel builds is the scatter's.

The layer builds the slab by looking each live row's scale up through the
token-gather table (`kernel.build_act_scale_slab`) wherever the per-token scales
fit in SMEM, and by scattering every routed pair's scale only beyond that. The
two agree on every row a routed pair computes on; they differ only on the
ROWBLK padding rows inside the live range, which the lookup fills with token
0's scale and the scatter leaves at zero, and past the slab's last row. The
kernel commits neither, which is what the layer-level bit-exactness tests
check end to end; this checks the table itself, shard by shard.

The routing plan is built on the host CPU; only the lookup runs on the chip.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.fused_moe.v2 import host, kernel  # noqa: E402

pytestmark = pytest.mark.tpu_test

# (ep, t_local, g_local, topk): Qwen3.8's EP32 layout at a decode and a prefill
# step, and the 397B layout over eight shards.
GEOMETRIES = [
    pytest.param(32, 48, 16, 10, id="qwen38_decode_t1536"),
    pytest.param(32, 512, 16, 10, id="qwen38_prefill_t16384"),
    pytest.param(8, 256, 64, 10, id="ep8_g64_t2048"),
]


def _plan(ep, t_local, g_local, topk, routing, seed):
    """Routing indices, the replicated plan and the builder's arguments."""
    e_total = g_local * ep
    T = t_local * ep
    rng = np.random.default_rng(seed)
    if routing == "skewed":
        hot = rng.choice(e_total, size=max(4, e_total // 12), replace=False)
        idx = rng.choice(hot, size=(T, topk))
    else:
        idx = rng.integers(0, e_total, (T, topk))
    if routing == "padded":
        # The layer's padding rows: every selection set to e_total. The second
        # half of each shard's rows, and all of shard 1's.
        for d in range(ep):
            start = d * t_local if d == 1 else d * t_local + t_local // 2
            idx[start : (d + 1) * t_local, :] = e_total
    capacity = 128
    kwargs = dict(
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=host.routing_block(t_local, topk),
        tile_m=capacity,
        shard_stride=host.ragged_stride_bound(T, topk, e_total, capacity),
    )
    with jax.default_device(jax.devices("cpu")[0]):
        idx = jnp.asarray(idx.astype(np.int32))
        plan = host.build_routing_tables(idx, **kwargs)
    return plan, kwargs


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("routing", ["uniform", "skewed", "padded"])
def test_lookup_scale_slab_matches_the_scatter(ep, t_local, g_local, topk, routing):
    plan, kwargs = _plan(ep, t_local, g_local, topk, routing, seed=5)
    e_total = kwargs["e_total"]
    stride = kwargs["shard_stride"]
    T = ep * t_local
    assert kernel.scale_slab_fits(T)
    rng = np.random.default_rng(6)
    # Row-scale bit patterns as the routing exchange carries them: one
    # positive f32 per token, so a live row's bits are never zero.
    scale_bits = (rng.random(T, dtype=np.float32) + 0.5).view(np.int32)
    scale_rows = host.act_scale_slab_rows(stride)
    view_rows = scale_rows * host.HIDDEN_LANE_BLOCK
    table_rows = host.align_up(stride, kernel.SCALE_SLAB_WINDOW)
    lookup = kernel.build_act_scale_slab(
        n_tokens=T,
        table_rows=table_rows,
        out_rows=host.align_up(view_rows, kernel.SCALE_SLAB_WINDOW),
    )

    for me in sorted({0, 1, ep // 2, ep - 1}):
        with jax.default_device(jax.devices("cpu")[0]):
            rows = host.local_slab_rows(plan, me, shard_stride=stride)
            token_gather = np.asarray(
                host.shard_token_gather(plan, me, shard_stride=stride, rows=rows)
            )
            scattered = np.asarray(
                host.slab_scatter(
                    view_rows,
                    rows,
                    jnp.asarray(np.repeat(scale_bits, topk)),
                    by_set=False,
                )
            )
            live = int(
                np.asarray(
                    host.shard_expert_slabs(plan, me, e_total=e_total, ep=ep)[0]
                ).sum()
            )
        got = np.asarray(
            lookup(
                jnp.int32(live),
                jnp.asarray(np.pad(token_gather, (0, table_rows - stride))),
                jnp.asarray(scale_bits),
            )
        )[:view_rows]

        computed = scattered[:stride] != 0  # the rows a routed pair lands on
        assert not computed[live:].any(), f"shard {me}: a pair lands past live"
        np.testing.assert_array_equal(
            got[:stride][computed],
            scattered[:stride][computed],
            err_msg=f"shard {me}: a computed row's scale differs",
        )
        np.testing.assert_array_equal(
            got[live:],
            np.zeros(view_rows - live, np.int32),
            err_msg=f"shard {me}: rows past the live range are not cleared",
        )
        # The padding rows inside the live range read token 0's scale.
        padding = ~computed[:live]
        np.testing.assert_array_equal(
            got[:live][padding],
            np.full(padding.sum(), scale_bits[0], np.int32),
            err_msg=f"shard {me}: a padding row is not token 0's scale",
        )
