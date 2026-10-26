# SPDX-License-Identifier: Apache-2.0
"""The decode routing plan kernel against the XLA sharded plan, on the host.

plan_kernel.build_decode_plan runs in Pallas interpret mode here, so its
integer bookkeeping is checked without a chip; Mosaic lowering is covered by
the device test test_single_host_decode_glue_is_bitexact.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.fused_moe.v2 import host  # noqa: E402
from vllm_torchtpu.kernels.fused_moe.v2.plan_kernel import (  # noqa: E402
    build_decode_plan,
)

TILE_M = 128


@pytest.fixture(autouse=True)
def _cpu_backend():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _xla_plan(pairs_g, scale_g, me, e_total, ep, t_local, topk, row_stride):
    T = ep * t_local
    block = host.routing_block(t_local, topk)
    stride = host.ragged_stride_bound(T, topk, e_total, TILE_M)
    local_blocks = pairs_g[me].reshape(-1, block)
    gathered = host.GatheredPairs(
        expert_blocks=pairs_g.reshape(T * topk // block, block),
        local_blocks=local_blocks,
        block_hist=host.pair_block_hist(local_blocks, e_total),
        rows_by_dest=host.pair_block_hist(pairs_g, e_total),
        n_tokens=T,
        topk=topk,
    )
    r = host.build_routing_tables_sharded(
        None,
        me,
        gathered=gathered,
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=block,
        tile_m=TILE_M,
        shard_stride=stride,
    )
    bt = host.shard_transport_tables_in_blocks(r, me, e_total=e_total, ep=ep)
    rt = host.shard_push_tables_in_rows(r, me, e_total=e_total, ep=ep)
    rows = host.local_slab_rows(r, me, shard_stride=stride)
    gather = host.shard_token_gather(
        r, me, shard_stride=stride, rows=rows, source_rows=(t_local, row_stride)
    )
    scale_rows = host.act_scale_slab_rows(stride)
    scale = host.slab_scatter_add(
        scale_rows * host.HIDDEN_LANE_BLOCK, rows, jnp.repeat(scale_g, topk)
    )
    expert_rows, slab_base = host.shard_expert_slabs(r, me, e_total=e_total, ep=ep)
    visit, _ = host.expert_visit_list(expert_rows, e_total // ep)
    return dict(
        tables=jnp.stack([bt[0], bt[1], bt[2], bt[5], rt[0]]),
        token_gather=jnp.pad(gather, (0, host.token_gather_window_rows(TILE_M))),
        scale_slab=scale.reshape(scale_rows, host.HIDDEN_LANE_BLOCK),
        expert_rows=expert_rows,
        slab_base=slab_base,
        visit=visit,
        counts=host.shard_count_vector(r, expert_rows, me, e_total=e_total, ep=ep),
        pos=r.pos,
        mirror_pos=r.mirror_pos,
    ), (block, stride)


def _pairs(rng, kind, e_total, ep, t_local, topk):
    """[ep, t_local * topk] selections: distinct experts per token."""
    g_local = e_total // ep
    pool = {
        "uniform": np.arange(e_total),
        "unrouted": np.arange(e_total),
        "hot": rng.choice(e_total, max(topk, 8), replace=False),
        "one_shard": np.arange(g_local) + 3 % ep * g_local,
    }[kind]
    rows = np.stack(
        [rng.choice(pool, topk, replace=False) for _ in range(ep * t_local)]
    )
    if kind == "unrouted":
        rows[rng.random(len(rows)) < 0.4] = e_total
    return jnp.asarray(rows.reshape(ep, t_local * topk), jnp.int32)


@pytest.mark.parametrize("kind", ["uniform", "hot", "one_shard", "unrouted"])
@pytest.mark.parametrize(
    "e_total,ep,t_local,topk",
    [(384, 16, 2, 6), (384, 16, 1, 6), (128, 8, 4, 4)],
)
def test_decode_plan_matches_sharded_plan(kind, e_total, ep, t_local, topk):
    rng = np.random.default_rng(hash((kind, e_total, ep, t_local)) % 2**32)
    pairs = _pairs(rng, kind, e_total, ep, t_local, topk)
    scale = jnp.asarray(rng.integers(0, 2**31 - 1, ep * t_local), jnp.int32)
    for row_stride in (t_local, t_local + 1):
        for me in (0, ep - 1):
            want, (block, stride) = _xla_plan(
                pairs, scale, me, e_total, ep, t_local, topk, row_stride
            )
            plan = build_decode_plan(
                pairs,
                scale,
                jnp.int32(me),
                e_total=e_total,
                ep=ep,
                t_local=t_local,
                topk=topk,
                block=block,
                tile_m=TILE_M,
                shard_stride=stride,
                row_stride=row_stride,
                interpret=True,
            )
            got = dict(
                tables=jnp.stack([*plan.block_tables, plan.true_rows]),
                token_gather=plan.token_gather,
                scale_slab=jax.lax.bitcast_convert_type(plan.scale_slab, jnp.int32),
                expert_rows=plan.expert_rows,
                slab_base=plan.slab_base,
                visit=plan.visit,
                counts=plan.counts,
                pos=plan.pos,
                mirror_pos=plan.mirror_pos,
            )
            for name in want:
                np.testing.assert_array_equal(
                    np.asarray(got[name]), np.asarray(want[name]), err_msg=name
                )


@pytest.mark.parametrize("n,length", [(12, 768), (192, 896)])
def test_onehot_slab_scatter_matches_scatter(n, length):
    rng = np.random.default_rng(n)
    rows = jnp.asarray(rng.integers(0, length + 200, n), jnp.int32)
    values = jnp.asarray(rng.integers(-(2**31), 2**31 - 1, n), jnp.int32)
    np.testing.assert_array_equal(
        np.asarray(host.slab_scatter_add(length, rows, values, one_hot=True)),
        np.asarray(host.slab_scatter_add(length, rows, values)),
    )
