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
"""The routing plan kernel against the sharded XLA plan, on CPU (interpret).

Every table the expert kernel and the combine read must hold the same
integers as the XLA plan's, on every shard, for skewed routings and for rows
routed nowhere. Only the scale slab's elements past the slab's last row may
differ; nothing reads them.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import layer
from vllm_torchtpu.kernels.fused_moe.v2.host import (
    GatheredPairs,
    act_scale_slab_rows,
    build_routing_tables_sharded,
    expert_visit_list,
    local_slab_rows,
    pair_block_hist,
    ragged_stride_bound,
    routing_block,
    shard_count_vector,
    shard_expert_slabs,
    shard_push_tables_in_rows,
    shard_token_gather,
    shard_transport_tables_in_blocks,
    token_gather_window_rows,
)
from vllm_torchtpu.kernels.fused_moe.v2.plan_kernel import (
    PlanShape,
    plan_kernel_fits,
    routing_plan,
)

EP = 8
TILE_M = 128


def _routing(kind, t_local, topk, e_total, seed):
    rng = np.random.default_rng(seed)
    T = EP * t_local
    if kind == "uniform":
        ids = np.stack([rng.choice(e_total, topk, replace=False) for _ in range(T)])
    elif kind == "skewed":
        # Most pairs on the first shard's first experts.
        hot = np.arange(topk)
        ids = np.stack(
            [
                hot if rng.random() < 0.7 else rng.choice(e_total, topk, replace=False)
                for _ in range(T)
            ]
        )
    else:
        ids = np.stack([rng.choice(e_total, topk, replace=False) for _ in range(T)])
        # Rows that route nowhere carry e_total in every slot.
        ids[rng.random(T) < 0.3] = e_total
    return ids.astype(np.int32)


def _xla_plan(exchange, me, ids, layout, t_local, topk, e_total, block, stride):
    T = EP * t_local
    pairs_g, scale_bits_g, rows_by_dest = layer._unpack_routing_exchange(
        exchange, layout, t_local, topk, e_total
    )
    local_blocks = jnp.asarray(ids[me * t_local : (me + 1) * t_local]).reshape(
        t_local * topk // block, block
    )
    gathered = GatheredPairs(
        expert_blocks=pairs_g.reshape(T * topk // block, block),
        local_blocks=local_blocks,
        block_hist=pair_block_hist(local_blocks, e_total),
        rows_by_dest=rows_by_dest,
        n_tokens=T,
        topk=topk,
    )
    routing = build_routing_tables_sharded(
        None,
        me,
        gathered=gathered,
        e_total=e_total,
        ep=EP,
        t_local=t_local,
        block=block,
        tile_m=TILE_M,
        shard_stride=stride,
    )
    block_tables = shard_transport_tables_in_blocks(routing, me, e_total=e_total, ep=EP)
    row_tables = shard_push_tables_in_rows(routing, me, e_total=e_total, ep=EP)
    slab_row = local_slab_rows(routing, me, shard_stride=stride)
    token_gather = shard_token_gather(routing, me, shard_stride=stride, rows=slab_row)
    token_gather = jnp.pad(token_gather, (0, token_gather_window_rows(TILE_M)))
    scale_bits = jnp.repeat(scale_bits_g, topk)
    scale_slab = (
        jnp.zeros((act_scale_slab_rows(stride) * 128,), jnp.int32)
        .at[slab_row]
        .add(scale_bits, mode="drop")
    )
    expert_rows, slab_base = shard_expert_slabs(routing, me, e_total=e_total, ep=EP)
    visit, _ = expert_visit_list(expert_rows, e_total // EP)
    counts = shard_count_vector(routing, expert_rows, me, e_total=e_total, ep=EP)
    return dict(
        kernel_tables=(*block_tables[:3], block_tables[5], row_tables[0]),
        token_gather=token_gather,
        scale_slab=scale_slab,
        expert_rows=expert_rows,
        slab_base=slab_base,
        visit=visit,
        counts=counts,
        pos=routing.pos,
        mirror_pos=routing.mirror_pos,
    )


def _plan_shape(t_local, topk, e_total):
    layout = layer._exchange_layout(t_local, topk, e_total, True)
    return PlanShape(
        ep=EP,
        e_total=e_total,
        topk=topk,
        t_local=t_local,
        block=routing_block(t_local, topk),
        tile_m=TILE_M,
        shard_stride=ragged_stride_bound(EP * t_local, topk, e_total, TILE_M),
        scale_at=layout[0],
        count_at=layout[1],
        width=layout[2],
    )


def test_plan_kernel_admits_at_most_one_lane_row_of_local_experts():
    """The visit list ranks a shard's experts within one 128-lane row, so a
    shard with more local experts takes the XLA plan."""
    assert plan_kernel_fits(_plan_shape(8, 6, EP * 128))
    assert not plan_kernel_fits(_plan_shape(8, 6, EP * 136))
    assert not plan_kernel_fits(_plan_shape(8, 6, EP * 256))


@pytest.mark.parametrize("kind", ["uniform", "skewed", "nowhere"])
@pytest.mark.parametrize(
    "t_local,topk,e_total",
    [
        (8, 6, 384),
        (16, 6, 384),
        (32, 6, 384),
        (40, 6, 384),
        (16, 3, 128),
        (8, 6, EP * 128),
    ],
)
def test_plan_kernel_matches_the_sharded_xla_plan(kind, t_local, topk, e_total):
    _check_against_the_xla_plan(kind, t_local, topk, e_total, interpret=True)


@pytest.mark.parametrize("kind", ["uniform", "nowhere"])
@pytest.mark.parametrize("t_local,topk,e_total", [(8, 6, 384), (40, 6, 384)])
def test_plan_kernel_lowered_for_tpu_matches_the_sharded_xla_plan(
    kind, t_local, topk, e_total
):
    """The same tables from the kernel compiled through Mosaic: the dynamic
    sublane slice, the bitcasts and the exact float32 matmuls."""
    if jax.default_backend() != "tpu":
        pytest.skip("lowers the plan kernel for TPU; needs a TPU")
    _check_against_the_xla_plan(kind, t_local, topk, e_total, interpret=False)


def _check_against_the_xla_plan(kind, t_local, topk, e_total, *, interpret):
    T = EP * t_local
    shape = _plan_shape(t_local, topk, e_total)
    block, stride = shape.block, shape.shard_stride
    layout = layer._exchange_layout(t_local, topk, e_total, True)
    assert plan_kernel_fits(shape)

    ids = _routing(kind, t_local, topk, e_total, seed=t_local * 31 + topk)
    rng = np.random.default_rng(7)
    scales = rng.uniform(0.01, 4.0, size=T).astype(np.float32)
    rows = []
    for src in range(EP):
        local = jnp.asarray(ids[src * t_local : (src + 1) * t_local]).reshape(
            t_local * topk // block, block
        )
        hist = pair_block_hist(local, e_total)
        bits = jax.lax.bitcast_convert_type(
            jnp.asarray(scales[src * t_local : (src + 1) * t_local]), jnp.int32
        )
        rows.append(
            layer._pack_routing_exchange(local.reshape(-1), bits, hist.sum(0), layout)
        )
    exchange = jnp.stack(rows)

    for me in range(EP):
        want = _xla_plan(
            exchange, me, ids, layout, t_local, topk, e_total, block, stride
        )
        got = routing_plan(
            exchange, jnp.full((1, 1), me, jnp.int32), shape, interpret=interpret
        )
        for k, (g, w) in enumerate(zip(got.kernel_tables, want["kernel_tables"])):
            np.testing.assert_array_equal(g, w, err_msg=f"shard {me} table {k}")
        np.testing.assert_array_equal(got.expert_rows, want["expert_rows"])
        np.testing.assert_array_equal(got.slab_base, want["slab_base"])
        np.testing.assert_array_equal(got.visit, want["visit"])
        np.testing.assert_array_equal(got.counts, want["counts"])
        np.testing.assert_array_equal(got.pos, want["pos"])
        np.testing.assert_array_equal(got.mirror_pos, want["mirror_pos"])
        np.testing.assert_array_equal(got.token_gather, want["token_gather"])
        np.testing.assert_array_equal(
            jax.lax.bitcast_convert_type(got.scale_slab, jnp.int32).reshape(-1)[
                :stride
            ],
            want["scale_slab"][:stride],
        )
